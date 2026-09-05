/- Trusted verifier support, inserted after the dataset's imports/context.
   Candidate text is parsed as exactly one term, never as Lean commands. -/
open Lean Elab in
elab "ppo_proof " proof:str : term <= expectedType => do
  let parsed ← match Parser.runParserCategory (← getEnv) `term proof.getString with
    | .ok parsed => pure parsed
    | .error message => throwError "invalid proof expression: {message}"
  Term.elabTerm parsed expectedType

namespace PPOAsyncVerifier

open Lean

structure AuditState where
  visited : NameSet := {}
  axioms : Array Name := #[]

-- Inspect both types and values, including transitive dependencies. This uses
-- the same constant traversal as Lean's #print axioms, but enforces the result.
partial def collect (env : Environment) (name : Name) : StateM AuditState Unit := do
  unless (← get).visited.contains name do
    modify fun state => { state with visited := state.visited.insert name }
    let visit (expr : Expr) := expr.getUsedConstants.forM (collect env)
    match env.find? name with
    | some (.axiomInfo _) =>
        modify fun state => { state with axioms := state.axioms.push name }
    | some (.defnInfo value) => visit value.type *> visit value.value
    | some (.thmInfo value) => visit value.type *> visit value.value
    | some (.opaqueInfo value) => visit value.type *> visit value.value
    | some (.ctorInfo value) => visit value.type
    | some (.recInfo value) => visit value.type
    | some (.inductInfo value) => visit value.type *> value.ctors.forM (collect env)
    | some (.quotInfo _) => pure ()
    | none => pure ()

open Lean Elab Command in
elab "ppo_audit " target:ident nonce:str : command => do
  let names ← liftCoreM <| realizeGlobalConstWithInfos target
  unless names.length == 1 do
    throwError "expected exactly one theorem"
  let env ← getEnv
  let name := names.head!
  match env.find? name with
  | some (.thmInfo _) => pure ()
  | _ => throwError "verification target is not a theorem"
  let (_, state) := (collect env name).run {}
  let allowed := #[`propext, `Classical.choice, `Quot.sound]
  let forbidden := state.axioms.filter fun dependency => !allowed.contains dependency
  unless forbidden.isEmpty do
    throwError "unapproved proof axioms: {forbidden.toList}"
  logInfo m!"PPO_ASYNC_VERIFIED {nonce.getString}"

end PPOAsyncVerifier
