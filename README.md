# PPO_ASYNC

PPO_ASYNC compares three reinforcement-learning systems for Lean 4 theorem
proving with `Qwen/Qwen3.5-4B`:

1. synchronous, on-policy PPO;
2. asynchronous PPO with continuously replenished SGLang requests;
3. the same asynchronous PPO system with Direct Double-Sided Importance
   Sampling (DIS).

The implementation uses pinned SLIME v0.3.2 and SGLang on one Modal node:

| Track | H200s | Placement |
| --- | --- | --- |
| Synchronous | 1 | Generation, critic, then actor take turns on GPU 0 |
| Asynchronous / DIS | 2 | Critic and actor take turns on GPU 0; generation runs on GPU 1 |

Inactive training models are offloaded to CPU. Sync also offloads SGLang before
training or saving. Sync enables SGLang CPU weight backup so checkpoint
release/resume restores the published weights, not just their storage. Critic completion is awaited before the actor wakes, so the
shared training GPU never runs both models together. Modal requests strict H200s
and 192 GiB of host memory for the offloaded models and optimizer states.
This allocation reduces reserved GPU time; measured throughput still depends on
proof lengths, CPU transfer overhead and verification time. Full 16k memory
headroom must be checked on the target runtime, not inferred from CPU tests.

## Data contract

The final curriculum contains 600 unique problems, each visited once with
a freshly generated proof (600 total training attempts):

- the first 300 retained `internlm/Lean-Workbook` rows, after requiring
  `status == "proved"`;
- the first 300 retained `marcusm117/ProofNet-Verified` rows.

Evaluation uses only:

- Gaokao-Formal;
- FATE-M.

Every source and model revision is pinned. Training preparation removes exact
duplicates and exact evaluation-statement matches after deterministic
source-text normalization. It does not use embeddings, fuzzy matching, AST
equivalence, or compilation results to filter data. Reference `tactic` and
`answer` fields are never written to SLIME prompt files.

The datasets retain their native Lean environments:

- Lean-Workbook: Lean/Mathlib `4.8.0-rc1`;
- ProofNet-Verified and FATE-M: Lean/Mathlib `4.28.0`;
- Gaokao-Formal: Lean/Mathlib `4.27.0`.

Prepare the pinned source artifacts:

```bash
uv run prepare-lean-workbook
uv run prepare-eval-datasets all
```

Prepare the bounded 16-example smoke curriculum and both complete evaluation
prompt files:

```bash
uv run prepare-ppo-data \
  --training-examples 16 \
  --output-root artifacts/prompts
```

The smoke curriculum contains eight Lean-Workbook and eight ProofNet-Verified
problems. The command refuses limits of 100 or more.

Prepare the immutable 600-example final curriculum and both complete
evaluation suites:

```bash
uv run prepare-ppo-data \
  --mode final \
  --output-root prepared_data/balanced-600-v3
```

`data-report.json` records content hashes, source counts, exclusions, and the
two-pass training contract. The prepared prompt files contain no reference
proof fields.

## Training loops

All arms use the same actor, critic, sparse kernel-verification reward, prompt
stream, optimizer, batch sizes, and two SGLang engines. SGLang behavior-token
log-probabilities are stored on each trajectory. The standard PPO arms use them
as PPO's old policy; DIS uses SLIME's TIS path with a recomputed learner-policy
baseline. This baseline difference remains part of the current DIS comparison.
Prompt, padding, and non-action tokens are excluded from policy/value losses.
Verifier infrastructure failures remove a trajectory rather than assigning a
zero reward.

Training rollouts and in-run evaluation permit up to 16,384 response tokens.
The 4,096-token dynamic packing target is not a response cap: SLIME places
longer individual samples alone in a microbatch. Peak memory for 16k responses
still requires validation on the GPU integration run.

Training and standalone evaluation share `lean.py` and `Verifier.lean`. The
candidate is parsed as exactly one Lean term. After elaboration, a trusted
command traverses the theorem's transitive axiom dependencies and permits only
`propext`, `Classical.choice`, and `Quot.sound`. Direct `sorryAx` is also
rejected before compilation. A zero compiler exit code without a successful
axiom-audit marker is rejected.

### Synchronous PPO

The actor weights remain fixed while a complete batch is generated and Lean
verified. Actor and critic update only after the batch finishes; the new actor
weights are then published to the SGLang engine.

### Asynchronous PPO

`training/stream.py` keeps eight generation/verification tasks alive across
learner updates. Each completed proof enters a queue capped at eight; freed
workers immediately reserve another prompt. The learner takes the first eight
completions, so a long proof cannot hold its original cohort up. Backpressure
bounds unconsumed work to eight queued proofs plus eight worker-held proofs.
There is no next-batch prefetch barrier in the driver.

Actor weights are published after every update using SLIME's native SGLang
pause/retract/transfer/resume path. Requests can span weight versions; actual
rollout token log probabilities remain the behavior-policy denominator. SGLang
returns a final version label rather than a version for every token, so request
submission age is logged explicitly as a **conservative upper bound** and checked
at learner dequeue. Proofs older than four updates are regenerated using the
same logical prompt ID. After three retries, the run fails instead of silently
skipping data or consuming unlimited compute.

Checkpoints save the source cursor together with original prompts for every
outstanding attempt, including completed but unconsumed proofs. On retry those
attempts are regenerated; already committed training examples are not replayed.
The producer never reserves more than the 600 logical attempts. Completion
order is intentionally nondeterministic, while the two-pass prompt multiset is
preserved. Generation continues during checkpoints and evaluation, subject to
queue backpressure. Smoke uses exactly this production scheduler.

### Asynchronous PPO + DIS

This arm uses the asynchronous schedule and SLIME's TIS path with a custom actor-gradient mask.
For each action token:

```text
ratio = exp(logp_current - logp_rollout)
keep  = (ratio > 1 - epsilon_low) and (ratio < 1 + epsilon_high)
```

The frozen smoke bounds are `epsilon_low=0.3` and `epsilon_high=5.0`. Standard
PPO clipping is applied first. DIS then zeros the actor-loss numerator for
out-of-region tokens while retaining the original valid-action denominator.
Ratio distributions and masked-token fractions are logged separately from PPO
clip fraction.

## Local validation

```bash
uv run pytest -q
git diff --check
```

The test suite checks all three command paths, shared-GPU handoffs, completion ordering, bounded queues and recovery, exact
data filtering, proof non-leakage, action-only GAE, behavior-policy accounting,
asymmetric open-interval DIS masking, and the training budget. Real verifier
regressions can run against installed Lean binaries in all three versions:

```bash
PPO_ASYNC_TEST_LEAN_BINARIES=/path/to/lean48:/path/to/lean427:/path/to/lean428 \
  uv run pytest -q tests/test_lean_verifier.py
```

## GPU smoke run

This integration smoke processes 16 training examples in two batches of 8:

```bash
uv run modal run modal_train.py \
  --arm sync-ppo \
  --training-examples 16 \
  --num-rollouts 2 \
  --run-name smoke-sync-ppo-16-16k-v3
```

The remote function attests one H200 for sync or two for async before downloading or training,
materializes the exact prompt set, converts the pinned checkpoint, starts Ray,
launches one actor, one critic, and one SGLang engine, performs the PPO updates,
and writes commands, hardware inventory, trajectories, metrics, checkpoints,
Hugging Face exports, and a completion report to `ppo-async-artifacts`.

To exercise another arm after the first integration gate passes, change
`--arm` to `async-ppo` or `async-ppo-dis` and use a fresh run name.

## Final integration smoke

Run `modal run --detach modal_train.py --mode integration-smoke --arm sync-ppo
--run-name <fresh-name>` (or `--arm async-ppo-dis`) to exercise the production
path with 16 training problems, two updates/checkpoints, a fixed 32-problem
selection evaluation after each checkpoint, and 64 final evaluation problems
on the winning HF export. Training/evaluation concurrency remains 8/16 and both
response caps remain 16,384 tokens. Production settings are unchanged.

## Full final runs

All final arms use rollout batch 8 and global learner batch 8 for exactly 75
updates (600 processed examples, one pass). Scalar rollout metrics use exact
10-example windows. Checkpoints and HF exports are saved at 200, 400, and 600
processed examples, followed by pass@1 evaluation on the same fixed 200 problems
(100 Gaokao-Formal and 100 FATE-M, selected by seeded statement hash).

The highest aggregate selection pass@1 wins; ties select the earlier checkpoint.
After training, a fresh inference process loads that checkpoint's HF export and
runs one final pass@1 evaluation on all 645 benchmark problems. These final sets
include the 200 selection problems, so their aggregate score is not an untouched
holdout estimate. Every evaluation uses one response per problem and a 16,384-token
cap. Evaluation has an independent concurrency and HTTP pool limit of 16;
training retains concurrency 8. SGLang allows at most 16 running requests in total,
so concurrent async training/eval requests can queue at the server.

Every recovery boundary contains actor, critic, optimizer, dataset cursor (plus
async outstanding prompts), selection scores, and an HF export. The transaction
is committed only after the checkpoint and selection evaluation both succeed.
Only the newest resumable training state is retained; all three HF exports remain
available for selection. Results are recorded in `evaluations.jsonl`,
`best-checkpoint.json`, and `final-evaluation.jsonl`. Retries resume from the last
complete transaction without counting a failed partial tail twice.

Deploy the production function once, then use the final-run launcher. `--arm
all` submits three independent persistent function calls and prints all call
IDs. A CPU dispatcher serializes production runs, selecting the one- or two-GPU
worker for each arm while the remaining calls stay queued. The launcher first
runs a CPU-only parse preflight for all pinned SLIME/Megatron commands:

```bash
uv run modal deploy modal_train.py
uv run python launch_final.py \
  --arm all \
  --run-prefix balanced600-1pass-qwen35-4b-h200-b8-v4
```

The arm name is appended to each artifact directory. A single arm can also be
submitted by passing `sync-ppo`, `async-ppo`, or `async-ppo-dis`. Without
`--arm`, the launcher selects DIS and synchronous PPO. Use fresh run names for
this experiment; checkpoints from the old curriculum are incompatible.

## Standalone evaluation

[`modal_benchmark.py`](modal_benchmark.py) is the frozen pass@1 evaluator. It
supports both required suites and can load the Hugging Face checkpoint exported
by any training arm from the shared artifact volume:

```bash
uv run modal run modal_benchmark.py \
  --dataset-name gaokao-formal \
  --checkpoint-path /training-artifacts/<run-name>/hf-<rollout-id> \
  --run-name qwen35-4b-gaokao-pass1

uv run modal run modal_benchmark.py \
  --dataset-name fate-m \
  --checkpoint-path /training-artifacts/<run-name>/hf-<rollout-id> \
  --run-name qwen35-4b-fate-m-pass1
```

It compiles every selected statement before allocating the generation GPU,
generates exactly one proof per theorem with SGLang, and verifies candidates in
the benchmark's pinned Lean environment. Evaluation output never enters the
training buffer.

GPU integration runs record `gpu-utilization.csv` every second alongside
`events.jsonl`. Compare GPU-seconds per trained example, learner time and stream
wait time after startup; smaller GPU allocations do not by themselves prove
higher throughput. The DIS arm still uses masked PPO through SLIME's TIS path;
this is an asynchronous PPO comparison, not a full reproduction of SAO's critic
training and direct objective.

The actor vocabulary softmax is checkpointed per 256-token chunk via SLIME’s
worker-init hook, with loss recomputation enabled. This avoids retaining a
full-sequence softmax alongside the logits at 16k response length. The OOM
regression preserves log-probabilities and gradients; GPU smoke jobs provide
the remaining peak-memory validation.
