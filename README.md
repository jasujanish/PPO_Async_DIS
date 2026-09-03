# PPO_ASYNC

PPO_ASYNC compares three reinforcement-learning systems for Lean 4 theorem
proving with `Qwen/Qwen3.5-4B`:

1. synchronous, on-policy PPO;
2. asynchronous PPO with continuously replenished SGLang rollouts;
3. the same asynchronous PPO system with Direct Double-Sided Importance
   Sampling (DIS).

The implementation uses pinned SLIME v0.3.2 and SGLang. It is a true
four-GPU system on one Modal node:

| Physical GPU | Role |
| --- | --- |
| H100 0 | actor/learner |
| H100 1 | critic |
| H100 2 | SGLang rollout engine 0 |
| H100 3 | SGLang rollout engine 1 |

The custom SLIME driver assigns non-overlapping Ray placement-group slices to
these roles. `--colocate` is forbidden. Modal requests `H100!:4`, so H200
substitution cannot silently change the experiment.

## Data contract

The final one-pass curriculum contains:

- the first 1,633 retained `internlm/Lean-Workbook` rows, after requiring
  `status == "proved"`;
- the first 367 retained `marcusm117/ProofNet-Verified` rows.

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

Prepare the immutable 2,000-example final curriculum and both complete
evaluation suites:

```bash
uv run prepare-ppo-data \
  --mode final \
  --output-root prepared_data/final-v1
```

`data-report.json` records content hashes, source counts, exclusions, and the
one-pass selection contract. The prepared prompt files contain no reference
proof fields.

## Training loops

All arms use the same actor, critic, sparse kernel-verification reward, prompt
stream, optimizer, batch sizes, and two SGLang engines. SGLang behavior-token
log-probabilities are stored on each trajectory and used as PPO's old policy.
Prompt, padding, and non-action tokens are excluded from policy/value losses.
Verifier infrastructure failures remove a trajectory rather than assigning a
zero reward.

### Synchronous PPO

The actor weights remain fixed while a complete batch is generated and Lean
verified. Actor and critic update only after the batch finishes; the new actor
weights are then published atomically to both SGLang engines.

### Asynchronous PPO

The asynchronous driver overlaps learner work with exactly one next rollout
batch. At checkpoint/evaluation thresholds it stops prefetching and drains the
current batch before publishing weights. This keeps the global dataset cursor
identical to the number of trained examples and makes retries exact. Weight
versions are immutable within each trajectory. The configured publish cadence
permits at most one learner-update of policy lag; older trajectories are
loss-masked and recorded.

### Asynchronous PPO + DIS

This arm is identical to asynchronous PPO except for the actor-gradient mask.
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

The test suite checks all three command paths, physical GPU isolation, exact
data filtering, proof non-leakage, action-only GAE, behavior-policy accounting,
asymmetric open-interval DIS masking, and the paid-run example bound.

## Four-H100 smoke run

The authorized integration smoke uses 16 training examples and one rollout:

```bash
uv run modal run modal_train.py \
  --arm sync-ppo \
  --training-examples 16 \
  --num-rollouts 1 \
  --run-name smoke-sync-ppo-16-v1
```

The remote function attests four visible H100s before downloading or training,
materializes the exact prompt set, converts the pinned checkpoint, starts Ray,
launches one actor, one critic, and two SGLang engines, performs the PPO update,
and writes commands, hardware inventory, trajectories, metrics, checkpoints,
Hugging Face exports, and a completion report to `ppo-async-artifacts`.

To exercise another arm after the first integration gate passes, change
`--arm` to `async-ppo` or `async-ppo-dis` and use a fresh run name.

## Full final runs

All final arms use rollout batch 8 and global learner batch 8 for exactly 250
updates (2,000 processed examples). Scalar rollout metrics are aggregated into
exact 10-example windows. Because batch 8 does not divide 100, checkpoint,
evaluation, and export actions run after the first completed batch crossing
each 100-example threshold: 104, 200, 304, 400, and so on through 2,000.

Every action boundary contains a resumable actor, critic, optimizer, and global
dataset cursor checkpoint; complete Gaokao-Formal and FATE-M pass@1 records;
and a Hugging Face actor export. Actor/critic training state retains the newest
complete boundary while all scheduled Hugging Face exports and compact metric
records remain available. Modal retries resume only from a complete transaction
marker, so a failed partial tail is not counted twice.

Deploy the production function once, then use the final-run launcher. `--arm
all` submits three independent persistent function calls and prints all call
IDs. A shared container limit runs one four-H100 arm at a time, keeping total
use at four GPUs while the remaining calls stay queued. The launcher first
runs a CPU-only parse preflight for all pinned SLIME/Megatron commands:

```bash
uv run modal deploy modal_train.py
uv run python launch_final.py \
  --arm all \
  --run-prefix final-qwen35-4b-b8-v1
```

The arm name is appended to each artifact directory. A single arm can also be
submitted by passing `sync-ppo`, `async-ppo`, or `async-ppo-dis`.

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
