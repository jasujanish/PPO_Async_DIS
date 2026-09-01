# OPSD

## Research idea

- One major issue with OPD is teacher-student mismatch If the teacher is much stronger than the student, teacher's reasoning traces have little overlap with the student's reasoning traces, so teacher capabilities fail to distill into the student.
- In SAO, DIS zeros gradeints when the ratio between the current policy and rollout policy is too extreme.
- What if we used a DIS inspired clipping to reduce insability in OPD?
- I don't have all the details worked out yet, but I find this interesting.
- We could also try this in OPSD.

## Dataset preparation

This project prepares the
[`internlm/Lean-Workbook`](https://huggingface.co/datasets/internlm/Lean-Workbook)
dataset for downstream work. It keeps every source column but retains only rows
whose `status` is exactly `proved`.

## Setup

```bash
uv sync
```

## Prepare the dataset

```bash
uv run prepare-lean-workbook
```

The filtered Hugging Face dataset is written to
`data/lean-workbook-proved/`. That directory can be loaded later with:

```python
from datasets import load_from_disk

dataset = load_from_disk("data/lean-workbook-proved")
```

Choose a different location or source split if needed:

```bash
uv run prepare-lean-workbook --split train --output-dir path/to/output
```

The source is pinned to Hugging Face revision
`2e066e310b2c6d2c27616927ae131f82901c8f1c` by default, and preparation writes
the source revision and filter into `opsd_metadata.json` beside the saved
dataset.

The command validates the source schema and its result. It fails rather than
silently producing invalid data if the `status` column is missing or any
non-`proved` row survives the filter.

## Test

```bash
uv run pytest
```

## Qwen3.5-4B benchmarks on Modal

The harness supports four independently reported benchmarks:

- `lean-workbook` (Lean 4.8.0-rc1; 10,434 unique proved theorems)
- `fate-m` (Lean 4.28.0; 150 theorems)
- `gaokao-formal` (Lean 4.27.0; 495 theorems)
- `proofnet-verified` (Lean 4.28.0; 367 theorems)

All prepared artifacts have the exact eight-column Lean-Workbook schema. Source
revisions, source checksums, Arrow checksums, row counts, ordering, and unique IDs
are validated before use. Dataset-specific imports, namespace openings, options,
and helper declarations are carried in `state_before` and supplied unchanged to
both the prompt and verifier. Natural-language comments and reference proof fields
are excluded from prompts.

FATE-M's JSON omits the `open` and `open scoped` directives present in 22 of
the pinned official `FATEM/<id>.lean` exercise files. Preparation restores those
exact per-exercise directives in `state_before` and records their source revision;
without them, otherwise valid statements using polynomial, conjugation, pointwise,
opposite-group, or classical notation do not elaborate.

Gaokao-Formal's published statements contain 144 Lean-3-style finite-set big
operator binders such as `∑ i in s, ...`. Preparation records and applies the
syntax-only Lean 4.27 normalization `in` to `∈` in those binders; no theorem term,
type, or proof is otherwise changed.

Each run uses one Modal H100 and one SGLang server to generate exactly one
sampled proof candidate per selected theorem. It uses
Qwen3.5's recommended thinking-mode profile for precise coding tasks:
`temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0`,
`presence_penalty=0`, and `repetition_penalty=1`. Thinking is explicitly enabled,
the completion limit is 16,384 tokens, and the server context limit is 32,768.

SGLang is configured for up to 256 active requests and the client maintains a
512-request backlog. The server may lower the active-request count when its
Mamba-state cache is the limiting resource. GPU utilization, power, memory, and
temperature are sampled every five seconds into the result artifact.

A dataset-specific CPU function first groups selected placeholder theorems by
their exact preamble and compiles bounded chunks before any H100 request is sent.
Each statement receives a private preflight namespace, while Mathlib is imported
once per chunk instead of once per theorem. Candidate proofs remain independently
verified after generation. Successful preflights report the preamble count,
chunk count, and elapsed time. Benchmark data is copied only into the final image
layer, so changing an Arrow artifact no longer invalidates the expensive Mathlib
build.

Use `--preflight-only` to validate statements without requesting an H100 or
generating model samples.

Lean-Workbook contains two malformed stored declarations in the first 50 unique
theorems. The loader applies exact-hash-guarded repairs to
`lean_workbook_plus_56` (unresolved real trigonometric identifiers) and
`lean_workbook_plus_246` (missing proposition colon); the normalization counts
are recorded in dataset provenance and every run configuration.

Run it once with:

```bash
uv run modal run modal_benchmark.py \
  --dataset-name fate-m \
  --limit 100 \
  --run-name qwen35-4b-fate-m-thinking-16k-first100-pass1
```

Completed generation or verification cannot be repeated. If a container is
interrupted, the app validates the saved configuration and resumes only missing
theorem indices, so completed model requests are not regenerated. Results are
written to the Modal Volume `opsd-benchmark-results` under
`qwen35-4b-lean-workbook-thinking-16k-pass1/`.
