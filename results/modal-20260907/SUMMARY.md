# Experiment 2: Async PPO + DIS Masking

Run: `experiment2-dis-masking-balanced400-h200-v1`  
[Modal run](https://modal.com/apps/jasujanish/main/ap-FuLJI5i06KIX1mFesxG96G)

Completed successfully on 2026-09-07. The completion record reports no resume,
and the downloaded artifacts contain one training command attempt.

| Measurement | Result |
| --- | ---: |
| Gaokao-Formal pass@1 | 7/495 |
| FATE-M pass@1 | 4/150 |
| Overall pass@1 | **11/645 (1.71%)** |
| Training problems / actor updates | 400 / 50 |
| Verified training proofs | 42/400 (10.50%) |
| Mean response length | 4,349.59 tokens |
| Training truncation rate | 13.00% |
| Time inside learner updates | 50.43 minutes |
| Total reported runtime, including evaluations | 168.60 minutes |
| Allocated GPUs | 2 H200s |

The checkpoint after 200 problems scored 0/200 on the selection set; the
checkpoint after 400 scored 2/200 and was selected for the full evaluation.
The final set includes the selection set. Compared with the earlier results,
this is three more verified proofs than Async PPO, one fewer than the base
model, and five fewer than baseline-policy masking. Single runs with these
small solved counts do not demonstrate a statistically reliable improvement.

## Verification and provenance

- Completion record, final evaluation, and best-checkpoint record agree.
- Exactly 50 learner steps and 400 unique consumed trajectory IDs are recorded.
- All 40 scalar windows and both scheduled selection evaluations are present.
- Training and evaluation dataset hashes match all three earlier runs.
- The recorded command selects `direct_dis_policy_loss` and rollout log probabilities.
- Downloaded 20 result/configuration/tracking files (1,269,032 bytes); sizes were
  checked against Modal listings, and local SHA-256 digests are in the manifest.
- Model weights and optimizer checkpoints remain on Modal and were not downloaded.

[Machine-readable summary](summary.json) · [Download manifest](download-manifest.json) ·
[Completion record](experiment2-dis-masking-balanced400-h200-v1/RUN_COMPLETE.json) ·
[Final evaluation](experiment2-dis-masking-balanced400-h200-v1/final-evaluation.jsonl)
