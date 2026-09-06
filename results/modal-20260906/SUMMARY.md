All three runs completed successfully. Both training runs completed 50 updates on 400 unique logical attempts. Dataset hashes match across all three runs.

| Model | Gaokao-Formal (495) | FATE-M (150) | Overall pass@1 (645) | Selected checkpoint | Reported elapsed time |
|---|---:|---:|---:|---|---|
| Base | 6/495 (1.21%) | 6/150 (4.00%) | 12/645 (1.86%) | Base | 1h 05m |
| Async PPO | 5/495 (1.01%) | 3/150 (2.00%) | 8/645 (1.24%) | 200 | 3h 52m |
| Async PPO + DIS | 10/495 (2.02%) | 6/150 (4.00%) | 16/645 (2.48%) | 400 | 2h 23m |

Async PPO used the checkpoint after 200 training problems; PPO + DIS used the checkpoint after 400.

Checkpoint selection used the same fixed 200 problems. Async PPO scored 2/200 then 0/200; PPO + DIS scored 4/200 then 5/200. Final evaluation covers all 645 problems, including the 200 selection problems, as requested.

PPO + DIS scored four more proofs than the base model; plain PPO scored four fewer. These are single stochastic evaluations with small success counts, not evidence of a statistically established improvement or an isolated causal effect of DIS.

Reported elapsed times include training and evaluations (base: evaluation only), and exclude some provisioning. Both trained runs retained two H200s, including during single-GPU final evaluation.

Downloaded raw evaluation reports, completion reports, training metrics/events, GPU statistics, command/config metadata, selection prompts, and tracking snapshots. Model weights and optimizer checkpoints remain on Modal.

Training metrics:
- Async PPO: 63/400 training proofs verified; mean response length 8207 tokens; 62.4 minutes inside learner steps.
- Async PPO + DIS: 69/400 training proofs verified; mean response length 4271 tokens; 48.3 minutes inside learner steps.
