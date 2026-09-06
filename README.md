# PPO_ASYNC

## PPO Background

Proximal Policy Optimization (PPO) is a policy gradient method that uses a learned critic to estimate advantages and a clipped objective to improve learning stability.

For a generated token $`a_t`$, let $`s_t`$ be the theorem context and preceding tokens. PPO updates the current policy $`\pi_\theta`$ using the rollout generated with the older policy $`\pi_{\mathrm{old}}`$ using the following objective:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\mathrm{old}}(a_t \mid s_t)}$$

$$L_t^{\mathrm{clip}}(\theta)
= \min\left[
    r_t(\theta)\hat{A}_t,\;
    \text{clip}\!\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_t
  \right]$$

## Async PPO Background

Asynchronous PPO overlaps proof generation with learning. A set of async workers generate rollouts that enter a bounded queue, and the learner updates on the next eight completions while generation continues. After each update, the new policy is sent to the async workers. This reduces training time as proofs can be generated while the critic and actor are being trained, but introduces **policy lag**, which can greatly reduce the effectiveness of PPO.

## DIS

Direct Double-Sided Importance Sampling (DIS), introduced in [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning (SAO), Hou et al., 2026](https://arxiv.org/html/2607.07508v1#S3.SS1), addresses policy lag by comparing token probabilities under the learner and rollout policies. Tokens outside a double-sided trust region are excluded from actor gradients, regardless of their advantage sign.

Let $`\pi_{\mathrm{rollout}}`$ be the policy generating the rollout, and $`\pi_{\mathrm{b}}`$ be the policy we are currently trying to optimize. The mask (defined by parameters $`\epsilon_\ell, \epsilon_h`$) is:

$$\rho_t
= \frac{\pi_{\mathrm{b}}(a_t \mid s_t)}{\pi_{\mathrm{rollout}}(a_t \mid s_t)}
= \exp\!\left(
    \log\pi_{\mathrm{b}}(a_t \mid s_t)
    - \log\pi_{\mathrm{rollout}}(a_t \mid s_t)
  \right)$$

$$f_t(x; \epsilon_\ell, \epsilon_h)
= \begin{cases}
    1, & \text{if } 1-\epsilon_{\ell} < x < 1+\epsilon_h, \\
    0, & \text{otherwise}.
  \end{cases}$$

The objective is:

$$J_{\mathrm{PPO+DIS}}(\theta)
= \frac{1}{T}\sum_{t=1}^{T}
    f_t(L_t^{\mathrm{clip}}(\theta); \epsilon_\ell, \epsilon_h)$$


## Goal

The goal is to investigate whether asynchronous reinforcement learning with DIS can improve **Qwen/Qwen3.5-4B** at Lean 4 theorem proving within a small compute budget, using a simple pipeline built on SLIME and SGLang.

The RL environment supplies a formal theorem and its Lean context. The model generates a candidate proof of up to **16,384 tokens**. A Lean verifier checks the proof and audits its axiom dependencies: valid proofs receive reward **1**, and rejected proofs receive **0**. Reference proofs are excluded from prompts, and infrastructure errors are treated as failures rather than negative rewards.

We compare the base model, Async PPO, and Async PPO + DIS. Each training run processes **400 problems in one pass**, with batch size eight and **50 actor and 50 critic updates**.

## Data

| Dataset | Role | Description | Problems used |
| --- | --- | --- | ---: |
| [Lean-Workbook](https://huggingface.co/datasets/internlm/Lean-Workbook) | Training | Mathematical problems formalized in Lean 4; we retain rows marked `proved`. | 200 |
| [ProofNet-Verified](https://github.com/marcusm117/ProofNet-Verified) | Training | An audited and corrected version of ProofNet, with verified Lean 4 formalizations of textbook mathematics. | 200 |
| [Gaokao-Formal](https://github.com/Huawei-AI4Math/Mathesis) | Evaluation | Proof problems derived from China's college entrance examination; we provide the existing formal statements to the prover. | 495 |
| [FATE-M](https://github.com/frenzymath/FATE-M) | Evaluation | Abstract algebra textbook exercises formalized in Lean. | 150 |

Training takes the first 200 retained problems from each source after filtering. Source and model revisions are pinned. Preparation removes exact duplicates and exact evaluation-statement matches after text normalization, and excludes reference proof fields from prompts. This does not guarantee removal of semantically equivalent problems.

Checkpoints at 200 and 400 training problems are evaluated on the same fixed 200-problem selection set: 100 problems from each evaluation dataset. The best checkpoint is then evaluated on **all 645 problems**, including the selection set. Evaluation uses one response per problem, a 16,384-token cap, and concurrency 16. Training concurrency remains eight.

## Final results

Both training runs and all final evaluations completed successfully. Final **pass@1** is the fraction of problems solved by one generated, Lean-verified proof:

| Model | Gaokao-Formal | FATE-M | Overall pass@1 |
| --- | ---: | ---: | ---: |
| Base | 6/495 | 6/150 | **12/645 — 1.86%** |
| Async PPO | 5/495 | 3/150 | **8/645 — 1.24%** |
| Async PPO + DIS | 10/495 | 6/150 | **16/645 — 2.48%** |

**Async PPO + DIS achieved the highest observed score**, verifying four more proofs than the base model and twice as many as plain Async PPO. Final evaluation used plain PPO's checkpoint after 200 training problems and PPO + DIS's checkpoint after 400.

| Training-run measurement | Async PPO | Async PPO + DIS |
| --- | ---: | ---: |
| Training problems / actor updates | 400 / 50 | 400 / 50 |
| Verified training proofs | 63/400 (15.75%) | 69/400 (17.25%) |
| Mean training response length | 8,207 tokens | 4,271 tokens |
| Time inside learner updates | 62.4 min | 48.3 min |
| Total reported runtime, including evaluations | 3h 52m | 2h 23m |
| H200s allocated per run | 2 | 2 |

The PPO + DIS run finished soon as it had substantially shorter training responses. 