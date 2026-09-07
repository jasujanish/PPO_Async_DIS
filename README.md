# PPO_ASYNC
## Background
### PPO Background

Proximal Policy Optimization (PPO) is a policy gradient method that uses a learned critic to estimate advantages and a clipped objective to improve learning stability.

For a generated token $`a_t`$, let $`s_t`$ be the theorem context and preceding tokens. PPO updates the current policy $`\pi_\theta`$ using the rollout generated with the older policy $`\pi_{\mathrm{old}}`$ using the following objective:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\mathrm{old}}(a_t \mid s_t)}$$

$$L_t^{\mathrm{clip}}(\theta)
= \min\left[
    r_t(\theta)\hat{A}_t,\;
    \text{clip}\left(r_t(\theta), 1-\epsilon_{val}, 1+\epsilon_{val}\right)\hat{A}_t
  \right]$$

### Async PPO Background

Asynchronous PPO overlaps proof generation with learning. A set of async workers generate rollouts that enter a bounded queue, and the learner updates on the next eight completions while generation continues. After each batch update, the new policy is sent to the async workers. This reduces training time as proofs can be generated while the critic and actor are being trained, but introduces **policy lag**, which can greatly reduce the effectiveness of PPO.

### DIS

Direct Double-Sided Importance Sampling (DIS), introduced in [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning (SAO), Hou et al., 2026](https://arxiv.org/html/2607.07508v1#S3.SS1), addresses policy lag by comparing token probabilities under the learner and rollout policies. Tokens outside a double-sided trust region are excluded from actor gradients, regardless of their advantage sign.

Let $`\pi_{\mathrm{rollout}}`$ be the policy generating the rollout, and $`\pi_{\theta}`$ be the policy we are currently trying to optimize. The mask (defined by parameters $`\epsilon_\ell, \epsilon_h`$) is:

$$r_t(\theta)
= \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\mathrm{rollout}}(a_t \mid s_t)}
= \exp\left(
    \log\pi_{\theta}(a_t \mid s_t)
    - \log\pi_{\mathrm{rollout}}(a_t \mid s_t)
  \right)$$

$$f_t(x; \epsilon_\ell, \epsilon_h)
= \begin{cases}
    x, & \text{if } 1-\epsilon_{\ell} < x < 1+\epsilon_h, \\
    0, & \text{otherwise}.
  \end{cases}$$

The objective used in the SAO paper is as follows: 

$$L(\theta)=\mathbb{E}_t[f(r_t(\theta), \epsilon_\ell, \epsilon_h) * \hat A_t*\log\pi_\theta(a_t \mid s_t)]$$

---

## Runs
### Run 0: Baseline Async PPO

This is the **Async PPO** run in the results below. It uses the standard clipped
PPO objective with the recorded rollout probabilities as the old-policy
baseline, and **no DIS mask**:

$$
r_t^{(0)}(\theta)
= \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\mathrm{rollout}}(a_t \mid s_t)}.
$$

The PPO clipping width is $`\epsilon_{val}`$; the separate mask uses widths $`\epsilon_\ell`$ and $`\epsilon_h`$.

For a trajectory with $`T`$ valid action tokens, the actor maximizes

$$
J_0(\theta)
= \frac{1}{T}\sum_{t=1}^{T}
\min\left[
    r_t^{(0)}(\theta)\hat{A}_t,\;
    \text{clip}\left(r_t^{(0)}(\theta),1-\epsilon_{val},1+\epsilon_{val}\right)\hat{A}_t
\right].
$$

Here $`\pi_\theta`$ is the actor being optimized, and $`\pi_{\mathrm{rollout}}`$ represents the behavior probabilities recorded when each token was generated. A trajectory can span multiple published policy versions; its recorded token probabilities remain fixed during training.

### Experiment 1: Async PPO + Baseline-Policy Masking

This is the run labeled **Async PPO + Baseline-Policy Masking** in the results below. It retains
PPO clipping and distinguishes three policy roles:

| Policy | Meaning in this experiment |
| --- | --- |
| $`\pi_{\mathrm{rollout}}`$ | The behavior policy that actually generated each token. Its token log probabilities are recorded during generation. A long proof can contain tokens generated under several weight versions. |
| $`\pi_b`$ | The learner policy immediately before this batch's actor update. It evaluates the completed proofs to obtain baseline token log probabilities, which are then held fixed during the update. It can be newer than the policies that generated the proofs. |
| $`\pi_\theta`$ | The actor whose parameters are being optimized. It starts the update with the same weights as $`\pi_b`$; gradients are computed through this actor, while the baseline probabilities remain fixed. |

For example, a proof might start under V0 and finish under V1, then enter
training when the learner is at V2. Its rollout probabilities come from V0/V1;
its baseline probabilities are recomputed under V2. The actor starts at V2 and
becomes V3 after the optimizer step. The baseline is refreshed for each batch.
These are three policy roles, not three simultaneously resident model copies:
we retain the rollout and baseline log probabilities needed for the loss.

Both ratios below evaluate the **same token given the same preceding context**:

```math
r_t^{(1)}(\theta)
= \frac{\pi_\theta(a_t \mid s_t)}{\pi_b(a_t \mid s_t)},
\qquad
x_t
= \frac{\pi_b(a_t \mid s_t)}{\pi_{\mathrm{rollout}}(a_t \mid s_t)}.
```

PPO clipping uses $`r_t^{(1)}`$ to measure the actor's change from the learner
baseline. A separate binary mask uses $`x_t`$ to reject tokens with excessive
learner-to-rollout mismatch:

```math
m_t(x_t)
= \begin{cases}
    1, & \text{if } 1-\epsilon_\ell < x_t < 1+\epsilon_h, \\
    0, & \text{otherwise}.
  \end{cases}
```

The actor maximizes

```math
J_1(\theta)
= \frac{1}{T}\sum_{t=1}^{T}m_t(x_t)
\min\left[
    r_t^{(1)}(\theta)\hat{A}_t,\;
    \text{clip}\left(r_t^{(1)}(\theta),1-\epsilon_{val},1+\epsilon_{val}\right)\hat{A}_t
\right].
```

The baseline probabilities and mask are fixed during the actor update. Masked tokens contribute zero actor gradient while remaining in the denominator $`T`$; trajectory objectives are averaged across the batch in both experiments. **Retained losses are not additionally multiplied by $`x_t`$.** This is binary masking on a PPO objective, not SAO's direct importance-weighted objective.

### Experiment 2: Async PPO + DIS Masking

**Completed.** This is Run 0 plus a DIS-style binary rejection
mask. PPO clipping and masking both use the current actor-to-rollout ratio;
there is no recomputed baseline policy $`\pi_b`$ in the actor objective.

```math
r_t^{(2)}(\theta)
= \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\mathrm{rollout}}(a_t \mid s_t)},
\qquad
m_t(r_t^{(2)})
= \begin{cases}
    1, & \text{if } 1-\epsilon_\ell < r_t^{(2)}(\theta) < 1+\epsilon_h, \\
    0, & \text{otherwise}.
  \end{cases}
```

The actor maximizes

```math
J_2(\theta)
= \frac{1}{T}\sum_{t=1}^{T}m_t(r_t^{(2)})
\min\left[
    r_t^{(2)}(\theta)\hat{A}_t,\;
    \text{clip}\left(r_t^{(2)}(\theta),1-\epsilon_{val},1+\epsilon_{val}\right)\hat{A}_t
\right].
```

The mask is nondifferentiable, while gradients flow through the PPO ratio.
There is no extra importance multiplier: the PPO ratio already supplies it.
Rejected tokens remain in the original denominator $`T`$, and trajectory
objectives are averaged across the batch. The critic, GAE, batch size, and
single actor update per batch remain the same as Run 0. Unlike Experiment 1's
actor-to-baseline ratio, this ratio can differ from one before the actor step.

This hybrid retains PPO clipping in addition to DIS-style rejection; it is
not the exact DIS objective displayed in SAO. Masking may reduce extreme
updates but also discards useful signal and does not fully correct off-policy
bias. Final results for this experiment are included below.


## Compute + Scope Note
- This is a small project done to learn more about LLM post-training
- I'd be eager to perform complete ablations on this work, but I'm compute constrained (running of free credits)
- I'd be eager to do a true academic study on improving async post-training for LLMs in the future, but this repo is not a full research project, just a few small experiments

## Goal

The goal is to investigate whether asynchronous reinforcement learning with DIS can improve **Qwen/Qwen3.5-4B** at Lean 4 theorem proving within a small compute budget, using a simple pipeline built on SLIME and SGLang.

The RL environment supplies a formal theorem and its Lean context. The model generates a candidate proof of up to **16,384 tokens**. A Lean verifier checks the proof and audits its axiom dependencies: valid proofs receive reward **1**, and rejected proofs receive **0**. Reference proofs are excluded from prompts, and infrastructure errors are treated as failures rather than negative rewards.

We compare the base model, Async PPO, Async PPO + Baseline-Policy Masking, and Async PPO + DIS Masking. Each training run processes **400 problems in one pass**, with batch size eight and **50 actor and 50 critic updates**.

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

All three training runs and their final evaluations completed successfully. Final **pass@1** is the fraction of problems solved by one generated, Lean-verified proof:

| Model | Gaokao-Formal | FATE-M | Overall pass@1 |
| --- | ---: | ---: | ---: |
| Base | 6/495 | 6/150 | **12/645 — 1.86%** |
| Async PPO | 5/495 | 3/150 | **8/645 — 1.24%** |
| Async PPO + Baseline-Policy Masking | 10/495 | 6/150 | **16/645 — 2.48%** |
| Async PPO + DIS Masking | 7/495 | 4/150 | **11/645 — 1.71%** |

**Async PPO + Baseline-Policy Masking achieved the highest observed score**, verifying four more proofs than the base model and twice as many as plain Async PPO. Final evaluation used plain PPO's checkpoint after 200 training problems and both masking variants' checkpoints after 400.

| Training-run measurement | Async PPO | Async PPO + Baseline-Policy Masking | Async PPO + DIS Masking |
| --- | ---: | ---: | ---: |
| Training problems / actor updates | 400 / 50 | 400 / 50 | 400 / 50 |
| Verified training proofs | 63/400 (15.75%) | 69/400 (17.25%) | 42/400 (10.50%) |
| Mean training response length | 8,207 tokens | 4,271 tokens | 4,350 tokens |
| Time inside learner updates | 62.4 min | 48.3 min | 50.4 min |
| Total reported runtime, including evaluations | 3h 52m | 2h 23m | 2h 49m |
| GPUs allocated per run | 2 | 2 | 2 |

The baseline-policy masking run finished soon as it had substantially shorter training responses. 