# Canonical CLASP screen from current SFT checkpoints (development only)

Date: 2026-08-31. This protocol is a pre-RL diagnostic. It is not a final CLASP result and is
not publication-eligible.

## Scientific question

Before spending the full GRPO budget, establish:

1. the current SFT policy's exact best-of-8 feasibility and conditional optimality gap;
2. whether canonical CLASP produces finite, centered, non-degenerate credit from that policy;
3. whether two optimizer steps preserve or improve held-out feasibility/gap under identical
   decoding; and
4. which failure mode should be addressed before the full run (no feasibility signal, no quality
   signal, unstable update, excessive duplication, or evaluation regression).

The short horizon can validate implementation and immediate direction. It cannot estimate the
converged RL gain.

## Algorithm under test

For one prompt, sample `B=9` conditionally IID candidates. Deployment draws `K=8`. All nine
size-eight leave-one-out subsets are enumerated exactly. For subset `S`,

```text
success(S)  = any strictly verifier-feasible candidate in S
quality(S)  = max objective_quality(i), over feasible i in S; 0 if S fails
d_i         = violation_i / (1 + violation_i)
recovery(S) = tau * (logsumexp(-d_i / tau, i in S) - log K), only if S fails
failure(S)  = 1 - success(S)
U(S)        = quality(S) + rho * failure(S) * recovery(S) - lambda * failure(S)
```

The screen uses `tau=0.15`, `rho=1`, and defines the reporting-scale quality as
`q=exp(-3 * signed_gap)` for nonnegative gaps. The repository's intentional reward convention
passes `objective_score=2*q` into canonical CLASP (with a bounded continuation for a negative
gap), so successful subset quality is on a nominal `[0,2]` rather than `[0,1]` scale. Reporting
below keeps `q` on `[0,1]`; this factor-of-two distinction matters when interpreting the relative
recovery and dual scales. Infeasible candidates always receive zero objective score.

For candidate `i`, canonical credit is

```text
A_i = mean[U(S) | i in S] - mean[U(S) | i not in S].
```

Because the trainer mean-reduces across the nine candidates, its policy coefficient is `K*A_i`.
For every realized group, `sum_i A_i = 0` up to floating-point error. Under the on-policy IID and
fixed symmetric-utility assumptions, `(K/B) * sum_i A_i grad log pi(y_i|x)` is an unbiased
score-function estimator for expected deployed size-eight utility. PPO clipping, token-level
importance sampling, and token-length normalization make the actual training loss a practical
surrogate, not the literal unbiased estimator; their effect must therefore be monitored.

The exact deployment-failure observation is the U-statistic

```text
failure@8 = choose(number_infeasible, 8) / choose(9, 8).
```

At each optimizer boundary, per task,

```text
failure_ema <- observed failure@8 on the first update,
               otherwise 0.9 * old + 0.1 * observed
lambda <- clip(lambda + 0.05 * (failure_ema - 0.005), 0, 100).
```

The component-level dual controller is frozen at zero. The only active constraint controller is
the set-level failure-at-eight controller.

### Important estimator behavior

`B=K+1` is the cheapest exact leave-out construction, but it is also coarse:

- There are only nine subsets. Per group, `failure@8` can only be `0`, `1/9`, or `1`, depending on
  whether at most seven, exactly eight, or all nine candidates are infeasible.
- With global generation batch 36, only four prompt groups inform each optimizer step. The
  controller observation is therefore noisy; a two-step lambda trajectory is a sign/wiring check,
  not a calibrated constraint estimate.
- If all nine candidates fail, `-lambda * failure` is constant over all nine subsets and has zero
  canonical advantage. Learning then depends entirely on the bounded recovery ranking. If all
  violation scores tie (for example identical parser failures), the complete group has zero policy
  signal even if lambda is large.
- If all nine candidates are feasible, quality credit is nonzero only where leaving a candidate out
  changes the subset maximum. Exact objective ties can therefore make quality credit sparse.

For `B=9,K=8`, every subset excludes one candidate. If `U_-j` is the utility after excluding
candidate `j`, then `K*A_i = 9 * (mean_j(U_-j) - U_-i)`. This makes the credit especially easy to
audit: a candidate is positive exactly when removing it hurts the deployed-set utility relative to
the average removal.

These are properties of the exact set objective, not implementation bugs. The smoke must inspect
recovery/quality coefficient mass and tie/zero-signal rates. If informative mass is systematically
near zero, the next development comparison should increase `B` above `K+1` (with compute held
constant as far as possible) or improve strictly verifier-derived recovery resolution before the
full campaign.

The canonical arm also uses `beta=0`. This avoids an expensive second policy and matches the current
arm, but offers no KL drift protection. The two-step screen can detect immediate instability only.
If a longer pilot shows policy/format drift, compare a small fixed-KL arm using the exact SFT adapter
reference; do not use an inferred foundation-model reference.

The current matched arm explicitly permits TRL token-level importance sampling and uses GRPO's
per-sequence mean over completion tokens. With `num_iterations=1`, token- and sequence-level TRL
importance ratios are both one at the update and have the same first-order, length-normalized
gradient. That remains a practical surrogate: the literal score-function theorem uses the sum of
token log-probability gradients. A theorem-faithful sequence-sum loss should be a separate tested
arm before making an unbiased-estimator claim; the present screen does not make that claim.

## Preserved production settings

- Four A100 GPUs; no H100/V100 fallback.
- Exact current-checkpoint representation: `jiang_f638`, no-reason inference prefix, and legacy
  Jiang MIS response rendering (`CLASP_MIS_JIANG_RESPONSE_OUTPUT=0`).
- `B=9`, `K=8`, global generation batch 36.
- Per-device train batch 3, gradient accumulation 3, effective batch 36.
- Temperature 0.7, top-p 0.9, top-k 50, repetition penalty 1.0.
- Maximum prompt/completion lengths 29,768/3,000.
- LoRA rank/alpha 64/64, seven Jiang target modules, bias `all`.
- AdamW 8-bit, learning rate `1e-6`, weight decay 0.01, linear schedule, warmup 0.03.
- GRPO clipping epsilon 0.2, beta 0, reward scaling disabled.
- No pairwise, elite, replay, local-search, feasibility-warmup, or adaptive-length auxiliary loss.
- Exact saved-policy fidelity is required for both input and emitted checkpoints.

This representation binding is essential. The generic older CLASP production launcher is hardcoded
to `alpaca --no_reason` and native MIS response rendering; that is not the prompt/output contract
used to train or evaluate these current Jiang-F638 SFT checkpoints. A full RL launch from this SFT
family must retain the explicit Jiang settings used by this screen.

Development-only changes are a fixed two-step horizon, no internal evaluation, offline/no
reporting, and one checkpoint per step. Independent held-out evaluation is used instead.

## Checkpoint inventory used

| Task | Current checkpoint | Decision |
| --- | ---: | --- |
| CVRP | 20,000 | Baseline plus canonical two-step screen |
| JSSP | 25,000 | Baseline plus canonical two-step screen |
| MIS | 20,000 | Baseline plus canonical two-step screen |
| OP | 25,000 | Baseline plus canonical two-step screen |
| TSP | old 20,000 | Rejected: the original trajectory collapsed |
| TSP repaired | no checkpoint yet | Wait for the stable recovery run's first certified save |

CVRP and MIS ran first because they are the feasibility bottlenecks and exercise different
verifier/objective structures. JSSP and OP use the identical protocol to exercise the quality-only
regime after the wiring gate passed.

## Evaluation contract

Both pre- and post-update evaluations use the same external `eval[200:300]` rows, seed 271828,
Jiang-F638 SFT representation profile, FP16, and one-call best-of-8 sampling with the same decoding
parameters. Report at minimum:

- best-of-8 feasible-instance rate (primary feasibility metric);
- candidate-level feasibility rate and all-infeasible group rate;
- conditional gap among feasible best-of-8 selections, with task direction handled correctly;
- unconditional solved-quality summary where supported;
- result count, parse/error count, instance-size distribution, and exact checkpoint identity.

The before/after comparison is paired by semantic instance. Report the 2x2 feasibility transition
counts (fail/fail, fail/pass, pass/fail, pass/pass) and a McNemar or exact-binomial interval/test on
the discordant pairs. Compare gaps on the jointly feasible subset, but do not use that conditional
comparison alone: feasibility changes alter which instances enter the mean. Also report the
deployment-aligned unconditional quality

```text
mean(1[best-of-8 feasible] * exp(-3 * best_feasible_gap)).
```

This jointly penalizes failure and poor feasible quality without assigning objective credit to an
infeasible answer.

Training logs must additionally show finite loss/grad norm, nonzero total coefficient mass when
the group is informative, quality/recovery/failure coefficient mass, `failure@8`, lambda before and
after the optimizer boundary, centering residual, and component-decomposition residual.

## Go/no-go interpretation

- **Go:** no runtime/fidelity/verifier failures; centering and decomposition residuals remain near
  numerical zero; at least one intended credit component is nonzero; post-update feasibility does
  not show an obvious paired-slice regression; gap remains finite and interpretable.
- **Investigate before full RL:** all coefficients are zero despite heterogeneous candidates;
  recovery dominates on otherwise successful groups; lambda moves with the wrong sign; infeasible
  candidates receive objective credit; parse/length failures dominate; or paired held-out results
  regress materially.
- **Do not overclaim:** two optimizer steps and 100 held-out instances are a direction/wiring screen,
  not evidence of converged superiority. Any apparent small percentage change needs a longer,
  preregistered multi-seed run.

## Submitted Jean Zay jobs

- Baseline CVRP: 1610042
- Baseline JSSP: 1610043
- Baseline MIS: 1610044
- Baseline OP: 1610045
- Canonical CVRP two-step training: 1611573 (complete)
- Canonical MIS two-step training: 1611575 (complete)
- Corrected paired CVRP evaluation: 1611954 (complete)
- Corrected paired MIS evaluation: 1611955 (complete)
- Canonical JSSP two-step training: 1612022 (complete)
- Paired JSSP evaluation: 1612027 (complete; `afterok:1612022`)
- Canonical OP two-step training: 1612023 (complete)
- Paired OP evaluation: 1612102 (complete; `afterok:1612023`)

Jobs 1610123 and 1610304 failed before optimization because the SFT-only environment could not
honor `scale_rewards='none'`. Jobs 1610443/1610444 were superseded before optimization after Slurm
had snapshotted a pre-correction launcher. The first paired evaluations 1611574/1611576 correctly
failed on a representation-contract mismatch before generation. No failed attempt produced a
policy result. The experiment-scoped evaluation profile was then made to reproduce the normalized
representation recorded by the v3 GRPO checkpoints, and the v4 evaluations passed the guard.
