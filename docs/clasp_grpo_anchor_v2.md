# GRPO-anchored canonical CLASP v2

## Decision

The next development arm preserves the complete classic GRPO policy signal and
adds a small exact deployment-set residual:

```text
A_v2 = z_group(feasible_only_objective) + 0.25 * A_exact_K8
```

It uses rollout B=9, deployment K=8, token-level importance sampling,
`CLASP_POLICY_LENGTH_ALPHA=0`, and the final certified SFT checkpoint at step
31250.  The canonical failure-at-K controller and all-infeasible recovery remain
active.  No dense rank heuristic or component-dual recovery term is enabled.

## Evidence behind the change

On the matched 100-instance CVRP validation, classic GRPO reached candidate
feasibility 0.60125 and pass@8 0.98.  Canonical sequence-alpha-0.5 reached
0.52625 and 0.93; strong-v1 reached 0.4775 and 0.82.  The corresponding jointly
feasible objective gaps were 0.03934 for classic versus 0.04228 for alpha-0.5,
and 0.03724 for classic versus 0.04775 for strong-v1 in its comparison.

The alpha-0.5 arm removed classic's dense candidate-level objective credit in
favor of sparse best-of-8 leave-out credit. Its reduction also weights each
sequence's mean-token gradient by `sqrt(length / 3000)`. Historical mean raw
gradient norms were 0.06559 versus classic GRPO's 0.23811. Both runs used
`adamw_8bit`, whose adaptive normalization approximately cancels a common
positive scaling of gradient history. No parameter-update norms were logged,
so these raw norms do not establish fourfold smaller updates or slower
learning. Length-dependent weighting can still change gradient direction, and
the simultaneous estimator change prevents isolating alpha's effect. The
[optimizer audit](../analysis_outputs/clasp_grpo_fair_dev_20260905/current/alpha05_optimizer_audit_v1/audit.json)
retains the configurations and terminal diagnostics. Strong-v1 restored raw
gradient magnitude but replaced the dense GRPO signal with several hand-scaled
terms, used fewer prompt groups per generation, and was initialized from
checkpoint 20000 rather than checkpoint 31250. It reduced parse failures while
increasing the dominant capacity and coverage failures.

The v2 arm is therefore conservative: it cannot lose classic's dense feasible
objective ranking signal at the coefficient-construction layer, while the exact
residual adds deployment-aware best-of-K quality, failure boundary, and
all-infeasible recovery information that classic lacks.

## Development launch

`scripts/submit_clasp_grpo_anchor_v2_a100.sh` submits classic and v2 for both
CVRP and MIS.  All four jobs use four A100 GPUs, `qos_gpu_a100-dev`, a two-hour
wall time, seed 42, 100 optimizer steps, and task-matched checkpoint-31250
initialization.  These runs are development evidence and are not
publication-eligible.
