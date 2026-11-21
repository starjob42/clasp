# Canonical-set CLASP training branch

This branch is opt-in. Existing primal-dual, BoN-aligned, and Frontier-CDF
runs are unchanged unless canonical mode is explicitly enabled.

## B=9, K=8 activation

Add the following arguments to the normal `clasp-grpo` launch:

```text
--use_clasp
--clasp_canonical_set
--num_generations 9
--clasp_deployment_set_size 8
--grpo_loss_type grpo
--clasp_policy_length_alpha 0.0
--clasp_canonical_recovery_scale 1.0
--clasp_canonical_dual_recovery_coupling 0.0
--clasp_canonical_recovery_temperature 0.15
```

`--num_generations` is the rollout count `B`; it is no longer interpreted as
the deployment set size in this branch. The branch requires `1 <= K < B` and
uses exact subset enumeration. `B=K+1` is recommended because it has exactly
`B` size-`K` subsets. `--clasp_canonical_max_subsets` (default `1024`) rejects
an accidentally combinatorial configuration.

Equivalent experiment environment variables are:

```text
CLASP_CANONICAL_SET=1
CLASP_DEPLOYMENT_SET_SIZE=8
CLASP_CANONICAL_RECOVERY_SCALE=1.0
CLASP_CANONICAL_RECOVERY_TEMPERATURE=0.15
CLASP_CANONICAL_MAX_SUBSETS=1024
CLASP_POLICY_LENGTH_ALPHA=0.0
```

The signed task-level controller retains the existing explicit variables:

```text
CLASP_BON_TARGET_FAILURE_RATE=0.01
CLASP_BON_TARGET_FAILURE_RATES=tsp:0.01
CLASP_BON_DUAL_LR=0.05
CLASP_BON_DUAL_EMA_BETA=0.9
CLASP_BON_LAMBDA_INIT=0.0
CLASP_BON_LAMBDA_MAX=100.0
```

CLI/environment conflicts and malformed values fail closed.

## Objective implemented

For every exact size-`K` subset, the trainer computes:

```text
U = best strictly feasible high-is-good evaluator quality
    + rho * failed_subset * bounded_softmin_recovery
    - lambda * failed_subset
```

The declared Lagrangian constant `+lambda*target_failure_rate` is omitted from
floating-point subset evaluation because it cancels from every canonical
difference; the target still drives the signed controller update.

The default coupling of zero above is the exact binary-failure control. The
named experimental recovery-pressure arm uses
`rho_effective=rho+gamma*lambda`, with the task lambda frozen for the entire
rollout group and updated only at the optimizer boundary. Positive gamma gives
all-infeasible groups stronger violation-recovery credit as failure pressure
accumulates. It does not change the measured binary failure@K controller and
must be reported as shaped auxiliary credit, not silently described as the
original binary-risk Lagrangian.

Infeasible candidates receive no objective-quality credit. Raw non-negative
violation is mapped to `d=v/(1+v)` before the log-mean-exp recovery, so failed
subset recovery lies in `[-1,0]`. The generic canonical transform produces
leave-out advantages `A`; the current outer sequence-mean GRPO loss receives
the exactly scaled coefficient `K*A`.

The controller queues
`choose(number_infeasible,K)/choose(B,K)`, not the all-`B` failure indicator,
and commits one signed projected update at an optimizer synchronization
boundary. It first initializes the task EMA to the first queued batch rate, or
thereafter applies `ema <- beta*ema + (1-beta)*batch_rate`, then projects
`lambda <- lambda + lr*(ema-target)`. Setting `beta=0` recovers the raw batch
update after initialization. The multiplier used for current credit is
detached and frozen until that boundary; multiplier, EMA, and observed-group
count are checkpointed, and the in-step queue is cleared after a successful
boundary commit.

## Core-arm isolation

Canonical configuration disables adaptive length, self replay, and legacy
component-dual updates. The reward manager remains active only as the strict
evaluator. Auxiliary preference, imitation, distillation, warmup replacement,
and hard-group rescaling weights are rejected if nonzero.

Sequence-level importance sampling remains the default importance-ratio mode.
Exact set credit now exposes the response-length choice explicitly:

```text
sequence_loss_i = token_sum_i /
                  (length_i ** (1-alpha) * max_completion_length ** alpha)
```

`--clasp_policy_length_alpha=0` exactly preserves the historical per-response
token mean. `alpha=1` is the fixed-horizon sequence-score endpoint, and `0.5`
is the preregistered midpoint pilot. Values above zero require sequence-level
importance sampling. KL, if enabled, remains a separately reduced per-response
token mean, so changing alpha does not silently change the KL contract.

This parameterization is deliberate: with outcome rewards there is a
fundamental trade-off between length-invariant trajectory weighting and the
literal sequence score-function gradient. It is not scientifically valid to
label either endpoint universally unbiased and length-neutral. The pilot must
choose alpha from held-out feasibility/quality and stability, not rhetoric.

A matched token-importance alpha-zero ablation is supported only through the
explicit acknowledgement:

```text
PD_IMPORTANCE_SAMPLING_LEVEL=token
PD_ALLOW_TOKEN_IMPORTANCE_ABLATION=1
```

Token-level ratios, geometric-mean sequence ratios, PPO clipping, and rollout
reuse are practical departures and are outside the unclipped canonical
policy-gradient theorem. The emitted `K*A` coefficient and its centering remain
auditable identities; they do not make every downstream surrogate unbiased.

## Required audit metrics

Canonical runs log the rollout/deployment sizes, subset count, failure@`K`,
full-`B` failure diagnostic, lambda, base/effective recovery settings and
dual-recovery contribution, component coefficient
magnitudes, centering residual, component-decomposition residual, and whether
sequence or token importance sampling was used. They also log alpha, realized
length weights, completion/horizon fraction, feasible candidates per group,
nonzero coefficient rate, and the fraction of groups with quality, recovery,
failure, or no policy signal. Per-task failure@`K` is also logged and drives
validation pass@`K = 1 - failure@K`.

## A100 development estimator arms

`scripts/train_clasp_canonical_checkpoint_dev_a100.slurm` accepts seven
named arms through `DEV_ESTIMATOR_ARM`:

- `token_a0_control`: historical token ratio and alpha zero;
- `sequence_a0`: isolates sequence-ratio clipping at the historical reduction;
- `sequence_a05`: midpoint length trade-off;
- `sequence_a1`: fixed-horizon sequence-score endpoint.
- `sequence_a0_dualrec1`: alpha zero plus gamma-one recovery pressure;
- `sequence_a05_dualrec1`: alpha 0.5 plus gamma-one recovery pressure;
- `sequence_a1_dualrec1`: alpha one plus gamma-one recovery pressure.

The default remains `token_a0_control`, so old invocations do not change.
`scripts/submit_clasp_vnext_length_pilot_a100.sh` submits the three sequence
arms for CVRP and MIS from their saved Jiang-F638 checkpoints, uses 40 steps in
the two-hour four-A100 development queue, and chains identical external BoN=8
evaluation jobs. This is development evidence only.
