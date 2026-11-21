# Anchored CLASP v3: norm-matched tail credit

This is a development-only response to the CVRP step-200 comparison between
classic GRPO and anchored CLASP v2. It is not confirmation or SOTA evidence.

## Diagnosis

V2 combined a group-standardized feasible-objective GRPO coefficient with
`0.25` times a raw-utility canonical best-of-8 coefficient. Those terms do not
have comparable units. Across CVRP steps 101--200, the canonical quality term
had only 9.5% of the anchor's mean absolute magnitude (4.7% at the median
step). Canonical failure and all-infeasible recovery credit were present in
only 5.8% and 8.8% of prompt groups, respectively. In the paired evaluation,
500 of 800 candidate outcomes were identical between v2 and classic GRPO.

The B=9/K=8 leave-one-out quality estimator is necessarily tail-sparse: for a
successful group it primarily separates the best feasible candidate from the
runner-up. This is the desired best-of-8 direction, but its raw transformed
objective margin was too small beside a unit-scale GRPO z-score. The benefit
that did appear was concentrated on 81+ node CVRP instances, supporting a
stronger tail signal rather than a new dense feasibility heuristic.

## V3 coefficient

For one gathered generation batch, let `g` be the classic feasible-only GRPO
anchor and let `q` be the centered canonical `K*A` quality coefficient. V3
uses

```
s = min(4, RMS(g) / RMS(q))
A_v3 = g + 0.5 * s * q + raw_canonical_recovery - lambda * raw_failure_credit
```

The rescale is detached and disabled when either RMS is at most `1e-8`.
Because `q` is centered, scalar norm matching preserves its realized
within-group centering. The 4x cap prevents a nearly tied best/runner-up pair
from being arbitrarily amplified.

Only the quality component is norm matched. Recovery and binary failure terms
remain in their original Lagrangian units so controller magnitude continues to
mean what it reports. V3 retains B=9, K=8, global generation batch 36,
token-level importance sampling, alpha 0, and the full classic anchor. It adds
no pairwise, elite-imitation, dense-lexicographic, replay, or adaptive-length
loss.

The norm-matched quality coefficient is a deliberate practical surrogate, not
the raw unbiased canonical finite-set estimator. Logs expose its raw RMS,
rescale, cap activation, achieved anchor ratio, and cosine with the GRPO
anchor.

## Matched development test

The targeted test branches from the certified v2 step-100 checkpoint and
trains v3 for the same additional 100 optimizer steps used by the completed v2
continuation. A dependent one-A100 job freshly evaluates classic GRPO, v2, and
v3 step 200 on CVRP `eval[100:200]`, seed 271828, exact BoN=8. This makes the
v2-v3 comparison isolate the changed second-half policy coefficient while the
classic comparison retains the established reference.
