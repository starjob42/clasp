# CLASP vNext: SOTA-gap analysis and development gate

Date: 2026-09-01. Status: development protocol, not a SOTA claim.

## Bottom line

CLASP's credit directly targets the deployed best-of-`K` set utility. This
alignment does not establish better learning or performance than ordinary
per-candidate GRPO; the estimator can have much weaker observed credit.
Its main limitations are not one missing loss term. In descending
order of practical importance they are:

1. **Insufficient policy support.** If the SFT checkpoint almost never samples
   a feasible large CVRP/MIS solution, on-policy CLASP has no feasible solution
   to reinforce. Recovery can improve violation geometry, but it cannot turn an
   absent mode into reliable objective learning in a two-step run.
2. **Sparse and tied exact max credit.** With `B=9,K=8`, leaving out a candidate
   changes successful-set quality only when it removes the unique best feasible
   candidate. Discrete MIS/JSSP objectives and duplicate generations therefore
   produce many exact zeros. All-infeasible groups rely on recovery; if their
   violation scores also tie, the whole group has zero policy signal.
3. **A hard-failure dual dead zone.** The `-lambda * failure` term has nonzero
   leave-out credit only at the feasibility boundary. For `B=9,K=8`, it is
   constant when all nine candidates fail and when at least two candidates are
   feasible. Thus increasing lambda cannot strengthen the gradient in an
   all-infeasible group; only the fixed recovery term learns there. This is
   mathematically correct for the hard indicator but poor continuation
   behavior far from the feasible set.
4. **An implicit response-length estimator choice.** The old trainer used a
   per-response token mean and usually token importance ratios. That is not the
   literal sequence score-function estimator associated with the canonical
   set theorem, and it can change the relative influence of short and long
   solutions.
5. **A representation-invalid production path.** The old generic production
   launcher uses Alpaca prompts and native MIS response rendering. The current
   Jiang-F638 SFT checkpoints were trained with `jiang_f638 --no_reason` and
   `CLASP_MIS_JIANG_RESPONSE_OUTPUT=0`. RL from a different representation is
   not a meaningful algorithm comparison.
6. **Too little evidence for algorithm selection.** The completed canonical
   checkpoint screen had only one effective nonzero-learning-rate update. On
   the paired 100-instance slice, CVRP pass@8 moved 0.63 to 0.64, MIS stayed
   0.86, and OP/JSSP stayed 1.00. Those results validated the path but did not
   distinguish estimators or establish improvement.
7. **No evidence-backed choice among existing auxiliaries.** Pairwise loss,
   elite imitation, replay, local-search distillation, and hard-group weighting
   already exist in the repository. Bundling them would prevent attribution and
   can optimize a proxy different from deployed raw best-of-eight performance.
8. **No independent SOTA evidence.** Current confirmation is blocked on the
   repository's reference/IID campaign requirements. “SOTA” requires external
   comparator coverage, independent seeds, frozen evaluation, and uncertainty;
   a clean loss implementation is necessary but not sufficient.

The correct near-term move is therefore a controlled estimator pilot from the
same saved SFT checkpoints, followed by a longer multi-seed run only if it
passes feasibility, signal-density, stability, and paired held-out gates.

## Relation to current GRPO research

GRPO was introduced as a critic-free PPO variant in
[DeepSeekMath](https://arxiv.org/abs/2402.03300). Subsequent work identified
several implementation choices that materially affect outcome-reward RL:

- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)
  identifies response-length and group-normalization biases and proposes the
  Dr.GRPO fixed-horizon reduction.
- [DAPO](https://arxiv.org/abs/2503.14476) combines decoupled clipping, dynamic
  sampling, token-level loss, and overlong-response shaping. These techniques
  are useful hypotheses, not a license to bundle all of them into CLASP.
- [GSPO](https://arxiv.org/abs/2507.18071) moves likelihood ratios and clipping
  to the sequence level for improved stability.
- [On the Impossibility of Unbiased and Length-Invariant Policy Optimization
  with Outcome Rewards](https://arxiv.org/abs/2607.23364) shows that the GRPO
  and Dr.GRPO endpoints make an unavoidable trade-off. Its family
  `f_alpha(L)=L^(alpha-1)` motivates exposing alpha rather than declaring one
  endpoint universally correct.

CLASP has an additional distinction: its reward unit is a symmetric set of
`K` candidates, not an individual response. The exact leave-out coefficient is
therefore worth preserving while estimator and sampling choices are ablated
around it.

## Implemented vNext changes

For exact canonical or fixed-null set credit only, the trainer now reduces the
policy term for response `i` as

```text
policy_loss_i(alpha) = token_sum_i /
                       (L_i ** (1-alpha) * C ** alpha)
```

where `L_i` is the realized completion length and `C` is the configured fixed
`max_completion_length`.

- `alpha=0`: exactly the historical per-response token mean;
- `alpha=1`: the fixed-horizon sequence-score endpoint;
- `alpha=0.5`: a preregistered midpoint, not a claimed optimum.

For sequence-level importance sampling, the importance ratio is the geometric
mean token likelihood ratio and is clipped once per sequence. At the on-policy
point, alpha one restores a token-sum score gradient up to the fixed `1/C`
scale; alpha zero retains length-invariant response weighting. Intermediate
alpha makes the trade-off measurable.

The KL term remains separately token-mean reduced. Consequently, alpha changes
only the policy estimator and cannot silently rescale KL regularization.

The implementation fails closed when:

- alpha is non-finite or outside `[0,1]`;
- nonzero alpha is requested outside exact set credit;
- nonzero alpha is combined with token-level importance sampling;
- a completion is empty or exceeds the fixed horizon; or
- the fixed horizon is absent/nonpositive.

The historical default is alpha zero. Existing production and publication
arms do not change implicitly.

The exact-set trainer also exposes one named remedy for the hard-failure dual
dead zone:

```text
rho_effective(group) = rho + gamma * frozen_lambda(group)
U(S) = quality(S) + rho_effective * failure(S) * recovery(S)
       - lambda * failure(S)
```

`--clasp_canonical_dual_recovery_coupling=0` is the unchanged canonical
control. A positive gamma makes accumulated failure pressure amplify bounded
violation recovery in all-infeasible groups, where the binary failure
leave-out term itself is constant. Lambda remains frozen across the rollout
group and is committed only at an optimizer boundary, so credit is still exact
for the resulting symmetric shaped set utility. This is an auxiliary
continuation arm—not a claim that it is the Lagrangian of the original binary
failure constraint—and is logged separately from binary failure@K pressure.

## New observability

Every exact-set update now records:

- policy alpha and realized length-weight mean/min/max;
- completion length and completion/horizon fraction;
- candidate feasibility and feasible candidates per rollout group;
- nonzero candidate-coefficient fraction;
- informative and zero-signal group fractions;
- separate quality-, recovery-, and failure-signal group fractions;
- the existing failure@`K`, lambda, coefficient decomposition, and centering
  invariants.

These metrics distinguish four very different outcomes:

1. no feasible support but useful recovery signal;
2. feasible support with exact-max ties and little quality signal;
3. informative credit whose gradient is too small/unstable under an estimator;
4. genuinely improving held-out policy behavior.

## Two-hour A100 pilot

Use only the current Jiang-F638 representation and four A100s. The development
launcher exposes the following arms:

| Arm | Importance ratio | Alpha | Purpose |
| --- | --- | ---: | --- |
| `token_a0_control` | token | 0.0 | reproduce the historical practical surrogate |
| `sequence_a0` | sequence | 0.0 | isolate sequence clipping |
| `sequence_a05` | sequence | 0.5 | midpoint length trade-off |
| `sequence_a1` | sequence | 1.0 | fixed-horizon sequence-score endpoint |
| `sequence_a0_dualrec1` | sequence | 0.0 | alpha-zero plus gamma=1 recovery pressure |
| `sequence_a05_dualrec1` | sequence | 0.5 | midpoint plus gamma=1 recovery pressure |
| `sequence_a1_dualrec1` | sequence | 1.0 | alpha-one plus gamma=1 recovery pressure |

The default submission grid uses CVRP and MIS, because these are the current
feasibility bottlenecks and have different output lengths/discrete objective
geometry. It runs the three sequence arms for 40 optimizer steps from the same
saved SFT checkpoint and seed, saving every ten steps, then chains a one-A100
external evaluation on exactly `eval[200:300]`, seed 271828, raw one-call BoN=8.
The already completed token-alpha-zero run is historical context, but a fresh
matched control can be included by setting `PILOT_ARMS`.

```bash
PILOT_SOURCE_SNAPSHOT=/path/to/new/immutable/source-snapshot \
PILOT_OUTPUT_ROOT=/lustre/.../clasp_vnext_length_pilot_20260901_v1 \
scripts/submit_clasp_vnext_length_pilot_a100.sh
```

The submitter records train/eval Slurm IDs and immutable paths in a read-only
TSV ledger. Training is refused on non-A100 devices; both training and
evaluation use the two-hour A100 development QoS.

## Decision gates

No arm advances merely because its training loss decreases.

### Correctness gate

- all checkpoints and fidelity receipts are present;
- finite loss, gradient norm, advantages, and length weights;
- coefficient centering and component decomposition residuals remain near
  floating-point zero;
- sequence arms report sequence importance, and alpha matches the receipt;
- representation is exactly Jiang-F638 for training and evaluation.

### Learning-signal gate

- informative-group fraction is reported for every step;
- zero-signal fraction must not dominate the run without a documented cause;
- CVRP must show recovery signal on failed groups;
- successful groups must show nonzero quality signal often enough to support
  objective learning;
- if support is absent, stop RL tuning and improve SFT/data/decoding first.

A provisional investigation threshold is an informative-group rate below
25% over the last 20 steps. This is a diagnostic threshold, not a publication
criterion.

### Stability gate

- no growth in clipped/overlong completions or parser failures;
- no collapse of entropy or candidate diversity;
- no extreme alpha-dependent gradient-norm divergence;
- failure@8 controller moves with the correct sign and does not saturate.

### Held-out performance gate

For each task/arm report:

- pass@8 and candidate feasibility;
- paired fail→pass and pass→fail counts against the same SFT baseline;
- conditional feasible gap and jointly-feasible paired gap;
- unconditional solved quality
  `mean(feasible * exp(-3 * max(gap,0)))`;
- exact-binomial/McNemar uncertainty for feasibility transitions.

Advance an estimator only if it has no obvious paired pass@8 regression and
improves either pass@8 or unconditional solved quality without violating the
stability/signal gates. A one-seed 100-instance pilot selects what to study; it
does not establish superiority.

## What follows if the pilot passes

1. Repeat the winning estimator and alpha-zero sequence control for at least
   three training seeds on CVRP/MIS with a 100--200 step fixed horizon.
2. Tune learning rate jointly with alpha. Fixed-horizon alpha changes absolute
   gradient scale, particularly for short MIS responses, even though Adam is
   partly scale-adaptive.
3. Add OP/JSSP, then the repaired TSP checkpoint, only after the feasibility
   gate is credible.
4. Test one sparse-credit remedy at a time. The launcher now provides the
   bounded dual-coupled recovery continuation
   `(rho + gamma*lambda) * failure * recovery`, because it gives the signed
   failure controller causal leverage in all-infeasible groups while retaining
   zero recovery on successful subsets. It remains an experimental shaped
   utility, so gamma is zero by default and gamma one must be compared as a
   named arm after selecting the length estimator. Then test rollout-budget
   allocation/dynamic resampling based on verifier outcomes. A smooth-max or
   dense preference auxiliary is a separate biased-proxy arm and must not
   replace the canonical control silently.
5. Only after development selection, freeze a new publication source snapshot,
   prespecify seeds/checkpoints, compare against GRPO/Dr.GRPO/GSPO-style and
   task baselines, and run the independent IID confirmation protocol.

## Explicit non-claims

- The code change is a principled improvement in configurability,
  correctness, and observability; it is not evidence that alpha 0.5 or 1 is
  better on these tasks.
- Exact best-of-`K` alignment does not overcome missing SFT support.
- Two-hour dev jobs are not publication evidence.
- CLASP should be called SOTA only after a frozen, independent, statistically
  supported comparison says so.
