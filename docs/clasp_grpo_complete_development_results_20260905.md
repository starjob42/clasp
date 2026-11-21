The completed development study does **not** establish that CLASP beats classic
GRPO on feasibility and gap. The fixes broaden nonzero set-credit support.
The observed CVRP point estimates improve against matched B18 GRPO, but original B9
GRPO still solves more CVRP instances. On MIS, the B18 methods tie on every
instance's selected feasibility and objective, and both have a worse mean
gap than B9 GRPO. No tested CLASP variant achieves the requested overall win.

All 10 RL training runs and 12 evaluations completed successfully on Jean Zay
A100 development QoS with two-hour limits. Training used four GPUs per job;
evaluation used one. Every RL run started fresh from the same verified,
task-specific SFT-LoRA checkpoint-31250 and completed 100 optimizer updates.
The existing SFT checkpoints were reused; SFT training is outside this study's
allocation accounting.

| Task | Method | B | Feasible@8 | Candidate feasibility | Own-feasible gap |
| --- | --- | ---: | ---: | ---: | ---: |
| CVRP | SFT (descriptive) | — | 95/100 | 51.625% | 3.363% |
| CVRP | Classic GRPO | 9 | 98/100 | 64.625% | 3.115% |
| CVRP | CLASP v3 | 9 | 96/100 | 65.875% | 2.992% |
| CVRP | CLASP v4 | 9 | 96/100 | 64.875% | 2.969% |
| CVRP | Classic GRPO | 18 | 96/100 | 65.625% | 3.234% |
| CVRP | CLASP v3 | 18 | 97/100 | 66.125% | 3.105% |
| MIS | SFT (descriptive) | — | 94/100 | 78.250% | 2.466% |
| MIS | Classic GRPO | 9 | 95/100 | 78.875% | 2.300% |
| MIS | CLASP v3 | 9 | 95/100 | 79.375% | 2.300% |
| MIS | CLASP v4 | 9 | 95/100 | 79.250% | 2.300% |
| MIS | Classic GRPO | 18 | 95/100 | 78.750% | 2.405% |
| MIS | CLASP v3 | 18 | 95/100 | 78.750% | 2.405% |

The [complete CSV](../analysis_outputs/clasp_grpo_fair_dev_20260905/combined_complete_v1/table.csv)
and [combined statistics](../analysis_outputs/clasp_grpo_fair_dev_20260905/combined_complete_v1/comparison.json)
retain full precision, all twelve rows and all twelve strict paired RL
comparisons. B is the number of training rollouts per prompt; evaluation
always uses K=8. Feasible@8 is the fraction of instances with at least one
strictly feasible raw candidate. Candidate feasibility counts feasible
candidates out of 800 per arm. These measures are not interchangeable.

The headline gap averages each method's own successful instances. It is
`100 × max(0, (distance-reference)/reference)` for CVRP and
`100 × (optimum-cardinality)/optimum` for MIS. Failures have no conditional
gap. CVRP references are verified heuristics, not certified optima; all 100
MIS reference optima have matching CP-SAT OPTIMAL objective/bound receipts.
Signed CVRP gaps and below-reference counts remain in the CSV. SFT is
descriptive because its stage/tokenizer serialization produces a different
strict evaluation fingerprint; the SFT pairing checks were not relaxed.

The main confirmed weakness is the geometry of best-set credit. With B=9
and K=8, every subset omits exactly one candidate. A unique best candidate
with best/runner-up quality margin Δ receives credit proportional to 8Δ;
all eight others receive −Δ. Two tied best candidates make every subset
maximum identical and erase set-quality credit. Non-best candidates can
therefore lose distinctions that ordinary GRPO reward comparisons retain.
Separately, binary failure credit can be nonzero only when the feasible
count F satisfies `1 <= F <= B-K`: at B9/K8 this means exactly one feasible
candidate. All-failing groups have constant failure utility, so increasing
the dual multiplier cannot create relative binary-failure credit. Violation
recovery can still provide a different signal. These are mathematical facts
about the active estimator, not proofs of the cause of each evaluation loss.
The relevant implementation is [canonical.py](../src/clasp/set_grpo/canonical.py).

The implemented and tested remedies are:

- **Anchored CLASP v3:** retain the same group-standardized feasible-objective
  reward as classic GRPO and add the set residual. The quality multiplier is
  `0.5 × min(4, anchor_RMS/raw_quality_RMS)` when normalization is active;
  failure and recovery retain their native scales. The anchor scale is 1.
  This preserves ordinary candidate comparisons while adding set credit.
  It is an intentional hybrid surrogate and does not inherit an unbiased
  gradient claim from the raw canonical estimator. See
  [primal_dual_grpo.py](../src/clasp/trainers/primal_dual_grpo.py).
- **Dense-feasibility v4:** add `0.5(1+lambda)(f_i-mean(f))` in mixed groups.
  This explicitly rewards candidate feasibility. It disappears in
  all-feasible/all-failing groups and introduces a different tradeoff from
  feasible@8. Its negative CVRP and tied MIS results are retained above.
- **Exact B18/K8:** broaden failure-credit support to F=1…10. At least eleven
  tied best candidates are required to force constant subset maxima. Replace
  `torch.combinations` with cached immutable `itertools.combinations` tuples
  and a fresh tensor on the requested device. Only C(18,8)=43,758 actual
  subsets are constructed, avoiding the prior B^K Cartesian intermediate.
  The original B9 behavior remains available. B18 requires explicit launcher
  and validator settings; it is tested against its own matched GRPO control.

The complete 100-update logs confirm broader component support:

| Training-group diagnostic | CVRP v3 B9 | CVRP v3 B18 | MIS v3 B9 | MIS v3 B18 |
| --- | ---: | ---: | ---: | ---: |
| Nonzero set-quality credit | 92.50% | 97.00% | 18.25% | 46.50% |
| Nonzero binary-failure credit | 5.50% | 45.50% | 2.50% | 13.00% |
| Mean absolute failure coefficient | 0.006970 | 0.004935 | 0.006361 | 0.004463 |

Broader support did not imply larger average failure credit. These are
on-policy diagnostics with different prompt coverage, not causal performance
estimates. At the shared budget of 36 completions/update, B9 sees 400 prompt
groups over 100 updates and B18 sees 200; both generate 3,600 completions.
The reduction in prompt coverage is a real tradeoff. The
[CVRP](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_telemetry_all_terminal100_v1/B9_B18_full100_diagnostics.json)
and [MIS](../analysis_outputs/clasp_grpo_fair_dev_20260905/mis_b18_telemetry_all_terminal100_v1/B9_B18_full100_diagnostics.json)
receipts retain full coefficients and their source snapshots.

The fair comparisons preserve seed/data seed 43, SFT/base weights, data,
LoRA rank/alpha 64/64, dropout zero, bias=all, BF16 training, AdamW with 8-bit
state, learning rate 1e-6, linear decay, 3% warmup, token-mean loss, length
alpha zero, sample reward standard deviation (correction=1), one update per
generated batch, and no KL penalty. Classic GRPO means the group-z GRPO
objective under the shared strict feasible-objective scalar reward; it does
not mean a different native entrypoint's default multi-head reward recipe.
All 180 recorded trainer-configuration fields match within each B18 pair
apart from run/output/log names. Dataset, context and distributed settings
also match. All ten terminal adapters passed native fidelity checks for
476 persisted trainable tensors. All six tokenizer files match within each
RL task, including across B9/B18.

Historical checkpoint-bias serialization and inactive tied-rank/Dr.GRPO
options were checked or corrected separately; they are not established as
active causes in this pilot. Likewise, smaller raw gradients under historical
length weighting do not imply proportionally smaller Adam updates. The
current active reward also allows better-than-reference solutions, so an
obsolete hard reference-reward clipping concern is not the explanation here.

Evaluation uses the same 100 instances per task, raw outputs with no repair,
K8 in one call, seed 271828, FP16, temperature 0.7, top-p 0.9, top-k 50,
repetition penalty 1, prompt limit 29,768 and completion limit 3,000. The
[benchmark](../data/development_fair_cvrp_mis_20260905_v3/README.md) was generated
before the original policy outputs, follows the repository distributions,
and has zero normalized-instance overlap in a scan of 1,005,800 SFT/RL/public
source rows. The effective RL pools are 2,000 CVRP and 2,900 MIS instances;
the unchanged MIS loader removes 300 public-evaluation duplicates from its
raw 3,200 rows. Both B18 follow-ups reuse the inspected development benchmark.
The MIS extension was declared before either CVRP B18 evaluation produced
outputs, but it is still development evidence, not untouched confirmation.

The paired results are the reason the headline gaps must be read carefully:

| Comparison, CLASP minus GRPO | Feasible@8 change | Joint-success gap change | Interpretation |
| --- | ---: | ---: | --- |
| CVRP v3 B9 vs GRPO B9 | −2 pp | −0.034369 pp; 95% CI [−0.252682, +0.169272] | Fewer successes; uncertain gap benefit |
| CVRP v4 B9 vs GRPO B9 | −2 pp | −0.057009 pp; 95% CI [−0.226668, +0.100914] | Fewer successes; uncertain gap benefit |
| CVRP v3 B18 vs GRPO B18 | +1 pp | −0.131605 pp; 95% CI [−0.348535, +0.079131] | Improved point estimates; uncertain effect |
| CVRP v3 B18 vs GRPO B9 | −1 pp | +0.015848 pp on 97 joint successes | Does not beat retained B9 baseline |
| MIS v3/v4 B9 vs GRPO B9 | 0 pp | 0 pp | Same selected outcomes on all instances |
| MIS v3 B18 vs GRPO B18 | 0 pp | 0 pp | Same selected outcomes on all instances |
| MIS v3 B18 vs GRPO B9 | 0 pp | +0.105263 pp | One selected outcome regresses |

For matched CVRP B18, the feasibility interval is [-2,+5] pp, with two wins
and one loss and exact two-sided McNemar p=1. Candidate feasibility improves
0.5 pp, interval [-1.5,+2.5]. On MIS B18, aggregate candidate feasibility is
also tied, with one instance improving and one regressing; its paired interval
is [-0.375,+0.375] pp. Identical observed selected outcomes do not establish
equivalent policies or population performance. The
[success-within-gap figure](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_success_gap_plot_v1/success_within_gap.pdf)
keeps failures in the denominator and illustrates the tradeoffs without
adding an outcome-selected threshold or a superiority test.

The [original targeted validation](../analysis_outputs/clasp_grpo_fair_dev_20260905/current/targeted_test_results.json)
passed 239 distinct tests, including a separately resolved optional solver
check. Additional [B18 canonical checks](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_compute_assessment_v1/artifact_manifest.json)
and [controller/management checks](../analysis_outputs/clasp_grpo_fair_dev_20260905/mis_b18_followup_v1/README.md)
cover the follow-up changes; overlapping suite counts are not added together.
Validation also includes native checkpoint audits and
independent re-parsing of all 1,200 evaluation records and 9,600 candidates.
The [original](../analysis_outputs/clasp_grpo_fair_dev_20260905/original_complete_crosscheck_v1/crosscheck.json),
[CVRP B18](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_cvrp_complete_crosscheck_v1/crosscheck.json)
and [MIS B18](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_mis_complete_crosscheck_v1/crosscheck.json)
cross-checks match every summary field and full paired statistic set to the
independent raw audits. Frozen source manifests, exact submission commands,
checkpoint identities, signed data references, raw solutions and bootstrap
contracts are retained with those artifacts. The combined table preserves
the original baselines byte-for-value rather than silently replacing them.

The [complete accounting](../analysis_outputs/clasp_grpo_fair_dev_20260905/complete_accounting_v1/accounting.json)
counts 22 unique successful jobs: 40.447778 allocated GPU-hours of RL training
and 4.309722 of evaluation, total 44.757500. It excludes prior SFT training,
historical runs, benchmark/reference work, overlap audits, unrelated jobs and
queue waiting. These are allocated GPU-hours, not measured GPU utilization.
Both B18 follow-up evaluation schedulers exited normally after all declared jobs.

This is one training seed, 100 updates and 100 development instances per task.
The 20,000 paired bootstrap draws estimate instance sampling uncertainty,
not training-seed uncertainty; candidates are clustered by instance and no
multiplicity correction is applied. The experiments verify the implementation
changes and their signal effects, but do not prove that those effects caused
the evaluation differences or that CLASP is superior. B9 GRPO remains the
better-supported reference under this pilot. Broader independent training
seeds and untouched confirmation data are needed before making a general
advantage claim.
