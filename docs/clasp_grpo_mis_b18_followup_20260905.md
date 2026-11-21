The completed MIS B18 extension shows a best-of-eight tie between matched
GRPO and CLASP v3. Both solve the same 95/100 instances with the same selected
cardinalities, candidate feasibility 78.750%, and mean certified-optimum gap
2.405385%. Original B9 GRPO solves the same instances with gap 2.300122%; one
selected outcome regresses in each B18 arm. The broader set-credit support
does not produce an observed performance benefit in this pilot.

Both training jobs and both evaluations completed successfully within their
two-hour A100 development limits. Training jobs 1796118 and 1796119 took
42m20s and 42m52s. Evaluation jobs 1797974 and 1798349 took 12m35s and 12m51s.
All terminal checkpoint, tokenizer, full trainer-configuration and raw-output
checks passed. The matched candidate-rate difference is zero with paired
interval [-0.375,+0.375] percentage points; one instance improves and one
regresses. Identical selected outcomes are not proof of equivalent policies.

The [complete report](clasp_grpo_complete_development_results_20260905.md),
[MIS comparison](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_mis_comparison_archive_v1/campaign/comparison_b18_v1/comparison.json),
[raw audit](../analysis_outputs/clasp_grpo_fair_dev_20260905/audit_b18_mis_complete_three_arm_v1/README.md),
and [exact cross-check](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_mis_complete_crosscheck_v1/crosscheck.json)
retain all outcomes and uncertainty. Across 100 v3 updates, quality credit is
active in 46.5% of B18 groups versus 18.25% at B9; binary failure credit is
active in 13% versus 2.5%. Mean absolute failure coefficients are 0.004463
and 0.006361, respectively. These on-policy diagnostics have different prompt
coverage and do not establish causal performance effects.

The original pre-evaluation rationale and declaration are retained below.

The MIS B18 extension applies the same larger-group correction already declared
for CVRP. It is being declared while both CVRP B18 policies are still training,
before either CVRP B18 evaluation has produced outputs. Its rationale comes
from the completed original MIS study, not from a B18 performance result.

In the original 100-update MIS v3 run, GRPO anchor credit is nonzero in 49.75%
of rollout groups and canonical set-quality credit in 18.25%. Their difference
gives a lower bound of 31.5 percentage points on groups with GRPO reward
variation but no set-quality credit. This does not identify suppressed
optimality comparisons: mixed-feasibility groups can have equal feasible
cardinalities. Uniformly optimal groups appropriately need no quality credit.
The concern is loss of useful candidate distinctions when a tied best result
leaves every eight-subset's maximum unchanged. The
[terminal telemetry](../analysis_outputs/clasp_grpo_fair_dev_20260905/current/telemetry_audit_all_terminal100_v1/README.md)
and [original results](clasp_grpo_fresh_development_results_20260905.md) retain
the measurements and the observed best-of-eight tie.

The proposed remedy is the already implemented B=18, K=8 complete subset
system. At B=9, two tied best candidates suffice to make every subset retain
a best candidate. At B=18, that requires at least eleven tied best candidates.
Mixed-group binary failure credit can be nonzero for one through ten feasible
candidates, instead of exactly one. All-failing groups still have zero
centered binary failure credit. These support properties do not guarantee
better gradients or final performance.

Both MIS arms start fresh from the same certified MIS SFT checkpoint-31250,
with seed 43, 100 uninterrupted updates, global generation batch 36, B=18,
sample reward standard deviation, token-mean loss, learning rate 1e-6, the
same linear schedule and warmup, four Jean Zay A100s and two-hour development
QoS. CLASP uses the existing v3 anchor at scale 1, quality ratio 0.5 with the
same cap and recovery/controller settings, and dense-feasibility scale zero.
Classic GRPO uses the same scalar reward and shared optimization settings.
There are two prompt groups per update and 3,600 training completions per arm.
The original B9 arms see four groups per update at that same completion budget.

The frozen v31 source is shared with the CVRP follow-up, manifest SHA256
`c6898baabb4c6448af835a268c08749535b8b654c3b4b66b8263e5ac0d841b99`.
Its existing launcher and validator already support MIS at B18; no training
algorithm or evaluator source change is required for this extension. The new
MIS campaign and management declarations are separate from the frozen CVRP
ones. All original and CVRP results remain required context.

Evaluation uses the same previously inspected 100 MIS development instances,
their fixed certified optima, K=8 raw candidates in one call, seed 271828,
FP16, temperature 0.7, top-p 0.9, top-k 50, prompt limit 29,768 and completion
limit 3,000. Each evaluation uses one A100 with a two-hour dev limit. Outputs
remain unrepaired. This is development tuning, not untouched confirmation.

Report both matched B18 GRPO and original B9 GRPO, feasibility per candidate,
feasible@8, own-feasible and jointly feasible gaps, and unconditional solved
quality. The original MIS GRPO result index has self-hash
`7fd75da111a3866536ad53a60fed91774e76b5478100032ffadb57954f6e8106`.
Strict inference and per-instance pairing checks must pass before computing
cross-B paired statistics. One training seed and 100 instances cannot
establish a robust general advantage, even if point estimates improve.
