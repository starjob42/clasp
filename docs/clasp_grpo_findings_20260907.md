# CLASP, modified GRPO and FOARL: measured results

The completed experiments do **not** establish that CLASP or the modified GRPO
beats classic GRPO on feasibility and gap together. The fresh 300-instance/task
evaluation also does not establish that joint improvement.
Several concrete defects and weaknesses were corrected, but correctness fixes
and plausible objectives have not translated into the requested joint win.

All six fresh evaluations completed successfully within their two-hour Jean
Zay A100 dev allocations and passed strict validation. Every predeclared
checkpoint was evaluated, including the weaker development arms. The separate
common repair/search comparison was declared before any fresh model outputs;
its first execution failed before any search because the runtime provenance
writer rejected the environment's Python symbolic link. A separate management
successor passed the actual Jean Zay interpreter/synthetic-operator preflight
and completed as job 1859098, with the original scientific settings preserved.
Its validated results give every assisted method 100% feasibility on both
tasks. They still do not establish a modified-GRPO advantage over the matched
controls, and remain separate from raw policy performance.

## Fresh fixed heldout evaluation: 300 instances per task

The same final step-100 adapters from the matched development experiment below
were evaluated on a separately generated, fixed draw with eight raw candidates
per instance. All 300 instances and all 2,400 candidates per arm are retained;
there is no repair in this table. The new draw passed the predeclared native
overlap audit against eight corpora totaling 1,006,000 rows. This is one training
seed on two task families, not a SOTA test or a literal Jiang reproduction.
Classic GRPO uses the strict task reward and beta=0; the FOARL port uses its
released reward heads and initial-SFT KL reference. Training samples and update
counts are matched, while their additional computation is measured separately.

| Task | Method | Feasible@8 | Candidate feasibility | Signed reference gap on own successes |
|---|---|---:|---:|---:|
| CVRP | Classic GRPO | 286/300 | 61.2917% | 3.4787% |
| CVRP | Modified GRPO | 286/300 | 62.7083% | 3.6078% |
| CVRP | Matched released-objective FOARL | 284/300 | 60.1250% | 3.3648% |
| MIS | Classic GRPO | 281/300 | 80.2917% | 2.2402% |
| MIS | Modified GRPO | 276/300 | 77.7500% | 2.2595% |
| MIS | Matched released-objective FOARL | 272/300 | 78.6250% | 1.8979% |

CVRP references are verified heuristic solutions. Negative instance gaps are
retained; the three arms beat their references on 7, 6 and 7 instances,
respectively. The corresponding legacy clipped mean gaps are 3.4956%, 3.6216%
and 3.3846%. MIS references are certified optima. Own-success gaps use the
success count in the same row, which can differ between methods.

The primary paired contrasts are modified minus each baseline. Gap differences
and intervals below are in percentage points; lower gap favors the modified arm.

| Task | Baseline | Feasibility change (pp) | Gained / lost | Shared successes | Shared-success gap change | 95% paired interval |
|---|---|---:|---:|---:|---:|---:|
| CVRP | Classic GRPO | 0.00000 | 2 / 2 | 284 | −0.11446 | [−0.27570, +0.03148] |
| CVRP | Matched FOARL | +0.66667 | 5 / 3 | 281 | −0.02163 | [−0.19003, +0.13683] |
| MIS | Classic GRPO | −1.66667 | 0 / 5 | 276 | +0.08302 | [−0.00983, +0.20175] |
| MIS | Matched FOARL | +1.33333 | 8 / 4 | 268 | +0.00013 | [−0.14471, +0.13806] |

CVRP's equal feasibility counts hide two gained and two lost instances. Its
own-success gap worsens, while its gap on shared successes improves slightly;
neither quantity should substitute for the other. The paired interval includes
zero. MIS loses five classic-GRPO successes and has a slightly worse shared
gap. FOARL's lower own-success MIS gap also does not imply better quality on
the same instances: the modified-versus-FOARL shared-success gap is nearly tied.

All eight predeclared 99.375% simultaneous intervals for feasibility and
shared-success gap include zero.
The MIS modified-versus-classic exact feasibility test gives p=0.0625 and
Holm-adjusted p=0.25 across the four feasibility tests; every other adjusted
p-value is 1.0. The 95% intervals above are descriptive paired bootstraps,
not replacements for that fixed multiplicity treatment. None establishes the
requested joint superiority.

All six Slurm endpoints are `COMPLETED 0:0`; their total allocation is
6.841389 A100-hours. The longest job took 5,997 seconds. Combined with the
matched training/development campaign below, this is 37.811111 allocated
A100-hours, excluding SFT, earlier experiments, correctness diagnostics,
fresh-data audit allocations and the separate secondary analysis.

[Exact heldout table](../analysis_outputs/grpo_incumbent_20260907_v1/heldout_summary_execution_v1/table.csv),
[full paired statistics, simultaneous intervals and accounting](../analysis_outputs/grpo_incumbent_20260907_v1/heldout_summary_execution_v1/comparison.json),
and [strict completion receipt](../analysis_outputs/grpo_incumbent_20260907_v1/heldout_v10_metadata_completion_v1.json).

## Fixed common assistance: the same 300 instances per task

This secondary comparison applies the same bounded, reference-free repair and
local search to all three policies' already retained eight candidates. CVRP
uses at most two passes of 2-opt/relocate/exchange; MIS uses at most two exchange
passes and 4,096 pair checks. Unparseable or unrecoverable invalid sources use
the same cached instance-only engine fallback. Each instance also has a
separate engine-only K=1 control. The worker receives geometry/graph and raw
completions; reference objectives are used only for final scoring.

All 600 instances, 14,400 model candidate slots and 600 engine controls were
retained. Every assisted candidate passed the strict verifier. Consequently,
all assisted method-to-method gap comparisons below use the same full cohort.
The science was fixed in v2 before primary outputs were inspected; v3 only
amends interpreter provenance after the diagnosed pre-search failure.

| Task | Method with common assistance | Feasible@8 | Signed reference gap |
|---|---|---:|---:|
| CVRP | Classic GRPO | 300/300 | 2.012202% |
| CVRP | Modified GRPO | 300/300 | 2.015297% |
| CVRP | Matched FOARL | 300/300 | 1.991352% |
| MIS | Classic GRPO | 300/300 | 0.942771% |
| MIS | Modified GRPO | 300/300 | 0.890523% |
| MIS | Matched FOARL | 300/300 | 0.890242% |

The engine-only K=1 control also solves 300/300 on each task, with much larger
gaps: CVRP 37.010501%, MIS 3.506485%. Model candidates therefore contribute
useful quality to this particular hybrid. This weak engine control is not a
comparison with strong specialized solvers.

The predeclared descriptive paired contrasts are modified minus baseline;
all differences and intervals are percentage points. Feasibility is tied in
all four comparisons.

| Task | Assisted baseline | Paired gap change | Descriptive 95% interval |
|---|---|---:|---:|
| CVRP | Classic GRPO | +0.003095 | [−0.086594, +0.094586] |
| CVRP | Matched FOARL | +0.023945 | [−0.069650, +0.120476] |
| MIS | Classic GRPO | −0.052249 | [−0.167379, +0.055152] |
| MIS | Matched FOARL | +0.000281 | [−0.088159, +0.086420] |

Each of these four intervals includes zero, as do the additional fixed original
raw joint-success cohort contrasts. These are exploratory intervals outside the
primary significance family. The common assistance improves feasibility and
quality for all policies; the modified policy has no established advantage.
Although some assisted gaps are numerically below Jiang's published values,
the differing cohorts, training, references and added search prevent a fair
cross-study superiority claim.

Job 1859098 completed `0:0` in 187 seconds, reserving one hidden A100 and eight
logical CPUs for four workers. Its cost is 0.051944 allocated A100-hours and
0.415556 allocated logical CPU-hours. Summed worker CPU time is 62.706935
seconds; parent CPU time is 78.517476 seconds and processing plus validation
wall time is 180.072100 seconds. These quantities are distinct from allocation
products. The exact report separately records candidate-operator CPU time and
the cached-fallback charge for independently deploying each arm.

[Exact secondary CSV](../analysis_outputs/grpo_incumbent_20260907_v1/secondary_aggregate_export_execution_v2/table.csv),
[aggregate statistics and measured timing](../analysis_outputs/grpo_incumbent_20260907_v1/secondary_aggregate_export_execution_v2/secondary_aggregate.json),
and [full allocation accounting including the failed predecessor](../analysis_outputs/grpo_incumbent_20260907_v1/secondary_aggregate_export_execution_v2/resource_accounting.json).
The [independent local audit](../analysis_outputs/grpo_incumbent_20260907_v1/secondary_common_assistance_local_audit_v3.json)
verified all five transported files, the 14 rows, 16 contrasts and exact
agreement of all six raw rows with the primary comparison.

The original secondary allocation, job 1857566, ended `FAILED 1:0` after
87 seconds: 0.024167 allocated A100-hours and 0.193333 allocated logical
CPU-hours. It produced no admission receipt, initialized no operators and
dispatched no candidate assistance. A bounded source-stack classifier and
filesystem-only probe established that `identity(sys.executable)` rejected
the valid `python -> python3.10` alias. The resolved binary matches the running
process. The separate correction resolves only executable provenance and
preserves strict nonsymlink requirements for scientific inputs. This amendment
is dated after inspection of primary results and binds the original prospective
secondary plan; it is not a new prospective scientific declaration.
Including this failure, the secondary allocations total 0.076111 A100-hours
and 0.608889 logical CPU-hours. The matched training/development, fresh raw
evaluation and secondary allocations together total 37.887222 A100-hours;
SFT, earlier campaigns, correctness diagnostics and fresh-data audit
allocations remain separate.

[Failure diagnosis and no-dispatch evidence](../analysis_outputs/grpo_incumbent_20260907_v1/secondary_v2_interpreter_identity_diagnosis_review_v1.md).

## Latest matched development experiment

All six arms start from the same task-specific SFT LoRA checkpoint and use
Qwen2.5-7B-base, LoRA rank 64, seed 43, 100 updates, 400 prompt groups and 3,600
raw training completions. Data/order, optimizer schedule, generation settings
and endpoint selection are matched. Each training job uses four Jean Zay A100s
with the requested two-hour dev limit. Evaluation uses 100 instances per task,
eight raw candidates per instance and the same strict verifier and decoder.
Training samples are matched; additional teacher and KL-reference compute is
included in the reported allocation costs.

| Task | Method | Feasible@8 | Candidate feasibility | Gap on own feasible instances |
|---|---|---:|---:|---:|
| CVRP | Classic GRPO | 96/100 | 65.000% | 3.1538% |
| CVRP | Modified GRPO | 94/100 | 65.875% | 3.0155% |
| CVRP | Matched released-objective FOARL | 95/100 | 63.625% | 3.0530% |
| MIS | Classic GRPO | 95/100 | 79.250% | 2.2625% |
| MIS | Modified GRPO | 94/100 | 77.125% | 2.2762% |
| MIS | Matched released-objective FOARL | 93/100 | 77.375% | 2.3304% |

CVRP references are verified heuristic solutions, not certified optima. The
table retains the evaluator's nonnegative reference gap. The signed CVRP
reference gaps are respectively 3.1538%, 3.0155% and 3.0517%; a negative signed
instance gap means the candidate beats its heuristic reference. MIS references
are certified optima. All feasibility denominators include every instance.

Comparing gap only over each method's own successful set can hide failures.
The paired modified-minus-classic results are:

| Task | Feasibility change | Gained/lost instances | Shared successes | Gap change on shared successes | 95% paired interval |
|---|---:|---:|---:|---:|---:|
| CVRP | −2 percentage points | 1 / 3 | 93 | +0.00992 percentage points | [−0.10618, +0.13011] |
| MIS | −1 percentage point | 0 / 1 | 94 | −0.01036 percentage points | [−0.14507, +0.11398] |

Lower gap is better. CVRP's apparent improvement in the first table does not
survive restriction to shared successes. Neither task establishes a joint win.
These intervals describe development-instance variation for one training seed;
they do not estimate variation between training seeds.

All six trainings completed 100 updates and passed all 13 policy consistency
inventories. Training allocations total 28.708889 A100-hours. Evaluation
allocations total 2.260833 A100-hours including preserved failures, giving
30.969722 A100-hours for this training/evaluation campaign. SFT, earlier
experiments, four correctness diagnostics and fresh-data audit allocations are
separate costs. Allocated GPU-hours are distinct from Slurm billing units.

Three original evaluation attempts failed before loading the model because the
read-only source lacked an empty logs directory; their complete costs and
one-attempt replacements are retained. Both FOARL evaluations produced all
100 raw instances, then failed an overly narrow checkpoint-file inventory
validator. A separately declared, strict postprocessing admission proved the
three extra files were the original bound KL-reference proofs. It admitted the
complete outputs without repeating inference or changing raw records. Their
actual Slurm `FAILED 1:0` statuses and full allocation costs remain in the
comparison. The heldout manager includes the prospective inventory correction.

[Exact CSV](../analysis_outputs/grpo_incumbent_20260907_v1/development_summary_execution_v1/table.csv),
[complete paired statistics and accounting](../analysis_outputs/grpo_incumbent_20260907_v1/development_summary_execution_v1/comparison.json),
[development report](../analysis_outputs/grpo_incumbent_20260907_v1/current_development_report.md),
and [algorithm interpretation](../analysis_outputs/grpo_jiang_benchmark_20260907/frozen_algorithm_interpretation_20260907.md).

## Earlier CLASP experiment

This is a separate completed experiment with its own matched controls, using
the same SFT, seed 43, B9, 100 updates and raw K=8 on 100 instances/task. Do not
combine its classic-GRPO row with the new modified or FOARL rows above.

| Task | Method | Feasible@8 | Candidate feasibility | Gap on own feasible instances |
|---|---|---:|---:|---:|
| CVRP | Classic GRPO | 98/100 | 64.625% | 3.115% |
| CVRP | CLASP v3 | 96/100 | 65.875% | 2.992% |
| CVRP | CLASP v4 | 96/100 | 64.875% | 2.969% |
| MIS | Classic GRPO | 95/100 | 78.875% | 2.300% |
| MIS | CLASP v3 | 95/100 | 79.375% | 2.300% |
| MIS | CLASP v4 | 95/100 | 79.250% | 2.300% |

The paired CVRP gap intervals include zero; MIS best-feasible results tie.
A subsequent corrected teacher pilot also failed to establish improved
feasibility, despite small conditional gap reductions.
[Original full table](../analysis_outputs/clasp_grpo_fair_dev_20260905/combined_complete_v1/table.md)
and [previous corrected pilot](clasp_grpo_findings_20260906.md).

## What was found and fixed

* **Sparse CLASP set credit.** With B9/K8, its binary set-failure coefficient
  differentiates samples only when exactly one candidate is feasible; it gives
  no relative credit when every candidate fails. Tied best solutions erase the
  corresponding quality component. In the inspected MIS v3 run, quality credit
  was active in only 18.25% of groups. Other GRPO-anchor and graded terms can
  still provide signal. This is an exact structural weakness, not a complete
  causal explanation of the performance difference.
* **Distributed teacher-training defect.** Generation restored gradient
  checkpointing without the declared non-reentrant setting. Multiple forwards
  then caused different gradients across ranks. The causal reproduction and
  actual A100 diagnostics verify the correction. This defect affected the
  later teacher path, not the earlier plain-CLASP result.
* **Inferior teacher targets.** An inspected CVRP slice had 103/134 offered
  teachers worse than the best raw feasible incumbent. Selection now verifies
  feasibility and objective and admits an improvement, an equal incumbent in
  mixed-feasibility groups, or recovery when all raw candidates fail. Raw
  on-policy tokens and probabilities are preserved.
* **Sparse teacher weighting and missed MIS opportunities.** The latest arm
  normalizes auxiliary loss across the whole optimizer update and admits an
  equal-quality incumbent when some raw samples are invalid. It selected 367
  CVRP and 151 MIS targets over 400 groups; 60 MIS targets are equal-quality
  mixed-group incumbents. These are measured training opportunities, not
  evidence of improved policy performance.
* **Evaluation provenance defects.** Missing read-only logs and an incomplete
  FOARL proof-file inventory were repaired through retained, narrowly scoped
  management amendments. They explain failed job statuses, not the algorithm's
  negative metrics.

The current modified arm combines graded verifier rewards and verified-teacher
NLL with weight 0.05. It is not an isolated ablation of either ingredient.
Near-zero centered on-policy scalar loss does not imply zero gradient. All six
current arms passed the recorded policy synchronization checks. Broad diversity collapse,
completion truncation and a guaranteed optimal reward weight are unsupported
explanations in the retained evidence.

The latest MIS output audit identifies a residual format failure: all eight
modified candidates on development instance 37 fail bracket-list extraction,
while classic and FOARL each produce eight feasible candidates there. This is
exactly the sole modified-versus-classic instance-level feasibility loss. The
failed outputs are only 25–30 tokens long. There are 15 such modified parse
failures across five instances, versus none for either control. This establishes
the verifier's rejection reason, not the validity of any hypothetical alternate
parsing or which training ingredient caused the failure.
[Retained parser diagnosis](../analysis_outputs/grpo_incumbent_20260907_v1/current_mis_parse_diagnosis.md).

The direct causal follow-up is a fixed 2×2 development experiment: strict versus
graded rewards, crossed with teacher weight zero versus 0.05, using the same
SFT, prompt stream, optimizer budget and decoder. The completed three-package
comparison cannot assign the format loss to either ingredient. This ablation
has not been run and must not be inferred from the current table.

A subsequent static audit found that every active MIS teacher renderer retains
the set and objective, tokenization retains the whole target and EOS, and the
auxiliary loss covers all target positions. It found no concrete omission bug.
Existing tests do not combine the actual MIS tokenizer, rollout prompt, left
padding, row shuffling and causal labels in one test; that coverage limit remains.
[Teacher-format audit](../analysis_outputs/grpo_jiang_benchmark_20260907/mis_teacher_format_static_audit_20260907.md).

[Trainer](../src/clasp/trainers/primal_dual_grpo.py),
[teacher selector](../src/clasp/rewards/verified_teacher_selection.py),
[latest teacher telemetry](../analysis_outputs/grpo_jiang_benchmark_20260907/current_incumbent_teacher_training_mechanisms.md),
and [earlier exact diagnosis and tests](clasp_grpo_findings_20260906.md).

## Jiang et al.: literature values, separate benchmark

The following reproduces the relevant rows of Table 1 of Jiang et al.,
*Large Language Models as End-to-end Combinatorial Optimization Solvers*,
NeurIPS 2025. It is a literature table, not a matched measurement on our new
instances. Their best-of-three experiment selection, training horizon,
hardware, benchmark/reference lineage and some decoding settings differ.

| Published method | Candidates | CVRP feasibility | CVRP gap | MIS feasibility | MIS gap |
|---|---:|---:|---:|---:|---:|
| SFT | 1 | 59% | 6.02% | 80% | 1.71% |
| SFT + FOARL | 1 | 80% | 8.27% | 83% | 1.34% |
| SFT + FOARL + best-of-8 | 8 | 100% | 4.53% | 94% | 1.04% |

The published CVRP feasibility already reaches 100%; a strict increase is
impossible. A valid target is to match that feasibility and improve gap under
matched conditions. Our matched FOARL port preserves the released ordered
reward heads and exact initial-SFT KL reference, while using the common
100-update experimental budget. It is not a literal reproduction of the
published training recipe. The released one-epoch recipe covers 2,000 CVRP or
3,200 MIS training instances, whereas our bounded development run uses 400
prompt groups per task. Matching our three arms does not remove that
cross-study training-budget difference. The current measurements establish no
SOTA claim.
[Published Table 1](https://papers.neurips.cc/paper_files/paper/2025/file/f0c6c93e1dbff84daf6082fef7e1a094-Paper-Conference.pdf)
and [pinned official implementation](https://github.com/Summer142857/LLMCoSolver/tree/f6388991d243c0aa91d9ee7e2fa18ed1c9e72477).

The requested combined presentation appends our three raw rows to those
published rows with an explicit study/cohort column:
[Markdown table](../analysis_outputs/grpo_jiang_benchmark_20260907/jiang_table1_with_local_context.md)
and [CSV](../analysis_outputs/grpo_jiang_benchmark_20260907/jiang_table1_with_local_context.csv).
It is cross-study context; only the three local rows form the controlled
comparison on the same instances.

## Recommended successor to whole-solution GRPO

Use explicit constraint state and learn verified edits from feasible
incumbents. For CVRP, maintain visited customers and remaining capacity; for
MIS, maintain the selected set and conflicts. Legal construction can preserve
feasibility under valid task assumptions and adequate step budgets, but cannot
guarantee good objective values.

Then train a short-horizon edit policy: CVRP relocate/swap/2-opt and MIS
add/delete/one-for-two moves, with an exact engine verifying every transition.
Group GRPO trajectories by the same instance and incumbent. Reward improvement
in the best feasible objective reached, using no test optimum or reference
label. Short action outputs concentrate credit on consequential choices;
neutral or temporarily worse intermediate states can support escape from local
optima while the best feasible incumbent is retained.

For the learning rule, compare terminal-return GRPO with independent
leave-one-out trajectories without group standard-deviation scaling, then
with reward-to-go credit. For step t, the proposed return is final best-feasible
gain minus the gain already attained before that step; subtract the other
independent trajectories' mean return at t. Sum each action's token log
probabilities and divide by the fixed number of trajectory slots, counting
sampled STOP actions and masking later absorbing slots. Keep conventional
GRPO's different token reduction as an explicit control. The fixed 9/8 factor
between B9 leave-one-out and group-mean centering must also be controlled.

This is a testable variance/weighting intervention, not a claim that terminal
credit has a biased policy gradient. An exact synthetic enumeration verifies
the same expected gradient for terminal and reward-to-go leave-one-out credit,
and demonstrates failures from copied trajectories, immediate-only rewards and
random active-length normalization. It is a mathematical prototype, not an
integrated or trained CO policy.
[Exact credit rule and retained check](../analysis_outputs/grpo_jiang_benchmark_20260907/next_grpo_edit_credit_rule.md).

The same state features, legal masks, initial solutions, search budget and
decoder must be given to classic GRPO and FOARL controls. Masked probabilities
must agree in rollout, old/current policy ratios and KL reference. Compare
engine-only and random-edit controls, quality versus total compute, independent
training seeds and strong specialized solvers. This is a research proposal
with testable mechanisms, not a demonstrated win or a new claim of novelty.
[Detailed implementation and comparison design](../analysis_outputs/grpo_jiang_benchmark_20260907/next_grpo_state_and_edit_design.md).

The successor also needs comparison with newer work such as
[FALCON/BOPO](https://arxiv.org/abs/2602.01090).
[A bounded source review](../analysis_outputs/grpo_jiang_benchmark_20260907/falcon_2026_context.md)
records its reported metrics and unresolved reproduction issues. Beating
Jiang's older table alone would not establish the current state of the art.
