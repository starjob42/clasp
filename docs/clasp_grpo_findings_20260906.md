# CLASP/GRPO investigation and corrected development experiment

The completed CLASP study does not establish a win on both feasibility and gap.
The investigation found exact weaknesses in its candidate credit, a separately
reproduced distributed-training defect in the later teacher-augmented GRPO path,
and inferior auxiliary targets. The latter two issues are corrected and tested.
The corrected paired experiment completed as well: it slightly lowers measured
gap but leaves feasible@8 unchanged on both tasks. It does not establish the
requested win on both feasibility and gap, or superiority over FOARL.

## Completed CLASP comparison

All rows below use the same task SFT-LoRA checkpoint, training seed 43, 100
updates, B9, 36 completions/update, and 100 evaluation instances/task with eight
raw samples, identical decoder settings, and no inference repair.

| Task | Method | Feasible@8 | Candidate feasibility | Gap on own successful instances |
| --- | --- | ---: | ---: | ---: |
| CVRP | Classic GRPO | 98/100 | 64.625% | 3.115% |
| CVRP | CLASP v3 | 96/100 | 65.875% | 2.992% |
| CVRP | CLASP v4 | 96/100 | 64.875% | 2.969% |
| MIS | Classic GRPO | 95/100 | 78.875% | 2.300% |
| MIS | CLASP v3 | 95/100 | 79.375% | 2.300% |
| MIS | CLASP v4 | 95/100 | 79.250% | 2.300% |

Lower conditional gap alone does not establish improvement when the successful
sets differ. On jointly feasible CVRP instances, v3 minus GRPO is −0.034
percentage points (95% paired interval −0.253 to +0.169); v4 is −0.057 points
(−0.227 to +0.101). MIS selected feasibility/objectives tie. The B18 follow-up
broadens credit support but halves prompt coverage at equal samples and also
fails to establish a win. CVRP uses verified heuristic references, so its gap is
not a certified optimality gap; MIS has certified optima.

[Full precision, B18 rows and paired statistics](../analysis_outputs/clasp_grpo_fair_dev_20260905/combined_complete_v1/table.md)
and [independent evidence audit](../analysis_outputs/clasp_grpo_diagnosis_20260906/evidence_audit.md).

## Reasons supported by the evidence

1. **Sparse set credit.** At B9/K8, binary set-failure credit differs between
   candidates only when exactly one sample is feasible. All-invalid groups
   receive no relative credit from that term, regardless of the dual multiplier.
   With a unique best solution, quality coefficients are `8Δ` for the best and
   `−Δ` for all others; tied best solutions erase the quality component. In MIS
   v3, quality credit is active in only 18.25% of training groups. The ordinary
   GRPO anchor and violation-recovery terms remain active where applicable.
   These mathematical facts identify a weakness, but do not by themselves prove
   its causal contribution to the measured performance difference.
2. **Candidate averages miss rare whole-instance successes.** V4 loses two CVRP
   instances for which GRPO had just one feasible sample. Its candidate gains
   occur on already-solved instances. Neither general diversity collapse nor a
   broad feasibility/quality tradeoff is supported by the retained diagnostics.
3. **A reproduced teacher-path training bug.** TRL 0.24 disables checkpointing
   during generation, then restores it without the configured
   `use_reentrant=False`. With policy and teacher forwards before backward,
   this leaves different gradients on different GPUs. The real unfixed MIS
   diagnostic failed before update one; a causal CPU reproduction also diverges
   before clipping. Reapplying the declared arguments fixes that reproduction.
   The corrected real four-A100 MIS run completed five updates with exact
   gradient and policy agreement at every update. This explains invalid teacher
   runs; it is not the established cause of earlier plain CLASP results.
4. **Dominated supervision.** In an offline audit of 20 already-inspected CVRP
   instances, 103/134 eligible teachers were worse than the best raw solution
   for their instance. The new selector retains one teacher only when it
   improves that incumbent, or the best repair when all raw candidates fail.
   It keeps 14 CVRP improvements and six MIS improvement/recovery opportunities
   in the inspected slice. These operator outcomes are not policy results.

[Exact formulas and source lines](../analysis_outputs/clasp_grpo_diagnosis_20260906/clasp_credit_audit_final.md),
[instance-level CVRP witnesses](../analysis_outputs/clasp_grpo_fair_dev_20260905/cvrp_post_result_mechanistic_audit_v1/POST_RESULT_NOTE.md),
[causal checkpointing receipt](../analysis_outputs/grpo_verified_ddp_diagnosis_20260906_v1/checkpointing_causal_receipt.json),
[native A100 verification](../analysis_outputs/grpo_verified_ddpfix_20260906_v2/full_model_diagnostic_v3/native_checkpoint_fix_independent_review.json),
and [teacher audit](../analysis_outputs/clasp_grpo_diagnosis_20260906/teacher_audit/report.md).

The completed current arms also show real parameter updates: their 13 checked
policy inventories are all distinct and synchronized across ranks. A near-zero
classic GRPO scalar loss is expected with centered advantages and an on-policy
ratio of one; it does not imply a zero derivative. Fixed-prompt policy drift was
not recorded, so these checks prove optimization activity without quantifying
its effect on behavior. [Optimization activity audit](../analysis_outputs/clasp_grpo_diagnosis_20260906/current_pilot_optimization_activity.md).

## Implemented correction and new experiment

The shared trainer restores its declared checkpointing mode after generation,
including exceptional exits. Opt-in exact policy checks reject rank divergence
early and at the saved endpoint. Both new arms receive this same correction.
The new modified arm uses ordinary GRPO plus graded, verifier-derived violation
rewards and `0.05 ×` selected verified-target NLL. Raw on-policy tokens and
probabilities are preserved; teacher solutions enter only auxiliary supervision.
Both changes remain opt-in, preserving other recipes.

The relevant regression suite passed 471 tests and 32 subtests. The actual
four-A100 corrected MIS diagnostic (job 1825156) completed successfully in
544 seconds, with exact gradient and policy checks at all five updates.

[Trainer](../src/clasp/trainers/primal_dual_grpo.py),
[teacher selection](../src/clasp/rewards/verified_teacher_selection.py),
[exact training checks](../src/clasp/models/training_fidelity.py),
[algorithm and fixed comparison contract](grpo_verified_incumbent_filtered_development_20260906.md).

The four full 100-update jobs are 1825454 (CVRP GRPO), 1825455 (CVRP modified),
1825456 (MIS GRPO), and 1825457 (MIS modified), each requesting four Jean Zay
A100s and a two-hour dev limit. Training and raw K8 evaluation protocols were
frozen before submission. They share initialization, data/order, seed, optimizer,
schedule, sampling settings, 3,600 raw completions, and endpoint selection.
Verified supervision adds compute, which is reported separately. This is a
sample-matched screening experiment on already-inspected development data.

All four training and all four evaluation jobs completed successfully. The new
comparison has its own corrected classic-GRPO controls; the earlier CLASP table
and this table are separate matched experiments.

| Task | Method | Feasible@8 | Candidate feasibility | Gap on own successful instances |
| --- | --- | ---: | ---: | ---: |
| CVRP | Classic GRPO, corrected shared trainer | 96/100 | 65.500% | 3.0689% |
| CVRP | GRPO + graded rewards + selected verified teachers | 96/100 | 66.000% | 3.0479% |
| MIS | Classic GRPO, corrected shared trainer | 95/100 | 79.375% | 2.3678% |
| MIS | GRPO + graded rewards + selected verified teachers | 95/100 | 78.875% | 2.2147% |

On the 94 jointly feasible CVRP instances, modified minus classic gap is
−0.0158 percentage points, with a 95% paired bootstrap interval of −0.2217 to
+0.1780 points. The modified method gains two successes and loses two others;
it improves 11 jointly feasible objectives and worsens 14. Its small mean gap
change does not establish improvement. On MIS, all 95 successful instances are
shared; two selected objectives improve and 93 tie. The gap change is −0.1531
points, with interval −0.4115 to 0.0000; the exact sign-test p-value is 0.5.
Candidate feasibility changes by +0.5 points on CVRP and −0.5 on MIS. Neither
task supports a claim of better feasibility and gap together.

The new training allocations total 17.7322 A100-hours, and evaluations 1.6333,
for 19.3656 A100-hours. Modified training costs 10.40% more than its CVRP
control and 5.55% more than its MIS control. Prior studies, failed runs, SFT,
and the 1.6467 A100-hours of checkpointing diagnostics are separate costs.
Training selected 347 CVRP teachers and 89 MIS teachers across 400 prompt groups
per task. Aggregate rank-local preparation timers sum to 34.4076 and 5.0837
seconds; these are wall timers, not isolated CPU or auxiliary GPU cost. The
two algorithm changes were combined, so this pilot does not attribute effects
between graded rewards and teacher supervision.

[Full-precision results and paired statistics](../analysis_outputs/grpo_verified_ddpfix_20260906_v2/comparison_ddpfix_v3/comparison.json),
[CSV table](../analysis_outputs/grpo_verified_ddpfix_20260906_v2/comparison_ddpfix_v3/table.csv),
[independent four-run recomputation and provenance audit](../analysis_outputs/grpo_verified_ddpfix_20260906_v2/final_four_arm_independent_review.json),
[compute audit](../analysis_outputs/clasp_grpo_diagnosis_20260906/compute_accounting/report.md),
and [campaign records and protocol identities](../analysis_outputs/grpo_verified_ddpfix_20260906_v2/STATUS.md).

A final provenance check found differing cached padding/truncation state in the
two CVRP tokenizer JSON files. All other tokenizer content matched. Native
reloading with the frozen evaluator produced identical prompt strings, token
IDs, attention masks, and decoded text for all 100 prompts. A separate reviewed
postprocessing amendment preserves both original fingerprints and admits only
this proven state difference; no checkpoint, raw output, decoder, or statistical
calculation was changed. The accounting amendment passed 11 focused tests and
the final reader 27; both passed independent review and native validation.
[Native equivalence proof](../analysis_outputs/clasp_grpo_diagnosis_20260906/native_tokenizer_equivalence_receipt.json)
and [declared reporting amendment](../analysis_outputs/grpo_verified_ddpfix_20260906_v2/final_summary_v3/declaration.json).

The feasibility ceiling limits this screening experiment's statistical power.
If a control solves 98/100 and a new method solves 100/100 with exactly two
gains and no losses, the two-sided exact paired binomial test still gives
`p = 2 × (1/2)^2 = 0.5`. More instances and independent training seeds are
needed to establish a small improvement near this ceiling. The pilot remains
useful for rejecting correctness failures and identifying promising directions.

A separate [teacher-weight audit](../analysis_outputs/clasp_grpo_diagnosis_20260906/teacher_microbatch_weight_audit.md)
shows that averaging eligible targets within each accumulation microbatch
reduces the effective auxiliary weight when teachers are sparse. This is the
documented objective, not a newly discovered training bug. A mean over all
selected targets in the optimizer update is a future ablation; it was not
introduced into the frozen running comparison.

## Direction beyond CLASP

The stronger research direction is GRPO for verified solution improvements:
condition on a feasible incumbent, propose bounded edits, verify objective
changes, and retain the best feasible solution. Use legal-action masks for
constructive decoding and returned-output validity, with exactly the same
masked distribution in rollout and policy log probabilities. Compare controlled
infeasible intermediate search states separately; restricting all search states
to feasibility can hurt exploration. Improvement policies are established
research, so the contribution must be demonstrated rather than renamed.

Jiang et al.'s FOARL itself uses GRPO with feasibility and objective rewards.
A beta-zero classic control is not an exact FOARL baseline. Beating it requires
an explicit matched FOARL arm, an untouched test set, multiple training seeds,
a longer declared horizon, matched tuning and inference budgets, and costs
including teacher search. Any decoder or search aid must also be given to its
control. The present development experiments cannot establish a SOTA result.
[Primary Jiang paper](https://papers.neurips.cc/paper_files/paper/2025/file/f0c6c93e1dbff84daf6082fef7e1a094-Paper-Conference.pdf)
and [concrete follow-up research contract](grpo_verified_research_outlook_20260906.md).
