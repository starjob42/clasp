# Current attempt to improve on matched FOARL

There is no demonstrated win yet. The completed previous modified-GRPO experiment
did not beat both classic GRPO and matched FOARL on feasibility and gap. The new
FOARL-plus-verified-teacher candidate is a different training recipe; its results
must not be filled with the previous recipe's numbers.

The [completed comparison](clasp_grpo_findings_20260907.md) retains the complete
statistics, paired cohorts, and compute accounting. Its raw best-of-eight results
on the now-consumed 300 instances per task are:

| Task | Method | Feasible@8 | Candidate feasibility | Gap on own successes |
| --- | --- | ---: | ---: | ---: |
| CVRP | Classic GRPO | 286/300 | 61.2917% | 3.4787% |
| CVRP | Previous modified GRPO | 286/300 | 62.7083% | 3.6078% |
| CVRP | Matched FOARL | 284/300 | 60.1250% | 3.3648% |
| MIS | Classic GRPO | 281/300 | 80.2917% | 2.2402% |
| MIS | Previous modified GRPO | 276/300 | 77.7500% | 2.2595% |
| MIS | Matched FOARL | 272/300 | 78.6250% | 1.8979% |

Each gap mean uses that row's successes. These differing cohorts cannot establish
paired quality superiority. CVRP gaps are signed against verified heuristic
references; MIS uses certified optima. The primary paired comparisons did not
establish the requested joint improvement. This consumed set is excluded from
tuning the new attempt.

The **current attempt** uses the newly frozen **420-file source**, manifest
`f55230e0eef410cb730192b0e54e967360c0806a603dc546b31e9640a418ccc8`.
Its [runtime declaration](../experiments/co_stronger_teacher_runtime_v3/worktree/docs/co_stronger_teacher_ranklocal_development_20260909_v3.md)
and [source/staging review](../analysis_outputs/co_stronger_teacher_ranklocal_20260909_v3/training_freezer_profile_final_review.md)
describe stronger teacher search and a new contract that saves and restores
each rank's own optimizer state. This supersedes the earlier 408-file attempt
described below; its passes and failures are not substituted into the new gate.

As observed at **2026-09-09 03:38:33 UTC**, the current initial correctness jobs
had the following statuses in
[metadata 002](../analysis_outputs/co_stronger_teacher_ranklocal_20260909_v3/diagnostic_metadata_002.json):

| Current diagnostic | Job | Status | Elapsed allocation | Synthetic tokenizer probe |
| --- | ---: | --- | ---: | --- |
| MIS classic GRPO, five updates | 1928788 | RUNNING | 324 seconds | Passed |
| MIS matched FOARL, five updates | 1928789 | RUNNING | 213 seconds | Passed |
| MIS stronger-teacher candidate, five updates | 1928790 | PENDING | — | Not yet run |
| CVRP stronger-teacher candidate, five updates | 1928791 | PENDING | — | Not yet run |

The [actual diagnostic protocol](../analysis_outputs/co_stronger_teacher_ranklocal_20260909_v3/native_diagnostic_protocol.json)
declares these four initial jobs and three MIS continuations from their own
update-5 checkpoints. All seven must pass the new save/restoration checks;
none had completed at this observation, and no continuation had been submitted.
Each job requests four A100s for at most two hours. **Current production training
jobs submitted: 0; current development evaluations submitted: 0.** The current
[analysis directory](../analysis_outputs/co_stronger_teacher_ranklocal_20260909_v3/)
retains the source, declaration, submissions and dated metadata separately from
the historical receipts.

The current candidate retains FOARL's released reward heads and fixed SFT KL
reference and adds verified teacher loss at weight 0.05. It performs one owned
search per prompt and complete rollout window: eight proposal slots for CVRP,
or at most 8,192 branch-and-bound states for MIS. It retains the whole-group raw
best as the acceptance floor, strict verification, and the existing global
optimizer-window loss normalization. Weight-zero controls perform no teacher
search. All methods retain common SFT initialization, training instances,
seed 43, 400 updates, nine rollouts per prompt, global rollout batch 36 and
50-update restart boundaries. Full gradient and policy consensus checks now
run at every update for every method; their overhead is common. Extra teacher
work is reported separately. The prospective comparison uses identical raw
best-of-eight DEV100 evaluation at updates 100, 200 and 400, with 400 primary.
There are no policy results for this recipe yet.

The **historical 408-file attempt** and its
[development protocol](../analysis_outputs/co_live_resume50_20260909_v1/native_development_protocol.json)
remain preserved with zero production training or evaluation submissions.

As observed at **2026-09-09 02:45:05 UTC**, all six submitted correctness jobs
were terminal: four completed and two failed. Classic's initial five-update job
1925818, the MIS candidate 1925820, and the CVRP candidate 1925821 completed with
four saved-state proofs each. The MIS candidate continuation 1926616 completed
update 10 with four exact optimizer/scheduler/RNG restoration receipts and four
new saved-state proofs. Classic continuation 1926496 passed restoration on all
four ranks and reached update 10, then failed `save_compare`; MIS FOARL 1925819
failed that audit at update 5. The planned FOARL continuation remains unsubmitted.
Both failures remain failures. Each allocation requests four A100s for at most
two hours. These statuses are retained in
[diagnostic metadata 015](../analysis_outputs/co_live_resume50_20260909_v1/diagnostic_metadata_015.json).

The [bounded native CPU probe](../analysis_outputs/co_live_resume50_20260909_v1/foarl_failed_saved_state_cpu_v1.json)
successfully loaded and semantically hashed the failed FOARL job's optimizer and
scheduler files. It used 6.44 CPU seconds and initialized no CUDA context. This
rules out those particular saved-file loading/hash errors in the later probe;
it cannot recover lost live rank state or identify the original failing check.
The original audit suppressed rank-local exception details. A separately
declared reproduction, job **1926876**, uses an observability revision with the
same acceptance rules. It completed successfully after **901 seconds**, with
four update-5 saved-state proofs, four post-run receipts and a complete chunk
receipt, using 1.0011 allocated A100-hours. No failure diagnostic was recorded in
[observer metadata 005](../analysis_outputs/co_live_resume50_20260909_v1/save_observability_v1/metadata_005.json).
The earlier failure did not recur. Its cause remains
unresolved, and this separate job cannot replace a failed original gate.
All 48 training chunks and 18 evaluations in that historical campaign remain
unsubmitted. Its failed seven-job correctness gate is blocked and the campaign
is superseded; it is not the current 420-file diagnostic campaign.

The [historical runtime declaration](../experiments/co_chunk50_runtime_v2/worktree/docs/co_live_resume50_development_20260909.md)
explains the earlier numerical reproducibility failures and that attempt's direct
saved-to-live restoration checks. Those failures are retained as failures; they
are not evidence about the candidate's optimization quality. The stronger
teacher-search prototypes were audited separately before the current freeze.
A completed, independently reviewed [TRAIN-only opportunity audit](../analysis_outputs/co_live_resume50_20260909_v1/training_opportunity_v1/findings.md)
compared branches from the same constructive incumbent on fixed indices 0–31
per task. The stronger search beat repeated current polishing on 32/32 CVRP
instances and 2/32 MIS instances, tied the other 30 MIS instances, and never
worsened the returned target. All returned targets passed the audit's strict
verifier. The stronger branch used more CPU time: 10.64 versus 1.56 seconds for
CVRP and 4.31 versus 0.59 seconds for MIS, across 32 instances each. These
convenience-sample search gains are neither an equal-compute comparison nor a
trained-policy result. They do not measure acceptance against actual raw policy
siblings or improvement over FOARL.

Those MIS numbers describe the earlier eight-slot search prototype, not the
8,192-state search now included in the 420-file source. The latter's separate
[TRAIN-only audit](../analysis_outputs/co_live_resume50_20260909_v1/mis_bnb_opportunity_v1/findings.md) improved 8/32
instances over repeated old polishing, with 24 ties and no worse results;
mean cardinality rose from 22.96875 to 23.28125. These remain constructor-search
measurements, not trained-policy gains or a FOARL win.

The current own-rank state-preservation contract retains exact comparisons,
cross-rank diagnostics and common scheduler/configuration checks. It does not
explain the old failures, admit old failed checkpoints, or claim exact sampler
or future trajectory equivalence.

Any promising result on reused development data still requires a newly drawn,
overlap-audited confirmation set. One training seed or a comparison with published
numbers from a different benchmark cannot establish SOTA.
