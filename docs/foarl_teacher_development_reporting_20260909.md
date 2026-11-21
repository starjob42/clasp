# FOARL control versus verified teacher: development reporting

`scripts/compare_foarl_teacher_dev.py` reports one matched task and checkpoint milestone from the declared FOARL control/teacher campaign. It creates a new `comparison.json`, `table.csv`, and `report.md` only after both endpoints pass admission. It submits no jobs, generates no completions, applies no repair, and changes no training or evaluation artifacts.

The cohort is the existing decontaminated development set: 100 instances per task, eight raw candidates per instance, generation seed 271828, and training seed 43. Milestone 400 is the declared primary development endpoint; 100 and 200 are intermediate learning-curve checks. These are reused development data, not a fresh confirmation set. The prior consumed 300-instance study and Jiang's published table are outside this comparison.

## Admission and invocation

Run the helper from the frozen management snapshot, using the actual whole-file SHA256 of the campaign's `protocol.json`:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B /ABSOLUTE/MANAGEMENT_SOURCE/scripts/compare_foarl_teacher_dev.py \
  --campaign /ABSOLUTE/CAMPAIGN \
  --protocol-sha ACTUAL_PROTOCOL_FILE_SHA256 \
  --task cvrp --step 400 \
  --output /ABSOLUTE/EXISTING_PARENT/NEW_COMPARISON_DIRECTORY
```

The helper checks the externally pinned declaration and manifested implementations before importing them. The controller's read-only `read_only_pair` API revalidates the diagnostic gate, original SFT/model/source/data identities, every 100-step training chunk through the milestone, and both exact evaluation jobs. Actual Slurm `COMPLETED`/`0:0`, allocation, launch, raw details, configuration, and retained index checks are required. A matching filename alone admits nothing. The unchanged raw loader and pair validator check all 100 instance/reference identities, strict verifier, protocol fingerprint, and common K8 budget. Admission is repeated before any report directory is created; differing evidence aborts. Existing output directories cannot be overwritten.

## Metrics and interpretation

All deltas are **teacher minus control**. JSON stores fractions; CSV/Markdown display percentages and percentage points. The report retains:

- Candidate feasibility with denominator 800 and Feasible@8 with denominator 100, including all failed instances. Paired feasibility intervals resample instances, preserving the eight candidates together. Gained/lost instance counts and the exact paired McNemar diagnostic remain in JSON.
- Each policy's mean gap on its own feasible instances and its own denominator. Their difference is explicitly unpaired: changing which instances succeed can change these means independently of objective improvement.
- Mean paired gap difference on the shared-success intersection, its size, both intersection means, and a paired 95% interval. Empty intersections and absent own-success means remain undefined, not zero. Negative paired gap deltas favor the teacher.
- Signed CVRP gap `(cost - reference_cost) / reference_cost`, preserving solutions better than the verified heuristic reference. The native clipped-gap/objective/index best-candidate selection is independently checked before an in-memory signed view is made; raw records remain unchanged. MIS uses `(certified_optimum - size) / certified_optimum` and rejects nonintegral objectives or feasible sizes exceeding the certified optimum. Clipped-gap sensitivity is retained separately.

Statistics reuse the unchanged `compare_development.py` implementation: 20,000 paired percentile bootstrap draws, fixed base seed 20260814, and deterministic task/milestone-derived scope seeds. Signed and clipped views use identical draws. Gap intervals resample the fixed observed shared-success cohort; they do not estimate missing gaps for failed instances. All intervals and retained tests are descriptive and unadjusted across tasks, metrics, and milestones. They quantify development-instance variation conditional on this trained seed, not training-seed uncertainty. No significance flag, SOTA claim, or general superiority claim is produced.

## Compute and teacher telemetry

Training allocation sums every completed 100-step, four-A100 chunk through the endpoint, including the 300-step chunk when reporting step 400. Evaluation cost is that endpoint's one-A100 K8 job only. Actual scheduler elapsed time, allocated GPU/CPU hours, billing-unit hours, and job identities are retained separately. Earlier evaluations, SFT, diagnostics, other tasks, and total project cost are excluded. Equal updates and candidate budgets do not imply equal compute; allocated GPU hours include loading, CPU work, and idle time.

Teacher diagnostics come only from the admitted latest `trainer_state.json`. Counters gathered globally before logging are summed only with complete optimizer-step coverage. Stock TRL logs rank 0 local microbatch means: multiplying the logged preparation-seconds-per-completion means by nine local rows per optimizer step approximates rank 0 preparation wall work only. It is never multiplied by four ranks or described as global CPU time. Global teacher CPU time and teacher NLL GPU time are unavailable. NLL/length microbatch means include zero or dummy cases and are not presented as an aggregate teacher mean. Missing telemetry stays unavailable. None of these diagnostics is policy performance.

`tests/test_compare_foarl_teacher_dev.py` exercises an own-versus-shared cohort sign reversal, empty intersections, signed CVRP and MIS constraints, unchanged pairing guards, cumulative four-chunk cost arithmetic, incomplete telemetry, source tampering before import, changed admission, binary CSV output, and create-only reporting. The report is prepared for admitted future endpoints; preparing this helper runs no evaluation.
