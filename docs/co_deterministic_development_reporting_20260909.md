# Deterministic three-method development reporting

`scripts/compare_co_deterministic_dev.py` compares the verified-teacher candidate (`foarl_teacher`) against both `foarl_control` and `classic_grpo`. It creates one three-method table and two separately identified paired contrasts. Preparing or running this reporting helper does not submit jobs, generate completions, apply assistance, select checkpoints, or change training/evaluation inputs.

All methods use the original deterministic training source frozen as the 403-file manifest `8fa688e8421f2ef3deda29456ed7b6991a65ab34de492896619df1050d9a81f6`. Later management snapshots add only declared management scripts/tests/docs; every original training file remains unchanged. This report cannot admit the earlier nondeterministic checkpoints as deterministic evidence.

The fixed benchmark is the existing decontaminated development set: 100 instances per task, eight raw candidates per instance, generation seed 271828, training seed 43. Step 400 is the primary development endpoint; 100 and 200 are predeclared intermediate learning-curve checks. These are reused development data, not new confirmation data. The previously consumed 300-instance comparison and published Jiang tables are not substituted for these matched controls.

## Admission and invocation

```bash
PYTHONDONTWRITEBYTECODE=1 python -B /ABSOLUTE/MANAGEMENT_SOURCE/scripts/compare_co_deterministic_dev.py \
  --campaign /ABSOLUTE/CAMPAIGN \
  --protocol-sha ACTUAL_PROTOCOL_FILE_SHA256 \
  --task cvrp --step 400 \
  --output /ABSOLUTE/EXISTING_PARENT/NEW_COMPARISON_DIRECTORY
```

The externally pinned protocol and manifested report/controller/evaluator/statistics bytes are checked before dynamic imports. The new controller's `read_only_pair(..., baseline_method=...)` admits each requested contrast under its shared submission lock. It verifies original SFT/data/source identities, the declared native correctness gate, all 100-step training chunks through the milestone, actual applied deterministic-execution evidence, and the exact raw evaluation jobs. Both requested contrasts must refer to identical candidate-arm evidence.

The unchanged raw loader and strict pair validator retain all 100 instance/reference identities, verifier and protocol fingerprints, and common K8 sampling budget. Candidate outputs are loaded once. After all statistics and cost accounting, both pair admissions are repeated; any changed evidence aborts before creating a report directory. A report output directory is create-only.

The four outputs are `comparison.json`, `table.csv`, `paired_effects.csv`, and `report.md`. JSON retains both complete pair admissions, three unique method rows and cost records, two paired analyses, and implementation identities. The method CSV has exactly three rows. The effects CSV has six rows: three metrics for each of the two candidate-versus-baseline contrasts.

## Metrics and conditional cohorts

All effects are **candidate minus the named baseline**. JSON stores fractions; displayed rates/gaps are percentages and effects are percentage points.

- Feasible@8 uses all 100 instances. Candidate feasibility uses all800 candidates. Failures remain in both denominators. Paired feasibility intervals resample instances, keeping the eight candidates together. Gained/lost feasible instances and the exact paired McNemar diagnostic are retained.
- Each method's own-success mean gap is shown with its own feasible-instance denominator. Differences of these means are explicitly unpaired and may reverse direction relative to a paired gap effect.
- Each candidate-versus-baseline gap effect uses that pair's shared-success intersection and reports its size, both intersection means and paired interval. The two baseline comparisons can have different intersections. They are never presented as one common cohort. Empty intersections and absent own-success means remain undefined.
- CVRP signed gap is `(cost-reference_cost)/reference_cost`, preserving solutions better than the verified heuristic reference. The existing evaluator's clipped-gap/objective/index best-candidate rule is checked before an in-memory signed view is created. MIS uses `(certified_optimum-size)/certified_optimum`, requiring integral values and rejecting feasible sizes above the certified optimum. Raw artifacts remain unchanged; clipped-gap sensitivity is retained in JSON.

Positive feasibility effects and negative paired gap effects favor the candidate. Neither a point estimate nor an own-success mean difference establishes a joint improvement in feasibility and objective quality.

## Uncertainty and milestones

The unchanged `compare_development.py` implementation supplies 20,000 paired percentile bootstrap draws with base seed 20260814. Task, milestone and named baseline determine each scope seed. Signed/clipped sensitivity uses identical draws. Gap intervals resample within each fixed observed shared-success cohort; they do not impute gaps for failures.

Intervals and retained exact tests are descriptive and unadjusted across baselines, tasks, metrics and milestones. Across the two tasks, two baselines and three metrics, step 400 has 12 nominal intervals. They are not simultaneous confidence statements. Intermediate 100/200 checks add further descriptive comparisons. The helper emits no significance flag, joint-win decision, confirmation claim, training-seed superiority claim or SOTA claim. Instance-level uncertainty at one trained seed does not measure variation across training seeds.

## Compute and teacher overhead

Each method is counted once, even though the same candidate appears in both contrasts. Training cost sums every completed four-A100 chunk through the endpoint, including the 300-step chunk at step 400. Evaluation cost is only that endpoint's one-A100 K8 evaluation. Actual elapsed time, allocated GPU/CPU hours, billing-unit hours and unique job identities remain in JSON. Earlier evaluations, SFT, diagnostics, other tasks and total project compute are excluded.

Allocated A100-hours include loading, auditing, CPU work and idle time; they are not GPU-active kernel hours. Equal updates and raw candidate budgets do not imply equal compute. Allocation differences do not isolate teacher overhead.

Global teacher-selection counters are summed only with complete optimizer-step telemetry. Multiplying recorded rank 0 preparation-seconds-per-completion microbatch means by nine local rows per update approximates rank 0 preparation wall work only. It is never multiplied by four ranks or called global CPU time. Global teacher CPU seconds and teacher-NLL GPU seconds remain unavailable. NLL/length means containing empty or dummy microbatches are not reported as whole-teacher means. Missing telemetry stays unavailable, not zero; training telemetry is not evaluation performance.

`tests/test_compare_co_deterministic_dev.py` checks independent paired/own-cohort sign reversals, different baseline intersections, empty intersections, signed CVRP/MIS constraints, strict raw pairing for both baselines, candidate evidence consistency, unique three-method cost accounting, source rejection before import, repeated admission and create-only outputs. The core signed-record and cost/telemetry function bodies are checked against the already reviewed original report implementation.
