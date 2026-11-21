The next independent check will use **300 fresh CVRP instances and 300 fresh MIS instances**, generated with seeds **2026090701** and **2026090702**. This document and its accompanying self-hashed [plan v2](../analysis_outputs/grpo_incumbent_20260907_v1/confirmation_plan_v2.json) declare the data and reference procedure before those instances or any model outputs exist. They do not assert that generation, certification, overlap checking, model binding, or confirmation has passed.

Plan v2 makes one pre-generation provenance correction: audit and freeze now verify every executing helper file against the declared hashes before validating data and again before writing their final receipts. Retained source copies remain independently checked. The original prospective plan and its exact source/document copies remain under `analysis_outputs/grpo_incumbent_20260907_v1/confirmation_plan.json` and `holdout_pre_guard_source_v1/`; no holdout was generated under that version. Sampling, solver budgets, source coverage, reference acceptance, and statistical rules are unchanged.

This is a separate two-task confirmation study of the fixed development recipes. It does not replace the repository's five-task, at-least-1,000-instance publication contract. The inspected 100-row data-v3 split remains development data. No method, checkpoint, solver budget, sample count, seed, or reporting rule may be selected using the new model outputs.

The sampling implementation is the unchanged `_cvrp_row` and `_mis_row` in [generate_iid_benchmark.py](../scripts/generate_iid_benchmark.py). Each task starts its own `random.Random(seed)` and `numpy.random.RandomState(seed)` stream, then takes exactly the first 300 generated rows in order. The runtime must match the previous generator's NumPy 2.3.2 and NetworkX 3.5; record the interpreter, package identities, full source snapshot, and generated row hashes. The task seeds change; the generating distribution does not.

| Task | Fixed distribution |
|---|---|
| CVRP | Uniform integer customer count 10–100, plus depot; equal choice of uniform, two-component Gaussian mixture, and three-component Gaussian mixture coordinates. Coordinates use the existing min-max scaling/truncation to integers. Demands are uniform integers 1–10, depot demand zero. Capacity is `int(Triangular(3,6,25) * mean(customer demands))`. |
| MIS | Uniform integer node count 10–100; equal ER/BA choice. ER probability is uniform in [0.1,0.4]; BA attachment is uniform in {1,2,3,4}. Keep the existing graph, adjacency, and prompt rendering. |

Instances must retain their structured data, prompts, generation index, task seed, distribution family, ordered semantic identity, and normalized benchmark-row identity. A duplicate or overlap blocks admission of the entire draft. Do not silently regenerate, replace, discard, or reorder rows to obtain a favorable or easier set. A separate prospective amendment would be needed if the declared set cannot pass its gates.

Reference construction uses at most four worker processes and one solver thread per worker. Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`; use CPU resources without model inference. Preserve intermediate versions and native evidence. The existing builder writes a development-only staging manifest; it must remain labeled that way. A separate new confirmation manifest will bind the completed outputs and this declaration.

For CVRP, use the existing deterministic Clarke–Wright/raw-distance 2-opt initializer (100 passes maximum), then OR-Tools 9.15.6755 guided local search for **10 seconds per instance**, using its existing rounded integer matrix with distance scale 1,000,000. Independently check every route for exactly-once customer coverage, depot boundaries, capacity, valid indices, and finite raw Euclidean distance. Keep the best strictly feasible raw-distance candidate.

Run the existing LKH-3.0.8 refinement procedure on **all 300 CVRP instances**, using seeds 101 and 202, one run per seed, 10 seconds per run, at most 1,000,000 trials, four workers, distance scale 1,000, precision 100, and at most five seconds external timeout grace per run. This refinement is fixed in advance: the previous data-v3 refinement materially improved 54 of 100 references and had five additional changes no larger than `1e-8` in raw distance. Keep the minimum independently recomputed raw float64 distance among the initializer, OR-Tools candidate, and both valid LKH candidates; keep the earlier candidate on exact equality. Record every candidate and its validity even when it loses. A failed optional LKH candidate does not invalidate an already verified parent reference, but its failure must remain visible.

CVRP labels are **verified heuristic references**, never certified optima. The reference string should serialize the chosen recomputed distance with 17 significant digits. Report CVRP gap-to-reference, retain negative gaps if a model beats the heuristic, and never revise references after seeing model outputs. Report scale/rounding metadata separately for OR-Tools and LKH; neither integer search cost is the evaluation objective.

For MIS, use the existing binary CP-SAT model and OR-Tools 9.15.6755: one worker, fixed solver seed 20260905, zero absolute/relative gap limits, **5 deterministic seconds and 30 wall seconds per instance**. The solver seed deliberately remains distinct from the new instance-generation seeds. Accept an optimum only with native `OPTIMAL`, an integral best bound exactly equal to the independently verified set cardinality, and retained native response statistics. Every vertex must be unique, in range, and independent.

Refine every initially uncertified MIS row once, in ascending generation-index order, using **180 deterministic seconds and 300 wall seconds**, the same solver seed, and at most four concurrent workers. The bounded automatic refinement allowance is at most 20 rows (at most 6,000 additional solver-seconds). If more than 20 need refinement, if any of the attempted refinements remains uncertified, or if the global reference budget expires, retain every row and receipt and block confirmation. Do not replace hard graphs or evaluate only the certified subset. This follows the prior budget: its sole initially uncertified graph certified in 14.58 wall seconds under the larger limit.

The global reference-construction watchdog is two hours, including initial solving, refinement, and process cleanup; terminate child processes on expiry. The worst declared solver allocations are 3,000 seconds for initial CVRP, 9,000 for initial MIS, 6,000 for LKH plus at most 3,000 timeout-grace seconds, and 6,000 for bounded MIS refinement. Their sum is 27,000 worker-seconds, or 112.5 minutes at perfect four-worker utilization; startup, the deterministic initializer, and scheduling are additional. Thus completion within two hours is **not guaranteed**; an unfinished run blocks admission. The streaming overlap audit is a separate CPU-only stage, with no more than four cores, one full scanner process, and a two-hour watchdog. No job has been submitted by this plan.

The overlap audit must newly compare the 300 semantic hashes per task against all rows of each of these four mandatory source classes:

| Forbidden source | Rows per task to check |
|---|---|
| Actual corrected SFT training source | 500,000 CVRP and 500,000 MIS |
| Actual RL source, including rows excluded later by sampling/filtering | 2,000 CVRP; **3,200 MIS** for the source in the retained audit |
| Public evaluation source | 300 CVRP and 300 MIS |
| Inspected data-v3 development split | 100 CVRP and 100 MIS |

The standalone plan records the exact source paths, known byte identities, and development manifest identity. Reconcile these against the current SFT/checkpoint and training-campaign receipts before scanning; append any additional sources actually used rather than weakening coverage. The old `forbidden_sources.jean_zay.json` is stale for MIS: it says 2,000 rows, whereas its retained completed audit scanned 3,200. Reusing its number would fail the new audit.

Use `semantic_instance_sha256` from [benchmark.py](../src/clasp/data/benchmark.py), requiring structured data for every fresh row and allowing prompt-only extraction for historical sources. This identity strips reference answers and prompt formatting and normalizes the task's instance fields; it catches the same labeled optimization instance under different reference text. It is not a proof against arbitrary graph isomorphisms, CVRP customer permutations, or geometric equivalences. Hash each forbidden file before and after streaming, require its declared row count and complete successful semantic parsing, record its ordered semantic digest, and fail on any overlap. A changed source, missing source, parse failure, duplicate candidate, or incomplete scan also blocks admission. A clean old100 audit cannot establish disjointness of the new300 set.

The new [holdout builder](../scripts/build_grpo_incumbent_holdout_20260907.py) has separate `build`, `audit`, and `freeze` phases. Its build phase freezes the declaration and all 600 unlabeled instances before solving, retains all initial/refinement attempts, and writes every row even when a reference remains unresolved. Its freeze independently replays references and best-candidate selection, requires all 300 MIS certificates, and binds the eight-source overlap audit. It explicitly requires a later model predeclaration before evaluation and makes no confirmation or SOTA claim.

The recorded command is to be launched only after tests and review pass; it has not been executed:

```text
timeout --signal=TERM --kill-after=30s 2h
  /tmp/clasp_incumbent_holdout_20260907_env/bin/python
  scripts/build_grpo_incumbent_holdout_20260907.py build
  --plan analysis_outputs/grpo_incumbent_20260907_v1/confirmation_plan_v2.json
  --output-root data/grpo_incumbent_holdout_20260907_v1
  --lkh-binary analysis_outputs/grpo_incumbent_20260907_v1/holdout_runtime/LKH
```

The subsequent `audit` uses `--output-root` plus optional `--forbidden-sources` with a byte-identical source-replica path mapping; all eight declared source identities and row counts remain mandatory. Run its full training-data scan on bounded Jean Zay CPU resources, where the actual corrected training corpora exist. `freeze --output-root ...` reopens its evidence and refuses uncertified MIS rows or missing audits. Both stages create new receipts rather than updating old ones.

The LKH executable above is a byte-identical copy of the retained data-v3 binary, SHA-256 `4956791376d8e7a23735d848ae146b32fe66f137e1b958b0ddf73831bbc03e6b`. The historical source copy is mode0440 and remains unchanged; only the new runtime copy has execute permission. The builder checks both its byte identity and executable permission before creating its output directory.

For comparison, the following exact argument vector remains valid for the **older initial development-labeled reference staging build**. It is not the new holdout admission path and is recorded only:

```text
<pinned_cpu_python> scripts/build_fair_development_benchmark.py
  --output-root <new_confirmation_workspace>/reference_stage_v1
  --instances 300 --cvrp-seed 2026090701 --mis-seed 2026090702
  --workers 4 --cvrp-seconds 10
  --mis-deterministic-seconds 5 --mis-wall-seconds 30
  --ortools-site-packages <pinned_ortools_9.15.6755_site_packages>
```

Do **not** run the old CVRP refinement/finalization commands against 300 rows: [refine_fair_cvrp_references.py](../scripts/refine_fair_cvrp_references.py) and [finalize_fair_cvrp_reference_manifest.py](../scripts/finalize_fair_cvrp_reference_manifest.py) explicitly require 100. Their verified solver/serialization helpers can be reused by a separately reviewed 300-row orchestrator. Similarly, the old overlap CLI requires exactly the three old source labels and cannot establish the four-source-class contract above. The old MIS refinement CLI supports one row and should not be used to silently select a favorable subset. No historical script, receipt, or frozen dataset is to be rewritten for this study.

Before any model output, create a new, create-only confirmation admission receipt binding: this plan hash; source/runtime/solver identities; all 600 ordered instance and row hashes; finalized datasets and reference evidence; 300/300 certified MIS optima; CVRP heuristic status; the complete zero-overlap audit; the raw decoder/comparator implementations and arguments; and all model/checkpoint/tokenizer inventories. It must preserve intermediate development receipts as lineage and use a new identity, without changing their stated purpose. Admission is specific to this two-task300 study and is not the older five-task publication freeze.

The planned models are the final step-100 endpoints of classic GRPO, new verified-incumbent GRPO, and FOARL-f638 development recipes, with identical task SFT initialization and declared training seed 43. Recipe or checkpoint selection may use development data only; any such decision and its rationale must be finalized in a separate model predeclaration **before the first heldout model output**. Bind exact checkpoint inventories and native completion/distributed-consensus receipts before opening the confirmation evaluator. No change may use this holdout's outputs. Data generation may proceed while development training finishes.

The planned raw-generation evaluation keeps the prior K=8 sampler: seed 271828, temperature 0.7, top-p 0.9, top-k 50, repetition penalty 1.0, float16 inference, Jiang no-reason prompt, MIS response-output mode 0, 3,000 completion tokens, and 29,768 prompt tokens. Use the same decoder and strict verifier for all arms; no repair, search, constraint filtering, or extra candidate budget. Finalize remaining implementation arguments in the pre-output admission receipt, not after observing scores. Three arms × two tasks × 300 rows × eight candidates is 14,400 candidate generations.

Prespecify taskwise raw-K8 feasibility as the fraction of instances with at least one strictly feasible candidate; candidate feasibility is a secondary instance-clustered metric. Report quality as the mean relative reference gap of each arm's best feasible candidate on its own successful instances, plus a paired gap comparison on the same jointly solved instances and both explicit denominators. A missing feasible solution must not acquire a zero gap or disappear from feasibility reporting. Report both taskwise contrasts against classic GRPO and FOARL, including losses and ties, paired confidence intervals, and the feasibility discordant wins/losses with an exact two-sided binomial (McNemar) test. Keep feasibility and quality separate rather than collapsing them into a score.

Use 10,000 paired instance bootstrap resamples with fixed seed 2026090703 for descriptive 95% intervals. Resample instances with all eight candidates together; never treat the eight samples as independent benchmark rows. For gap, resample the explicit jointly successful cohort and report its size. Retain all four task/baseline comparisons in the report. If declaring simultaneous significance, report a Bonferroni 99.375% interval for each of the eight metric contrasts (four feasibility and four gap contrasts), and Holm-adjust the four exact feasibility-test p-values; describe each calculation and do not substitute nominal significance for a joint claim. Freeze the implemented comparator in the model admission receipt before evaluation.

A favorable point estimate, a confidence interval excluding zero, and a repeated-seed result are different claims. The old 100-row CVRP baseline solved 96 instances, leaving only four possible discordant feasibility wins; even winning all four gives a two-sided exact p-value of 0.125. The new 300-row denominator improves resolution but does not guarantee adequate power. One training seed and these two tasks do not establish broad SOTA or training-seed robustness.

Only data/reference implementation reviews and model-free checks may occur before that admission gate. Once any confirmation model output is inspected, this set cannot be reused to tune the method or select a checkpoint while retaining an unseen-confirmation claim.
