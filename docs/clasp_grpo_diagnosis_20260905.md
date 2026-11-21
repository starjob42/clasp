# CLASP versus GRPO: diagnosis and historical development evidence

The fresh seed-43 CVRP comparison now also does **not** demonstrate that CLASP
beats GRPO on both feasibility and gap: GRPO solves 98/100 fresh instances,
while anchored v3 and the v4 feasibility correction each solve 96/100. V4's
jointly feasible gap change is -0.057009 percentage points, with a 95% paired
interval of [-0.226668,+0.100914]. The complete verified fresh table and raw
evidence are in the
[CVRP triplet audit](../analysis_outputs/clasp_grpo_fair_dev_20260905/audit_cvrp_complete_triplet_v1/README.md).
The MIS part is also complete: GRPO, v3 and v4 solve the same 95/100 instances with identical selected objectives and 2.300122% gap. The [complete original study](clasp_grpo_fresh_development_results_20260905.md) reports all eight arms and paired uncertainty. A separately declared [B18 follow-up](clasp_grpo_b18_followup_20260905.md) is now training.

The historical seed-42 comparison retained below uses a different training
horizon and repeatedly inspected public evaluation data. In that historical
comparison, anchored CLASP v3 ties best-of-eight feasibility, has slightly
lower candidate feasibility, and has a smaller gap point estimate whose paired
confidence interval includes zero. Its numbers must not be combined with the
fresh seed-43 results.

The raw Jean Zay triplet, configurations, comparison receipts, and a reproducible
reanalysis are retained in
[`analysis_outputs/clasp_grpo_fair_dev_20260905/historical`](../analysis_outputs/clasp_grpo_fair_dev_20260905/historical).
Every raw details hash, config hash, row self-hash, comparison self-hash, and
paired metric was checked locally. The strict comparator's source hash matches
the remote implementation, and its recomputed per-task results match exactly.

All three rows below use CVRP `eval[100:200]`, 100 instances, seed 271828,
eight raw completions in one call, FP16, temperature 0.7, top-p 0.9, top-k 50,
repetition penalty 1, and the same Jiang-F638/no-reason representation and strict
recomputing verifier. Training used seed 42, B=9, global generation batch 36,
four A100 GPUs, and two-hour development allocations. The final policies have
200 optimizer steps. V3 means 100 steps of v2 followed by 100 steps of v3.

| Historical method | Best-of-8 feasible ↑ | Candidate feasible ↑ | Mean feasible gap ↓ | Unconditional solved quality ↑ |
| --- | ---: | ---: | ---: | ---: |
| Classic GRPO (`grpo_z`) | 97/100 (97.0%) | 63.875% | 4.3506% | 0.855234 |
| CLASP anchored v2 | 97/100 (97.0%) | 62.875% | 4.2173% | 0.858471 |
| CLASP anchored v3 | 97/100 (97.0%) | 62.625% | 4.2041% | 0.858941 |

Gap here is nonnegative reference excess gap,
`max((verified distance - reference distance) / reference distance, 0)`,
averaged over each method's feasible instances. The frozen evaluator clips
better-than-reference solutions to zero, so this table does not report signed
improvements over the reference. It is not a certified CVRP optimality gap. Solved quality is
`mean(feasible * exp(-3 * max(gap, 0)))`; failed instances contribute zero.

V3 versus classic has one fail-to-pass and one pass-to-fail transition. The
paired best-of-eight difference is 0 percentage points, with a 95% bootstrap
interval of [-3, +3] points and exact McNemar p=1. Candidate feasibility changes
by -1.25 points, interval [-3.625, +1.125]. On the 96 jointly feasible instances,
classic's mean gap is 4.3222%, versus v3's 4.1453%; the paired change is
-0.1769 points, interval [-0.4129, +0.0487]. Solved quality changes by +0.003707,
interval [-0.019614, +0.026190]. None establishes superiority. These intervals
resample paired instances and do not estimate training-seed uncertainty.

The verified reanalysis also shows where the remaining failures occur:

| CVRP size | Instances | GRPO feasible@8 | V3 feasible@8 | GRPO mean feasible gap | V3 mean feasible gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤25 | 17 | 100% | 100% | 0.9041% | 0.7163% |
| 26–50 | 31 | 100% | 100% | 3.2021% | 3.0857% |
| 51–75 | 25 | 100% | 100% | 5.7505% | 5.6284% |
| >75 | 27 | 88.89% | 88.89% | 6.8172% | 6.6356% |

Several structural differences can limit the original canonical CLASP's
short-horizon learning. They do not establish which difference caused a
measured performance change, or that a particular replacement will improve
held-out performance.

1. **Best-of-eight credit is much sparser than per-candidate GRPO credit.**
   With B=9 and K=8, each subset removes one completion. Writing `U_-j` for its
   utility, the applied canonical coefficient is
   `K*A_i = 9*(mean_j U_-j - U_-i)`. For successful groups, its quality term is
   driven by the best-versus-runner-up margin. When the best objective ties,
   removing any one tied winner leaves the maximum unchanged and that quality
   signal vanishes. The August 31 MIS SFT screen found tied best objectives in
   68/86 successful groups and only 4.98 unique completions per eight.
   Classic GRPO retains within-group objective comparisons among the other
   feasible completions. Exact deployment alignment does not imply lower
   variance or faster learning at a short horizon.
2. **The failure multiplier often has no policy gradient.** A B=9/K=8 group
   gives nonzero binary-failure credit only when exactly one completion is
   feasible. When all nine fail, every subset incurs the same failure penalty,
   so its centered credit is zero regardless of lambda. Learning then depends
   on the violation-recovery ranking; a tie in that ranking also gives zero
   credit. With at least two feasible completions, all eight-sized subsets
   succeed and the failure term again vanishes. Increasing lambda alone
   cannot fix the all-failure regime.
3. **The alpha-0.5 experiment also changed relative sequence weighting.**
   Its fixed-horizon reduction multiplies each sequence's mean-token gradient
   by `sqrt(length/3000)`. For a 300-token output this is approximately 0.316.
   The historical 100-step logs show mean raw gradient norms of 0.06559 for
   alpha-0.5 versus 0.23811 for classic GRPO, but the estimator also changed,
   so this difference cannot be attributed to alpha alone. Both runs used
   AdamW with 8-bit optimizer state. Adam's normalization approximately cancels
   a common positive gradient scaling; epsilon, clipping, and numerical
   effects limit exact invariance. No parameter-update or Adam-step norms
   were recorded, so the smaller raw gradient norm does **not** establish
   proportionally smaller parameter updates or slower learning. The weights
   vary with sequence length and can still change gradient direction. Alpha
   zero restores the comparator's reduction. The configurations, complete
   terminal logs, and reproducible calculation are retained in the
   [historical optimizer audit](../analysis_outputs/clasp_grpo_fair_dev_20260905/current/alpha05_optimizer_audit_v1/audit.json).
4. **Raw utility and standardized GRPO credit have different scales.** V2
   adds 0.25 times raw canonical credit to unit-scale group z-scores. Its
   prior CVRP continuation audit found canonical quality magnitude averaging
   only 9.5% of the anchor (median 4.7%), with failure/recovery signals in only
   5.8%/8.8% of groups. V3 norm-matches only the quality residual with a 4x
   cap. This is a practical surrogate and must not inherit an unbiasedness
   claim from the raw canonical estimator. Its modest historical gain does
   not solve the feasibility bottleneck.
5. **Feasibility errors remain structural.** In the latest 800-candidate
   evaluation, classic/v2/v3 have 170/179/172 capacity-overflow candidates,
   60/71/72 missing-customer candidates, and 30/34/38 parse failures. These
   counts overlap. The largest instances remain the main deployment failure
   regime. Changing quality credit alone has not reduced these violations.
   There are no truncated prompts. Five classic and six v2/v3 completions hit
   the 3,000-token cap, all repeating empty routes on the same 20-customer
   instance, which still succeeds at best-of-eight. Increasing context length
   would not address the dominant capacity/coverage failures.

Two historical confounds must not be mistaken for algorithm effects. The
`clasp_strong_v1` comparison changed SFT checkpoint 31,250 to 20,000, B=9 to
B=12, prompt groups per batch from four to three, and several auxiliary credit
terms together. Its lower score does not identify which modification caused
the regression. Older generic launchers also used Alpaca/native-MIS formatting,
whereas the current certified SFT family requires Jiang-F638 and legacy Jiang
MIS response rendering. Checkpoint representation must be explicit in both
training and evaluation.

Current historical starting policies share the same certified SFT adapter
SHA-256 `f8f634c3a59f53bf411bb0dfc85af98dbc16d936a53263cdfcd6e74ce88b25b8`
and checkpoint-fidelity hash. This avoids the older `bias=lora_only`
serialization defect described in
[`lora_bias_persistence_remediation_20260814.md`](lora_bias_persistence_remediation_20260814.md).
That defect is established for historical adapters; it is not an evidenced
cause of the current bias-safe v3 result.

The latest comparison still has limitations beyond sampling uncertainty:

- Classic's first 100 steps came from source v24; v2's first 100 came from
  source v27. Initial adapter bytes and preserved training settings match,
  but this is not a fresh comparison of both methods under one source tree.
  V2/v3 share the exact same first-100-step policy.
- Both 200-step paths extended a completed 100-step linear schedule. The
  continuation begins at zero learning rate, then resumes roughly half the
  original maximum before decaying again. This is a matched staged schedule,
  not an uninterrupted 200-step run whose horizon was frozen at step zero.
- Historical `grpo_z` computes a population-standardized group reward
  (`unbiased=False`). The installed native TRL uses sample standard deviation
  (`std(dim=1)`). At B=9 the former coefficients are about `sqrt(9/8)=1.06066`
  times larger, ignoring epsilon. The historical label denotes a classic
  GRPO objective with a documented normalization convention, not bit-exact
  native TRL. The native repository entrypoint also has different reward
  heads, so silently swapping entrypoints would introduce another confound.
- The public evaluation rows have been inspected repeatedly. They cannot
  establish generalization to an untouched distribution; any SFT overlap
  must also be treated explicitly. An independently generated, semantically
  disjoint development split is required for the next pilot. A later
  superiority claim requires multiple training seeds and separate untouched
  confirmation data.

A small fair next experiment should freeze one new source, one exact SFT
checkpoint per task, identical B=9/global batch 36, token reduction alpha zero,
optimizer, learning-rate horizon, prompt order, seed, verifier reward inputs,
and raw inference budget. Run fresh GRPO and one named CLASP correction for
100 steps on CVRP and MIS using four-A100 two-hour development jobs, followed
by the same strict best-of-eight evaluation on fresh disjoint instances. Keep
the unmodified baseline and its exact normalization visible. Report both
candidate and best-of-eight feasibility, jointly feasible gap, and solved
quality, including size bands and paired uncertainty. The desired result is
an empirical target; it cannot be guaranteed by an implementation change.

Reproduce the retained historical table with:

```bash
.venv_iid/bin/python analysis_outputs/clasp_grpo_fair_dev_20260905/historical/recompute_historical.py
```

This verifies the retained raw triplet and emits `historical_table.json`,
`historical_table.csv`, and `historical_size_table.csv`. Exact remote training
contracts and hyperparameters, their byte hashes, and installed TRL
normalization evidence are retained under `historical/training_provenance/`.

Fresh development data has also been generated before any model output on its
instances: [`development_fair_cvrp_mis_20260905_v3`](../data/development_fair_cvrp_mis_20260905_v3/README.md)
contains 100 CVRP and 100 MIS instances from the repository's existing
distribution generators, with independent fixed seeds. CVRP references take
the smallest strictly verified raw Euclidean cost among the initial ten-second
ORtools reference and two fixed ten-second LKH searches, seeded 101 and 202.
LKH materially improved 54 references; five further positive differences are
floating-point roundoff, and 41 parent references were retained. The average
relative reference decrease over all 100 instances is 0.826798%. All 200 LKH
candidates passed the strict verifier, with no solver failures or timeouts.
CVRP gaps are to these verified heuristic references, not certified optima.
All 100 MIS
references have native CP-SAT optimality certificates; one reference required
a separately frozen proof refinement, which left the evaluation JSON bytes
unchanged. V3 leaves every instance and non-reference prompt field unchanged;
its MIS dataset and certification evidence are byte-identical to v2.
The completed exact source-overlap audit scanned 1,005,800 rows
across the actual clean 500k SFT inputs, RL training inputs, and public
evaluation inputs and found zero semantic overlaps with the new instances.
Job 1783798 completed on one A100 development allocation in 5:28. A separate
verified receipt binds the unchanged instance set through v2 to the refined v3
reference manifest
`8d2565e29e60a1f5e12fe7fa3635bd2739cc108e7eb2415f112a8941d6ab4752`.
The v3 zero-overlap audit self-hash is
`fa10f2a70456223258d83e5d758acdfc552ac5a03681436a044f9969a072a7b7`.
Both earlier dataset versions, their receipts, and the full source scan remain
available. See [`fresh_reference_summary_v3.json`](../analysis_outputs/clasp_grpo_fair_dev_20260905/fresh_reference_summary_v3.json)
for file hashes, counts, solver statuses, and audit identities. This new data does not
imply that the current certified SFT checkpoint overlaps the public split;
that is a separate empirical question.

The fresh evaluation wrapper must select the SFT representation profile for
SFT checkpoints and the CLASP profile for RL checkpoints. Exact recorded SFT
metadata uses empty strings for two TSP switches; RL records `"0"`. The frozen
evaluator validates these switches even when evaluating CVRP or MIS, so using
the CLASP profile for SFT raises an error before generation. Both corrected
profiles render the same relevant CVRP/MIS representation, but the frozen
protocol fingerprint retains the two unrelated raw TSP strings. SFT results
therefore remain descriptive when that strict fingerprint differs; the three
RL arms remain eligible for strict paired comparisons. The
[`stage replay receipt`](../analysis_outputs/clasp_grpo_fair_dev_20260905/fresh_stage_representation_replay.json)
verifies all four recorded contracts against the exact remote source without
loading a model. This does not weaken the frozen comparison checks.

The first live CVRP SFT and GRPO evaluations also record different tokenizer
artifact identities: checkpoint directory names and the serialized
`tokenizer.json` and `tokenizer_config.json` differ. The RL files retain left
padding, truncation at 29,768 tokens, and `fix_mistral_regex=true`; the SFT
files retain right padding and no backend padding/truncation configuration.
The shared evaluator explicitly selects left padding and enables the regex
option for both stages, then applies the same explicit prompt truncation
limit. Loaded vocabulary, special-token, and chat-template fingerprints
agree. The first 30 corresponding CVRP records have identical prompt token
counts and truncation flags, with no truncation; records do not contain input
token-ID hashes, so this count check alone does not prove token-ID equality.
These observed serialization differences provide an additional reason to keep
the SFT row descriptive under the unchanged strict comparator. The completed
CVRP GRPO and v3 step-100 checkpoints have byte-identical copies of all six
tokenizer artifacts, including both configuration files; the strict RL
comparison has no tokenizer mismatch between those two arms. This runtime
observation does not alter the frozen evaluation addendum or stage replay
receipt.
