The 100-update study is too short to answer the full-budget FOARL question. This experiment compares **classic GRPO, FOARL, existing RLOO B9, and RLOO B9 with the corrected quality reward** at a common, prospectively fixed budget. No method has yet demonstrated a win on all five tasks.

**Current matched-budget CLASP-vs-FOARL table, its provenance and the exact
evidence-availability caveats are in [`RESULTS_matched_budget.md`](RESULTS_matched_budget.md)**,
recomputed from [`evidence/`](evidence) by [`verify_matched_table.py`](verify_matched_table.py).

**The controller is active under the 26 September RLOO-priority management amendment.** OP continuation job 232465 and MIS job 232466 were admitted alongside the existing JSSP allocations. [Live collected status](STATUS.md), [RLOO continuation](../rloo_epoch_continuation_20260926/README.md), [original launch evidence](OPERATIONS.md).

| Task | Released FOARL prompt exposures per default run | Updates for every new B9 method | Generated training completions per method |
|---|---:|---:|---:|
| CVRP | 2,000 | 500 | 18,000 |
| MIS | 3,200 | 800 | 28,800 |
| JSSP | 3,200 | 800 | 28,800 |
| TSP | 3,200 | 800 | 28,800 |
| OP | 5,000 | 1,250 | 45,000 |

The release budgets are inferred from the author's pinned one-epoch defaults, the actual released task sizes, and TRL's repeat sampler. They are not recovered author run logs. The release uses eight rollouts per prompt; this study keeps B9 for all four methods and therefore generates **12.5% more completions** at the same prompt exposure. All four methods have equal update, prompt and completion budgets within each task. Their sequence lengths and KL computations differ, so equal tokens or GPU time are not claimed. [Reproducible FOARL budget audit](../../research/grpo_budget_fairness_20260918/FOARL_BUDGET_AUDIT.md).

Every method starts fresh from the same task-specific SFT policy. The old 100-update learning-rate schedules ended at zero; their terminal policies are preserved as a separate experiment. New runs declare their full horizon before update one, with the same batch of 36 responses, nine responses per prompt, four GPUs, RL/data seed 43, AdamW learning rate 1e-6, linear decay, 3% warmup, 0.01 weight decay, Jiang prompts, 3,000 output-token limit and training sampling (temperature 0.7, top-p 0.9, top-k 50). These common nuisance settings intentionally differ from some released defaults; this is an algorithm comparison under a shared B9 recipe, not an exact official reproduction.

The intentional method differences are:

- **Classic GRPO:** strict objective reward, group-standardized advantage, standard GRPO loss reduction, beta 0.
- **FOARL:** released reward heads, group-standardized advantage, standard GRPO loss reduction, beta 0.05 against the fixed initial SFT policy.
- **Existing RLOO B9:** historical graded verifier reward, unscaled leave-one-out advantage, fixed-length Dr.GRPO reduction, beta 0.
- **RLOO B9 with corrected quality reward:** the same RLOO update with `reference_sigmoid_v2`, including the corrected invalid-solution diagnostics and quality calibration against canonically verified training answers.

Including both RLOO rewards separates longer training from the proposed reward correction. The historical RLOO sampling parameters were different; every method in this new prospective comparison uses the common decoder above. In the completed 100-update logs, OP and JSSP had no all-infeasible RLOO groups, so that failure mode alone does not explain their quality deficits. [Algorithm audit](../../research/grpo_budget_fairness_20260918/ALGORITHM_FAIRNESS_AUDIT.md), [existing 100-update comparison](../rloo_b9_completion_20260918/README.md).

TSP waits for the **already queued** repaired SFT chain to reach step 31,250. Its current stable checkpoint is only at 15,000. The campaign does not submit or alter SFT jobs and will admit TSP only after the existing four continuation jobs succeed and the terminal policy passes the source, state, saved-policy and reload-certificate gates. CVRP, MIS, JSSP and OP already have full step-31,250 initializers. [TSP continuation evidence](../../research/grpo_budget_fairness_20260918/TSP_SFT_CONTINUATION_AUDIT.md).

MIS retains the 2,900 clean training instances. Reaching 3,200 prompt exposures repeats 300 instances; excluded evaluation overlaps are never reintroduced. The final MIS chunk is 700→800: a native replay proved that this matches uninterrupted prompt order on all four ranks, whereas restarting at 750 changes that order. Every method saves and checks rank-specific optimizer and scheduler state, the immutable run contract and RNG artifacts. This does not claim bitwise equality of future generated policies. [Continuation audit and native tests](../../research/grpo_budget_fairness_20260918/CONTINUATION_AUDIT.md).

Each task first runs independent 5→10 resume diagnostics for all four methods, using the eventual full learning-rate horizon. Main training starts afresh only after all four diagnostics succeed. Checkpoints are evaluated at 100, 400 and the task's terminal update with the same previously inspected DEV100 instances, eight raw candidates, seed 271828 and no repair. The next training chunk waits for each milestone's canonical evaluation before checkpoint retention can remove that policy. Signed raw-evaluation evidence and policy hashes remain available for paired reports after checkpoint rotation.

Reports include feasibility@1 and @8, feasible-candidate rate, signed reference gap, paired joint-success gaps with instance-bootstrap intervals, and an all-instance quality-success curve. Negative gap means improvement over the declared reference; most references are not certified optima. A one-seed result on inspected development data cannot establish a general win. Matching the paper's best-of-three reporting requires a later symmetric multi-seed comparison, with a fresh confirmation cohort and a common selection rule.

The immutable graph contains **428 maximum allocations**: 40 diagnostic chunks, 328 main-training chunks, and 60 evaluations. The upper bound is **4,376 A100 reservation-hours**, including diagnostics and evaluation; actual usage can be lower. The original admission policy allowed two outstanding jobs and eight across related RL campaigns. The separately bound [26 September management amendment](../rloo_epoch_continuation_20260926/README.md) permits four from this campaign until all RLOO terminal evaluations complete, then restores two, with ten related jobs observed at admission and room reserved for the separate feasible-quality campaign. The development-QOS limit remains in force. The original 60-day submission window and total budget remain unchanged; no automatic GPU retries are allowed, and a failed allocation or audit stops advancement for diagnosis. Existing jobs and queue age are preserved.

Implementation: [training profile](../../src/clasp/baselines/budget_matched_dev.py), [controller](campaign.py), [chunk runner](run_chunk.py), [canonical evaluation adapter](budget_evaluation.py), [native CPU preflight](preflight.py). Deployment receipts and the current status are recorded alongside this file. Training outcomes remain pending until the signed evaluation reports are produced.
