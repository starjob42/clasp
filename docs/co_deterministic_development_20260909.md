# Prospective matched CO training experiment, 9 September 2026

The modified policy has **not demonstrated a win over FOARL**. The completed comparison below remains the result until a new, matched evaluation is available. This document describes a successor experiment; it contains no new performance measurements.

## Completed comparison

Same task-specific SFT initialization, 100 RL updates, seed43, 300 previously unused instances per task, eight raw samples per instance. Gap is the mean among instances with a feasible sample; those means cover different subsets. Feasibility is feasible-at-eight. The paired comparisons on shared feasible instances did not establish a modified-policy win.

| Task | Policy | Feasible instances /300 | Conditional gap (%) ↓ |
|---|---|---:|---:|
| CVRP | Classic GRPO | 286 | 3.4787 |
| CVRP | Previous modified GRPO | 286 | 3.6078 |
| CVRP | Matched FOARL | 284 | 3.3648 |
| MIS | Classic GRPO | 281 | 2.2402 |
| MIS | Previous modified GRPO | 276 | 2.2595 |
| MIS | Matched FOARL | 272 | 1.8979 |

These 600 instances are now consumed evaluation data and cannot select the successor. Full definitions, paired uncertainty, compute accounting, and separate published-paper numbers are in [the completed findings](clasp_grpo_findings_20260907.md). Published Jiang results use a different cohort and training/selection budget; juxtaposing their percentages with ours cannot establish a matched win.

## Why test this successor

* CLASP's binary group-relative term frequently has no contrast. With nine training samples and an eight-sample set objective, its feasibility credit is concentrated in the one-feasible-sample case. All-invalid and sufficiently feasible groups have no such relative signal. This is an algebraic limitation, not evidence that every other loss term vanishes.
* The previous teacher could train on a candidate worse than another sampled sibling. The successor compares targets with the verified best sampled incumbent for each prompt. Fully feasible groups require a strict improvement; mixed-feasibility groups may also use a target at least as good as that incumbent, and all-infeasible groups may use a verified repair or constructive solution.
* Teacher loss must be normalized over the complete optimizer window and distributed ranks. The implementation now does this and has direct value/gradient distributed tests, including an empty rank. The stock GRPO objective and sampling remain intact.
* The previous 100-update experiment used only400 prompt groups. A fixed400-update experiment tests a longer common training budget for all three methods. A longer schedule is a hypothesis, not an assured improvement.

The candidate is FOARL's released two reward heads and frozen SFT KL reference, plus weight0.05 on verified solution token likelihood. Classic GRPO retains the strict objective reward and beta0. FOARL control uses the same trainer with teacher weight0 and beta0.05. The candidate's extra verified-search and likelihood computation must be reported; equal rollout count does not imply equal training cost.

## Correctness issue isolated before the successor

Independent native MIS teacher runs diverged before restoration:411 saved trainable tensor hashes differed at step5 between full10 and staged5. The optimizer serialization also differed. Different RNG file serialization alone was not treated as proof of different RNG states. This pre-existing policy difference prevents those two runs from isolating checkpoint restoration. The pending continuation1922791 was cancelled while still pending; its original source, contracts, logs, and checkpoints were retained. The prior four-arm production declaration has no submitted jobs and is not the successor declaration.

The successor applies strict PyTorch2.9.1 deterministic algorithms before CUDA initialization, with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and the existing `PYTHONHASHSEED=43` before worker startup. It preserves optimizer, seed, sampling, attention backend eligibility, and method-specific rewards. Unsupported nondeterministic operations must raise. No exact cross-hardware reproducibility is assumed from these settings alone.

Before testing continuation, independent full and staged runs must have exactly equal semantic model, optimizer, scheduler, and all four ranks' Python/NumPy/Torch CPU/CUDA RNG states at step5. After that, a5-to10 continuation is compared with the uninterrupted10 endpoint. The classic beta0 branch saves its own lineage immediately at each checkpoint, after fidelity checks, so its intermediate step5 checkpoint can be admitted without reconstructing old telemetry later.

Six initial correctness jobs and two conditional continuations are declared prospectively: MIS teacher full10/staged5, MIS classic full10/staged5, CVRP teacher5, and MIS FOARL control10. Each job requests four A100 GPUs in Jean Zay's development queue and at most two hours. Submission is explicit, quota-aware, and never automatically retried. These short training diagnostics cannot establish a feasibility or gap improvement.

## Performance admission and reporting

All three methods must use identical task-specific SFT bytes, training instances/order, optimizer schedule, rollout count, token caps, and evaluation instances/seeds/sampling. The intended common horizon is400 optimizer updates, with100-update resumable development chunks. Evaluate raw best-of-eight at100/200/400 on the existing development100 per task; step400 is the primary development comparison. Report coverage, candidate feasibility, conditional gap, shared-feasible paired gap, uncertainty, and actual allocated A100 hours separately.

A new final test cohort and additional training seeds are required after selecting and freezing a promising recipe. This experiment does not use the consumed300-per-task cohort for further selection, add inference assistance to only our method, or fill future table cells with predictions.
