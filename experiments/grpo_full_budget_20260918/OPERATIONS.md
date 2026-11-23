The bounded full-budget controller was started on **19 September 2026 at approximately 00:03 Europe/Paris** (18 September 22:03 UTC). The first two jobs are accepted and pending; no new diagnostic or main-training result was complete at the startup observation.

| Initial allocation | Slurm job | Startup state | Request |
|---|---:|---|---|
| OP classic GRPO, diagnostic to update 5 | 2164644 | PENDING, Resources | 4 A100, 1 hour |
| OP FOARL, diagnostic to update 5 | 2164645 | PENDING, Priority | 4 A100, 1 hour |

The native controller runs as PID **1015764** on `jean-zay3`. Its command and live process were verified. [Startup scheduler records](native_startup_verification_v1.json), [controller bootstrap](native_launch_v1.json), [current collected status](STATUS.md).

The immutable numerical source is
`/lustre/fsn1/projects/rech/vas/uag72kz/grpo_full_budget_20260918/source_v1`,
with manifest SHA-256 `52633632e3151ea95f46adfb2c2fd353c5c08bdccc6298e7e017c93768f543f6`.
It contains 102 inventoried files. The local [package receipt](package_v1.json) binds the exact source archive.

The [launch declaration](launch_declaration_v1.json) has SHA-256
`f8c9a06d80fdacdc3b326e30c78b2d8ce85c10f4fe2e853b548b1160001daacd`.
Its native path is the same filename under
`/lustre/fsn1/projects/rech/vas/uag72kz/grpo_full_budget_20260918`.
The campaign writes exclusively under its new `campaign_v1` directory. No historical numerical source, checkpoint, controller, or existing Slurm job was replaced, cancelled or requeued.

Before submission:

- **139 local tests passed:** 42 profile/entrypoint tests, 26 scheduler tests, 55 evaluator tests, and 16 TSP initializer-gate tests; both Slurm wrappers passed shell syntax checking.
- **28 native CPU checkpoint tests passed**, including the real tiny-Qwen/TRL control save/restore path and optimizer/scheduler restoration checks. These do not substitute for the pending four-GPU diagnostics.
- **20 native data/profile cases passed:** all four methods across all five datasets, exact counts, canonical training-reference validation, identical ordered training-instance hashes and zero overlap with DEV100. Actual diagnostic/main entrypoint parsing was checked for the four completed SFT initializers. TSP entrypoint parsing remains deferred until its complete SFT initializer exists.
- The installed TRL implementation passed its source/version gate. All four ready SFT initializers passed the native dry controller admission. The TSP gate correctly returned pending for the existing SFT continuation.
- Both accepted Slurm requests match the declared account, partition, QOS, GPU/CPU request, one-node range, walltime, source launcher, comment and `Requeue=0`. Slurm reports `NumNodes=1-1` while these jobs are pending; each worker separately checks its actual one-node allocation and four visible A100 GPUs when it starts.

[Native preflight and dry admission evidence](native_admission_v1.json) preserves the 20 detailed case reports and exact bindings. [Continuation tests and sampler replay](../../research/grpo_budget_fairness_20260918/CONTINUATION_AUDIT.md) explain why every MIS run omits a restart at update 750.

The controller may execute only the 428 allocations in the declaration, with at most two of its jobs outstanding and a total reservation ceiling of 4,376 A100-hours. Each task's fresh full run requires successful update-5 and resumed-update-10 diagnostics for **all four methods**. Diagnostics use the full task learning-rate horizon but are never promoted into main training. Every main run evaluates updates 100, 400 and its terminal horizon; evaluation completion gates further checkpoint rotation.

The controller's submission window is 60 days from its recorded start. It neither automatically retries a GPU allocation nor cancels any job. An unsuccessful allocation, failed verification or uncertain submission requires diagnosis; a network/accounting read failure cannot authorize another submission. Existing accepted jobs remain untouched when the controller stops or expires. Source/declaration changes require a separately versioned experiment.

The local read-only collector runs as PID **2971688**, polling every five minutes for at most 61 days. [Collector bootstrap](collector_bootstrap_v1.json) binds its exact implementation; [collector observation](collector_status.json) includes controller liveness. It copies signed reports, updates [STATUS.md](STATUS.md), and produces [terminal comparison status](terminal_comparison_status.json). It never submits, retries, cancels or modifies remote work. Native reports contain all six method pairs, raw-candidate rechecks, signed reference gaps and instance-bootstrap intervals; the local terminal summary explicitly distinguishes descriptive point comparisons from confirmation.

The historical 100-update comparison and its TSP completion watcher continue separately. Its current results remain [here](../rloo_b9_completion_20260918/README.md). TSP full-budget training waits for the pre-existing repaired SFT jobs 1703800→1703803 and their certified step-31,250 endpoint. A complete one-seed development study still does not reproduce the paper's best-of-three procedure or establish a guaranteed all-task improvement.
