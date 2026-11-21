# Incumbent-teacher / FOARL development screen

This is a new, sample-matched development comparison on CVRP and MIS. It uses
the existing certified task SFT adapters at step 31,250 and the already inspected,
instance-disjoint `data_v3` development benchmark. It cannot establish a Jiang
reproduction, superiority on an untouched test, or state of the art.

| Setting | Fixed value |
|---|---|
| Training arms | `grpo_z`, `grpo_verified`, `foarl_f638_dev` |
| Initialization | Same task-specific corrected Jiang-F638 SFT, base, tokenizer |
| Training budget | Seed 43; 100 updates; B9; 36 raw completions/update; 3,600 completions; 400 prompt groups |
| Hardware | Four Jean Zay A100s; `qos_gpu_a100-dev`; two-hour limit |
| Optimization | Per-device batch 3; accumulation 3; AdamW 8-bit; LR 1e-6; weight decay .01; linear schedule; warmup .03 |
| Representation | Common Jiang-F638 no-reason prompt; MIS rendering flag 0 |
| Sampling | Temperature .7; top-p .9; top-k 50; prompt/completion caps 29,768/3,000 |
| Evaluation | Raw K8; seed 271828; float16; same common prompt and decoder; 100 benchmark instances/task; no repair |
| Checkpoint rule | Fresh SFT initialization; one terminal checkpoint at 100; no resume or automatic retry |

The classic arm keeps the previous strict verified-feasible quality reward and
group-relative sample-standard-deviation credit. The modified arm adds graded
violations and weight-.05 verified teacher NLL, using `feasibility_incumbent`
selection and `optimizer_window` normalization. Explicit `best_per_prompt`
selection is an optional normalization-only ablation and must be separately
declared. Inactive baseline teacher normalization is `microbatch`.

The FOARL arm retains the released two reward heads, their sum, group sample
standard deviation plus 1e-4, token-then-sequence mean reduction, and beta .05
against an independently loaded frozen copy of the exact initial SFT policy.
Its batch, optimizer, horizon, token caps, and common no-reason prompt are
deliberate development adaptations. Its label is a corrected F638 algorithm
port under a common budget, never an official Jiang reproduction. The existing
native/matched-publication FOARL profiles and historical trainer hashes remain
unchanged.

The new launcher is `scripts/train_grpo_incumbent_dev_a100.slurm`. It requires
the declared source-manifest hash, checks every frozen source byte and the
actual Slurm spool script, clears inherited research settings, and records the
executed argument vector, Slurm ID, runtime, initial weights, and method recipe
in a create-once run contract. A separate explicit diagnostic mode permits
exactly five updates and checks gradients and policies after every update;
diagnostic checkpoints are excluded from the 100-update comparison. Full runs
retain exact four-rank policy checks at steps 1, 2, 5, and each multiple of 10.

Equal raw sample counts and token caps do not mean equal actual tokens or
compute. Count generated tokens, selected target tokens, teacher search time,
reference forward work, allocated GPU-hours, and evaluation runtime. Report
candidate feasibility, feasible@8, each arm's conditional gap, paired gap on
jointly solved instances, and unconditional success/quality. Preserve signed
better-than-reference CVRP gaps and identify its references as heuristic;
MIS reference objectives are certified optima.

The existing SFTs completed one epoch over 500,000 cleaned demonstrations at
step 31,250 with the released 2e-4 learning rate, batch 4×4, 20 warmup steps,
seed 42, and pinned historical dependencies. They intentionally correct the
response mask, fresh-start call, and trainable/saved QKV biases; the cleaned
data also differs from the released corpus. These are strong common
initializations, not bitwise replicas of an unavailable author checkpoint.

At the read-only scheduler check on 2026-09-07 07:04 UTC, the dev QoS admitted
10 expanded pending/running jobs per user and 32 GPUs. An unrelated five-task
pending array occupied five slots. Submit at most five new jobs per tick and
count the current expanded queue; never cancel or alter unrelated jobs.

The previous 100-update classic/modified jobs took 75.65/83.52 minutes on CVRP
and 51.97/54.85 minutes on MIS. These support 100-update development chunks;
FOARL's additional reference cost still needs a native timing check. A future
300-update experiment should set `max_steps=300` from the first update, then
use the existing `GRPO_STOP_AT_GLOBAL_STEP` callback at 100/200/300 with complete
optimizer, scheduler, RNG, and checkpoint-fidelity validation. The completed
100-update checkpoints already exhausted a 100-step linear schedule; extending
them is a newly declared second training phase, not the same 300-step trajectory.
