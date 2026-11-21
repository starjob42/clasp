# CLASP

## Revised paper reproducibility

The revised paper workspace and its machine-readable result source of truth are
documented in
[`docs/paper_reproducibility.md`](docs/paper_reproducibility.md). Validate the
current manifest before editing paper tables or claims:

```bash
python scripts/validate_paper_source_of_truth.py
```

**C**onstrained **L**eave-out **A**dvantages for **S**et **P**olicies — training,
evaluation, and evidence tooling for LLM combinatorial optimization (CO).

The **RLOO9 paper** is [main.pdf](paper/rloo9_20261008/main.pdf). This is the single canonical manuscript and is updated in place; see its [sources and evidence guide](paper/rloo9_20261008/README.md).

> **Publication status (2026-08-15).** The historical `--use_clasp` trainer
> paths are development/ablation implementations; they are not the canonical
> ICLR method and no current checkpoint is publication-valid.  The reviewed
> Pareto-anchor formulation is retired because a group-constant additive
> anchor cancels under centering.  The opt-in canonical trainer is now
> integrated and tested: it separates rollout size `B` from deployment size
> `K`, evaluates exact best-feasible `K`-set utility, applies bounded failed-set
> recovery to `v/(1+v)`, emits `K*A`, and commits the failure@`K` EMA controller
> once per optimizer boundary. Three short 0.5B TSP runs and their 31-instance
> held-out development comparison are mechanism/system smokes, not publication
> evidence; the deployment-level feasibility and quality intervals include
> zero. The ideal score-function theorem
> also does not cover the current clipped, token-normalized TRL loss. See
> [`docs/canonical_set_training.md`](docs/canonical_set_training.md),
> [`paper/iclr2027/method_spec.md`](paper/iclr2027/method_spec.md),
> [`experiment_contract.md`](paper/iclr2027/experiment_contract.md), and
> [`source_and_artifact_audit.md`](paper/iclr2027/source_and_artifact_audit.md)
> before launching a claim-bearing run.

Five tasks are supported everywhere: `tsp`, `cvrp`, `op`, `mis`, `jssp`.
The active entry points expose these formula-distinct variants:

| Variant | Entry point | Flags |
| --- | --- | --- |
| Supervised fine-tuning | `clasp-sft` | — |
| Standard GRPO | `clasp-grpo-standard` | — |
| Primal-dual constrained GRPO | `clasp-grpo` | `--use_primal_dual` |
| Canonical CLASP | `clasp-grpo` | `--use_primal_dual --use_clasp --clasp_canonical_set --num_generations 9 --clasp_deployment_set_size 8` |
| Historical CLASP ablations | `clasp-grpo` | `--use_primal_dual --use_clasp` without `--clasp_canonical_set` |

The target ICLR method defines a symmetric best-feasible size-`K` set utility,
uses canonical centered leave-out credit from a larger rollout group `B>K`,
and updates a task multiplier with the signed failure@`K` residual.  Dense
violation recovery is a disclosed failed-set shaping utility; infeasible
outputs never receive objective-quality credit. A new formula-matched
fixed-null arm remains a required set-credit ablation; the existing
Frontier-CDF implementation cannot substitute for it. Legacy dense-rank, Pareto,
component-controller, adaptive-length, and alternate-ratio paths remain only
for explicit attribution experiments.

## Layout

```text
correct_clasp/
├── pyproject.toml
├── README.md
├── src/clasp/
│   ├── config.py              Argparse parsers, cross-validation, CLASP preset logic
│   ├── runtime.py              Run naming, W&B setup, batch/generation math, checkpoint discovery
│   ├── data/
│   │   ├── loaders.py          Per-task JSON loading, dedup, overlap removal, mixed-batch sampler
│   │   └── prompts.py          Alpaca-style prompt templates
│   ├── models/loading.py       Model/tokenizer loading, PEFT/LoRA detection and merging
│   ├── rewards/
│   │   ├── co_tasks.py         Per-task parsing + scoring (single source of truth)
│   │   ├── registry.py         TASK_REWARD_TRIPLETS / TASK_COST_COMPONENTS lookup tables
│   │   ├── heads.py            Mixed-batch reward dispatch, MultiTaskPrimalDualManager
│   │   ├── primal_dual.py      Legacy constrained reward/controllers and component profiles
│   │   └── clasp_math.py       CLASP advantage math as pure tensor functions
│   ├── trainers/
│   │   └── primal_dual_grpo.py PrimalDualGRPOTrainer (the active trainer)
│   ├── set_grpo/               Canonical credit, failure@K control, and legacy Frontier-CDF helpers
│   └── entrypoints/
│       ├── train_sft.py        train_grpo.py        train_grpo_standard.py
│       ├── train_pipeline.py   evaluate.py          audit_data.py
├── tests/                      Reward, advantage, and trainer regression tests (no model loading)
├── checkpoints/{sft,clasp,primal_dual,grpo}/
├── data/                       Local dataset mount point (contents git-ignored)
└── models/                     Local model mount point (contents git-ignored)
```

The dependency flow is one-directional:
`entrypoints → {config, runtime, data, models, rewards, trainers}` and
`trainers → {rewards, set_grpo}`. There are no cycles and no `sys.path` manipulation.

## Install

Python ≥ 3.10.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On GPU clusters, install the PyTorch build matching your CUDA stack first.

**Air-gapped clusters** (no PyPI access) — install against an environment that already
provides the dependency stack:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

Or skip installation entirely and run from the source tree with `PYTHONPATH=src`.
Verified stack: `torch 2.5.1`, `transformers 4.57.1`, `trl 0.24.0`, `peft 0.18.1`.

## Data

```text
<data_root>/<task>/<train|eval>/*.json      task ∈ {tsp, cvrp, op, mis, jssp}
```

Each file is a top-level JSON list. Every row requires non-empty `instruction`, `input`,
and `output`; the structured instance goes in `instance_json` (fallback: `instance`).

```json
[
  {
    "instruction": "Solve the TSP instance.",
    "input": "Node 0, coordinates: [0, 321], ...",
    "output": "Route: [0, 38, 52, ...]",
    "instance_json": [[0, 321], [276, 270]]
  }
]
```

The loader validates required fields, deduplicates exact duplicates, and removes
train/eval overlap within each task. Audit SFT/RL leakage before any long run:

```bash
clasp-audit-data \
  --sft_data_root  data/data_sft \
  --grpo_data_root data/data_rl \
  --tasks tsp,cvrp,op,mis,jssp \
  --fail_on_overlap
```

`--fail_on_overlap` exits non-zero on *any* overlap, which is what you want gating a long
job. Some dataset releases share the SFT and RL **eval** splits. That cross-eval overlap
alone is not training leakage, but it does not establish that either split is held out.
The frozen LLMCoSolver f638 audit finds that the author training union overlaps 299/300
CVRP, 300/300 MIS, and 299/300 OP RL-eval instances; those evaluation files are therefore
context-reproduction assets, not a leakage-free claim denominator. See
[`docs/llmcosolver_f638_public_data_addendum_v2.md`](docs/llmcosolver_f638_public_data_addendum_v2.md).
Drop `--fail_on_overlap` to inspect the per-split breakdown first, and only gate on it once
you have confirmed the `*_train_*` rows are clean. A claim-bearing evaluation requires the
separate preregistered fresh, disjoint IID workflow below.

## Training

The examples below are single-GPU; prefix with
`torchrun --nproc_per_node=<N>` for multi-GPU.

**1. Supervised fine-tuning**

```bash
clasp-sft \
  --model_path   models/Qwen2.5-7B-Instruct \
  --data_root    data/data_sft \
  --output_root  outputs \
  --tasks tsp,cvrp,op,mis,jssp \
  --max_seq_length 6144 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_steps 20000 \
  --report_to none
```

**2. Canonical CLASP (`B=9`, deployment `K=8`)** from an SFT checkpoint

This is the current method path. It exactly enumerates nine size-eight sets,
uses best strictly feasible evaluator quality, and disables legacy
component-dual, warmup, replay, and adaptive-length additions. The command is a
training template, not a claim-bearing campaign contract:

```bash
clasp-grpo \
  --model_path   outputs/sft/<run-name> \
  --data_root    data/data_rl \
  --output_root  outputs \
  --tasks tsp,cvrp,op,mis,jssp \
  --use_primal_dual --use_clasp --clasp_canonical_set \
  --num_generations 9 --clasp_deployment_set_size 8 \
  --clasp_canonical_recovery_scale 1.0 \
  --clasp_canonical_recovery_temperature 0.15 \
  --grpo_loss_type grpo \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 9 \
  --gradient_accumulation_steps 3 \
  --max_prompt_length 4096 --max_completion_length 2048 \
  --max_steps 2000 \
  --report_to none
```

The trainer supports the existing sequence-importance setting and an explicit
matched token-importance ablation. Both currently use TRL's per-sequence
token-mean `grpo` reduction and therefore remain outside the ideal
sequence-action theorem. To reproduce the token-importance pilot, set
`PD_IMPORTANCE_SAMPLING_LEVEL=token` and
`PD_ALLOW_TOKEN_IMPORTANCE_ABLATION=1`. See
[`docs/canonical_set_training.md`](docs/canonical_set_training.md) for the
complete isolation and audit contract.

**3. Historical CLASP-GRPO development path** from an SFT checkpoint

This command does **not** activate the canonical `B>K` ICLR estimator and must
not be used as the headline configuration.  It remains useful for reproducing
and auditing earlier development paths.

```bash
clasp-grpo \
  --model_path   outputs/sft/<run-name> \
  --data_root    data/data_rl \
  --output_root  outputs \
  --tasks tsp,cvrp,op,mis,jssp \
  --use_primal_dual --use_clasp \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_generations 8 \
  --max_prompt_length 4096 --max_completion_length 2048 \
  --max_steps 2000 \
  --report_to none
```

**4. Primal-dual constrained GRPO** — same command with `--use_primal_dual --no-use_clasp`.

**5. Standard GRPO** — `clasp-grpo-standard` with the same arguments minus the
primal-dual/CLASP flags. It accepts `--sft_checkpoint_path` to resolve an SFT output
directory or a `checkpoint-*` directory automatically.

**6. Chained SFT → GRPO** in one process. The pipeline runs SFT, merges the resulting
adapter into the base weights, and initializes GRPO from the merged model:

```bash
clasp-pipeline \
  --model_path     models/Qwen2.5-7B-Instruct \
  --sft_data_root  data/data_sft \
  --grpo_data_root data/data_rl \
  --output_root    outputs \
  --tasks tsp,cvrp,op,mis,jssp \
  --sft_max_steps 20000 --grpo_max_steps 2000 \
  --max_seq_length 6144 --max_prompt_length 4096 --max_completion_length 2048 \
  --sft_per_device_train_batch_size 2 --sft_gradient_accumulation_steps 8 \
  --grpo_per_device_train_batch_size 2 --grpo_per_device_eval_batch_size 8 \
  --grpo_gradient_accumulation_steps 4 --grpo_num_generations 8 \
  --report_to none
```

The merge step writes a full-precision copy of the base model (~15 GB for a 7B), so
`--output_root` needs the space.

### Batch-size constraints

TRL enforces, and `config.py` validates up front:

- `per_device_train_batch_size × world_size × gradient_accumulation_steps` divisible by `num_generations`
- `per_device_eval_batch_size × world_size` divisible by `num_generations`
- `generation_batch_size` divisible by the global batch size
- `--generation_batch_size` and `--steps_per_generation` are mutually exclusive

For multi-task runs keep `per_device_train_batch_size >= 2` — the loader guarantees mixed
local batches, which requires at least two slots per device.

For **single-task historical, noncanonical** CLASP runs, also narrow the
feasibility-warmup targets to the active task, since the legacy presets default
to all five and the validator rejects targets naming an inactive task:

```bash
clasp-grpo --tasks mis --use_primal_dual --use_clasp \
  --clasp_feasibility_warmup_targets mis:0.995 ...
```

## Cluster runs (Slurm / Jean Zay)

`scripts/train.slurm` launches the four historical presets below. Its
`VARIANT=clasp` setting is the noncanonical development path; it must not be
used for the ICLR method without adding and auditing the canonical flags from
the command above. It is the only general launcher in this repository by
design---one parameterized script rather than a directory of near-duplicates.

```bash
sbatch --export=ALL,VARIANT=clasp       scripts/train.slurm
sbatch --export=ALL,VARIANT=primal_dual scripts/train.slurm
sbatch --export=ALL,VARIANT=grpo        scripts/train.slurm
sbatch --export=ALL,VARIANT=sft         scripts/train.slurm

# Evaluate a checkpoint (1 GPU):
CHECKPOINT_PATH=checkpoints/clasp/checkpoint-40 sbatch --export=ALL scripts/evaluate.slurm
```

`sbatch --export` splits its value on commas, so a comma-containing setting such as
`TASKS=tsp,mis` must be exported into the environment first and picked up via `--export=ALL`,
not written inline.

Defaults target Jean Zay's H100 dev queue: 1 node x 4 GPUs, 2 h, `qos_gpu_h100-dev`,
account `vas@h100`, conda env `new_unsloth_env`. Override anything without editing the file:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VARIANT` | `clasp` | `sft`, `grpo`, `primal_dual`, or historical noncanonical `clasp` |
| `CLASP_PROJECT_ROOT` | `$SLURM_SUBMIT_DIR` | repository root |
| `CLASP_ASSET_ROOT` | `.../uag72kz/clasp` | root holding `models/` and `data/` |
| `CLASP_STORAGE_ROOT` | `$PROJECT_ROOT/outputs` | checkpoints and run outputs |
| `BASE_MODEL_PATH` | `$ASSET_ROOT/models/Qwen2.5-7B-Instruct` | base model (SFT only) |
| `SFT_CHECKPOINT_PATH` | `$ASSET_ROOT/models/sft/checkpoint-18400` | RL initialization |
| `GRPO_DATA_ROOT` / `SFT_DATA_ROOT` | `$ASSET_ROOT/data/data_rl` / `data_sft` | datasets |
| `CONDA_ENV` | `new_unsloth_env` | environment activated on the compute node |
| `GPUS_PER_NODE` | `4` | processes per node for `torchrun` |
| `NUM_GENERATIONS` | `8` | GRPO group size |
| `LORA_R` / `LORA_ALPHA` | `64` / `64` | matches the shipped checkpoints |
| `MAX_STEPS` | `40` | step budget |

Multi-node works by raising `--nodes`; the script derives the rendezvous endpoint from
`SLURM_JOB_NODELIST` and passes `SLURM_NNODES` to `torchrun`.

Compute nodes have no outbound network, so the script forces `HF_HUB_OFFLINE`,
`TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, and `WANDB_MODE=offline`.

## Evaluation

```bash
clasp-eval \
  --evaluation_purpose development \
  --checkpoint_path outputs/grpo/<run>/checkpoints/checkpoint-2000 \
  --stage grpo \
  --data_root data/data_rl \
  --tasks tsp,cvrp,op,mis,jssp \
  --max_eval_samples_per_task 300 \
  --seed 271828 \
  --dtype float16 \
  --num_return_sequences 8 \
  --generation_batch_size 8 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 50 \
  --max_new_tokens 3000 \
  --max_prompt_length 29768 \
  --force_no_reason_prompt \
  --save_results --save_details \
  --inference_root inference
```

Results land under `inference/<stage>/<checkpoint_name>/`. The protocol is raw best-of-N
with no inference-time repair or reranking. Evaluation fails closed when checkpoint metadata
does not identify the training prompt contract; use exactly one of `--force_no_reason_prompt`
or `--force_reasoning_prompt` only when you have verified that contract. Saved run labels carry
a content fingerprint over the model, data, prompting, decoding, verifier, and evaluation code.
Sampling uses deterministic per-example seeds derived from `--seed`, so skipping completed rows
during resume cannot shift the random stream for the remaining benchmark instances.

The command above is deliberately a **development** evaluation on previously observed data.
`scripts/evaluate.slurm` also defaults to `EVALUATION_PURPOSE=development`, and
`scripts/evaluate_a100.slurm` explicitly labels its historical `[200,300)` run as
`replication`. Neither launcher silently opts into confirmation or names an unresolved
confirmation asset.

### Publication workflow and current gate

As of 2026-08-14 the five-task IID reference contract is
`BLOCKED_UNRESOLVED_DO_NOT_USE`. No production references, schema-v3 frozen benchmark, or
valid confirmation result exists, so this repository currently supports **no SOTA claim**.
The immutable source snapshots `correct_clasp_sota_20260814_v1` (training smoke) and
`correct_clasp_sota_20260814_v2` (now-stale evaluation/benchmark smoke) are historical
receipts, not confirmation environments. They must not be edited in place. Create and
hash-verify a new immutable snapshot after the solver contract and current identity-binding
code are frozen.

The create-once snapshot procedure, remote prerequisite audit, retired 40-step
provenance, and requirements for a new registry-bound launch are in
[`docs/remote_publication_launch.md`](docs/remote_publication_launch.md).

The only publication-valid order is:

1. Before publication training or development evidence, run
   `scripts/freeze_forbidden_source_registry.py` over every authoritative SFT-train,
   RL-train, and historical-eval tree, then publish the registry/tree hashes. Training
   contracts consume this receipt. Then freeze and run the separate all-five-task
   non-instruction base-specialist campaign with `scripts/freeze_base_sft_campaign.py` and
   `scripts/submit_sft500k_base_specialists_a100.sh`. That campaign remains on HOLD until
   an operator uses `scripts/freeze_remote_environment.py` in the exact four-A100
   allocation to create its read-only, self-hashed Python/package/runtime receipt; every
   worker reopens the receipt before run mutation, inside the training Slurm step, and
   before finalization. Campaign v3 additionally requires one administratively sealed
   immutable-input trust root, an external create-once authority containing the campaign
   and all five run contracts, `save_total_limit=64`, and at least 1 TiB of retained-matrix
   capacity. Every completed 500-step checkpoint carries exact checkpoint and resume-state
   receipts for adapter, Trainer, optimizer, scheduler, and all four rank RNG files. Resume
   pins the verified checkpoint directory descriptor through Trainer load, audits the full
   loaded state and immediate RNG restoration, and forbids fallback from a newer incomplete
   checkpoint. No real v3 authority/environment/campaign is checked in, so the templates
   remain HOLD. After all five campaign-bound cells
   finalize, freeze their exact base/model/data lineage with
   `scripts/freeze_model_initialization_registry.py --base-sft-campaign <receipt>`; bind its
   self-hash into the advancement policy before launching any RL job. See
   [`docs/model_initialization_registry.md`](docs/model_initialization_registry.md).
2. Run development evaluations on the registered historical pool and preserve all scheduled
   task/method/training-seed/checkpoint rows. Freeze the checkpoint-selection policy—with
   the full registry and exact per-task historical identities embedded—before opening those
   results.
3. Run `scripts/select_sota_checkpoint.py` on development-only evidence whose file manifest
   exactly equals the registered `historical-eval` source. Its create-once,
   self-hashed `SELECTED` receipt binds the exact checkpoint artifact to its immutable
   training `run_contract.json`; test or confirmation evidence is forbidden.
4. In the fresh source snapshot, reuse that same pre-results registry, generate a new
   unlabeled IID version, attach only approved evidence-bearing
   references, and run `scripts/freeze_benchmark.py --forbidden-source-registry <receipt>`.
   The publication manifest embeds and exactly cross-binds that pre-generation registry;
   partial non-empty decoy trees fail. Publish the resulting schema-v3 manifest hash before
   inference. The existing v1 unlabeled draft and smoke receipts are non-publishable and
   must not be upgraded in place.
5. Evaluate each selected checkpoint once on every frozen row, then aggregate without
   retuning. A GRPO confirmation invocation must include all four fail-closed switches below:

The publication training launcher requires `SOTA_FORBIDDEN_SOURCE_REGISTRY`,
`SOTA_MODEL_INITIALIZATION_REGISTRY`, and the explicit approved `SOTA_BASE_MODEL_PATH`.
Every immutable GRPO run contract verifies and records the exact registered
`sft-train` and `rl-train` source identities; the initial SFT receipt's recorded
data root must resolve to that same SFT tree. A clean registry alongside an
unregistered training copy is therefore rejected.

```bash
# TEMPLATE ONLY: TSP specialist shown; required assets do not exist while BLOCKED.
clasp-eval \
  --evaluation_purpose confirmation \
  --benchmark_manifest /path/to/new/preinference_iid_manifest_v3.json \
  --campaign_manifest /path/to/frozen_campaign_manifest.json \
  --checkpoint_selection_receipt /path/to/development/checkpoint_selection_receipt.json \
  --checkpoint_path /path/to/exact/selected/checkpoint-N \
  --stage grpo \
  --data_root /path/to/new/assembled_iid \
  --tasks tsp \
  --split eval \
  --eval_start_index 0 \
  --max_eval_samples_per_task 0 \
  --no-prefer_subset_files \
  --seed 271828 --dtype float16 \
  --num_return_sequences 8 --generation_batch_size 8 \
  --do_sample --temperature 0.7 --top_p 0.9 --top_k 50 \
  --repetition_penalty 1.0 \
  --max_new_tokens 3000 --max_prompt_length 29768 \
  --force_no_reason_prompt \
  --save_results --save_details --resume \
  --inference_root /path/to/create-once/confirmation-output
```

Confirmation uses resume only for an identical fingerprint; point it at a new, preserved
output root and never use `--no-resume` to replace a confirmation attempt. Confirmation
forbids row caps, offsets, size filtering, subset-file preference, mutable best-run updates,
and automatic checkpoint choice. The evaluator verifies the benchmark
manifest self-hash, the selection receipt, selected checkpoint step/artifact hash, and
training-contract hash before generation.

Keep the system lanes separate:

- **raw K=8** is the primary lane: eight unmodified completions, strict feasible-first
  objective selection, no repair, reranking, or regeneration;
- **deterministic repair K=8** is a distinct hybrid produced from a completed raw artifact
  by `clasp-repair`; report both raw and repaired metrics plus repair time;
- **adaptive repair** is another distinct compute-adaptive system. Report its stopping rule,
  actual candidate count, generated tokens, GPU time, and wall time. Never place its values
  in either K=8 column.

Every saved raw record retains the raw representation `instance_sha256` and separate
`semantic_instance_sha256`, `benchmark_row_sha256`, and `reference_output_sha256` fields.
The run config/summary bind the experiment and protocol fingerprints, model and data
manifests, strict-verifier identity, benchmark manifest, selection receipt, source-row
locator, and evaluation implementation; `run_details__*.jsonl` preserves every candidate
and denominator. Completed run configs and summaries are self-hashed, and the config binds
the final details artifact hash, size, and record count. Rescore and repair records and
summaries add their own self-hashes, implementation fingerprint, and verbatim source
linkage. Do not publish from CSV or a summary without the matching completed config and
details artifact.

Report every prespecified training seed separately, then mean and standard deviation across
the three seeds. Local uncertainty uses a hierarchical bootstrap that resamples training
seeds first and paired benchmark instances second; decoding seeds are sensitivity runs, not
additional training replicates. Include failed runs and missing seeds in the ledger, keep
conditional-gap denominators explicit, and state task-by-task results when the full
five-task claim criterion is not met. See
[`experiments/sota_20260814/PROTOCOL.md`](experiments/sota_20260814/PROTOCOL.md),
[`docs/sota_advancement_and_selection.md`](docs/sota_advancement_and_selection.md), and
[`docs/independent_iid_benchmark.md`](docs/independent_iid_benchmark.md).
The IID workflow uses `scripts/run_reference_evidence.py solve-task` followed by
`export-references`; both phases fail closed on unresolved contracts, solver-artifact or
verifier drift, altered native logs, non-certifying OP/MIS/JSSP statuses, and any routing
backend that does not use CLASP's raw Euclidean objective. OP now has a strictly verified
raw-Euclidean bounded-MIP backend, but production remains blocked until its fixed-work policy,
audited wheel identity, and full 1,000-row evidence distribution are frozen and reviewed.

To compare CLASP and external systems under exactly the same strict verifier, export raw
candidates using the `run_details` JSONL schema and run:

```bash
clasp-rescore \
  --details inference/external/run_details.jsonl \
  --system_name external-baseline \
  --output inference/external/strict_rescore.jsonl
```

Each input record needs `task`, `instruction`, `input_problem`, `ground_truth_output`, and a
non-empty `candidates` list containing completion strings or objects with `completion`.
`instance_json` is strongly recommended and is saved automatically by new `clasp-eval` runs.

## Tests

```bash
pytest                 # full CPU correctness suite, no model loading
ruff check .           # lint
```

`tests/` covers reward parsing/scoring, `pd_component_profile` behavior,
canonical subset enumeration and credit, failure@`K` control, trainer-level
correctness, and legacy frontier-CDF helpers. These are the primary correctness
gate for any change to reward or advantage code.

## Checkpoints

`checkpoints/<variant>/` holds the latest checkpoint per variant, with a `MANIFEST.md`
recording provenance (source path, run name, step, and validation status). Weight files
are git-ignored; the manifests are not.

## Notes on invariants

- **Feasibility gating.** Canonical mode gives objective-quality credit only to
  strictly feasible candidates and takes the maximum over each size-`K` set;
  failed-set recovery uses violation only. The historical noncanonical preset
  instead selects a feasible-first lexicographic profile,
  `co_feasibility_lock_all`, and `cost_limit=0`. Changing either gate can
  silently break evaluation alignment; tests guard both paths.
- **`CLASP_*` environment variables.** `config.py::_apply_clasp_strict_defaults` translates
  high-level flags into concrete `pd_*` settings *and* exported `CLASP_*` env vars. A lot of
  runtime behavior is driven by those env vars rather than by passed arguments, so when
  tracing behavior, grep both the flag and its `CLASP_*` twin.
- **Opt-in cost components** are gated behind `CLASP_*` env vars and specific
  `pd_component_profile` values. Several are MIS-only; tests enforce that they do not leak
  into other tasks.
- Large data, model weights, checkpoints, W&B runs, and inference outputs are never
  committed. They live under `data/`, `models/`, or an external root passed via CLI/env.

## Verification

### Historical static/reorganization receipt

The table below predates the current campaign and is retained as provenance,
not as the status of the present working tree.  The 2026-08-15 pre-edit full
suite baseline was 765 passed / 42 failed; the failures were concentrated in
resume fidelity and frozen reference-evidence/source-hash contracts. On the
current snapshot, 41 canonical estimator/synthetic tests and 47 trainer
correctness tests pass; this focused receipt does not erase the unrelated
full-suite debt.

| Check | Result |
| --- | --- |
| `pytest` (bare, fresh clone) | 175 passed |
| `ruff check .` | clean |
| `pip install -e .` | 6 console scripts, imports from any cwd |
| Module import smoke, both clusters | 16/16 modules |

The 175 tests pass identically in the original research tree and here, which is the evidence
that the reorganization preserved semantics.

### End-to-end training, Jean Zay (1 node x 4 H100, `qos_gpu_h100-dev`)

Launched with `scripts/train.slurm` against real data (`data/data_rl`, five tasks) from the
real initialization (`sft/checkpoint-18400`): 40 steps, `num_generations=8`, resolved
`generation_batch_size=32`, bf16. All jobs reached `COMPLETED` and wrote checkpoints at
steps 20 and 40.

| Variant | Job | LoRA | Elapsed | Trend over 40 steps |
| --- | --- | --- | --- | --- |
| SFT | 456414 | 32/32 | 10:19 | loss 0.770 -> 0.650, grad-norm 0.305 -> 0.231 |
| Standard GRPO | 456413 | 32/32 | 13:44 | reward 2.752 -> 3.057 |
| Primal-dual | 456412 | 32/32 | 13:37 | candidate feasibility 0.531 -> 0.750, raw gap 0.157 -> 0.052 |
| Historical CLASP | 456070 | 32/32 | 16:27 | feasibility 0.531 -> 0.719, objective quality 0.846 -> 1.132, raw gap 2.339 -> 1.707, dual λ 0.008 -> 0.118 |
| Historical CLASP | 457464 | **64/64** | 16:05 | feasibility 0.531 -> 0.656, objective quality 0.846 -> 1.043, dual λ 0.008 -> 0.124 |

The first four ran before `LORA_R`/`LORA_ALPHA` defaults were added to the launcher and so
used the CLI default of 32/32. Job 457464 re-ran the historical CLASP preset
with the launcher exactly as shipped, confirming the 64/64 default that matches
the bundled legacy checkpoints. Neither job uses canonical `B>K` credit.

Two independent correctness signals:

- `[RewardCheck] Ground-truth verification passed for all checked samples. Checked=25` —
  the reward heads score the datasets' own reference solutions as feasible and optimal.
- Each variant emitted exactly the telemetry its configuration implies: standard GRPO logged
  no `constrained_grpo/*` metrics at all, primal-dual logged constraint metrics but no
  `clasp/*` ones, and only the historical CLASP arm logged the
  Pareto/dual/adaptive-length series.

**Do not rank the variants by these numbers.** Forty steps on independently sampled batches
is a smoke test, not an experiment, and the per-step series are noisy (the two CLASP runs
disagree on the raw-gap trend). Raw reward is especially misleading across arms, since
unconstrained GRPO does not pay for feasibility. Use `clasp-eval` for the metrics the method
actually targets.

### Evaluation, Jean Zay (1 H100)

`scripts/evaluate.slurm` on the bundled CLASP checkpoint at its Jean Zay source, raw
best-of-2 over 10 instances per task, no repair or reranking:

| Task | Candidate feasibility | Best-of-2 instance feasibility | Mean best gap |
| --- | ---: | ---: | ---: |
| `tsp` | 80.0% | 100.0% | 3.42% |
| `mis` | 65.0% | 70.0% | 1.47% |

Per-task CSV, a JSON run summary, and a streaming JSONL detail file were all written. This
is a smoke-scale evaluation (10 instances, N=2) that confirms the aggregation path produces
real feasibility and gap numbers; it is not a benchmark result. The reported protocol uses
300 instances per task at N=8, which is `scripts/evaluate.slurm`'s default.

### Single-GPU

The full matrix — SFT, all three RL variants, `clasp-pipeline` (SFT -> merge -> GRPO), and
`clasp-eval` — also runs on a single 48 GB GPU, which is how the three defects below were
found.

### Dependency stacks exercised

| Host | torch | transformers | trl | peft |
| --- | --- | --- | --- | --- |
| Jean Zay (`new_unsloth_env`) | 2.9.1 | 4.57.3 | 0.24.0 | 0.18.0 |
| LAMSADE (`unsloth_env`) | 2.5.1 | 4.57.1 | 0.24.0 | 0.18.1 |

`trl >= 0.24` matters: two of the three fixes below exist because of constraints that
version introduced or enforces.

### Evaluating a checkpoint elsewhere

`clasp-eval` resolves a PEFT checkpoint's base model **only** from
`adapter_config.json::base_model_name_or_path`; `--model_path` is an alias of
`--checkpoint_path`, not an override, and passing both with different values is rejected.
The bundled checkpoints record a Jean Zay path, so off-cluster you must edit that field or
symlink the path. See `checkpoints/README.md`.

## Provenance

This repository is the minimal, production-grade extraction of the `clasp` research tree.
Only the transitive import closure of the six real entry points was ported — experiment
scripts, rebuttal artifacts, plotting utilities, and dead modules were dropped.

Reward, advantage, and trainer internals were ported **byte-faithful**: module layout,
imports, and packaging changed, but function bodies did not. This is deliberate — the
reward/advantage math is scientifically sensitive and rewriting it for style would risk
silently changing results. The 175 tests pass identically before and after the port.

Module mapping from the research tree:

| Original | Here |
| --- | --- |
| `clasp/common.py` | `src/clasp/runtime.py` |
| `clasp/cli.py` | `src/clasp/config.py` |
| `clasp/data.py` | `src/clasp/data/loaders.py` |
| `clasp/prompting.py` | `src/clasp/data/prompts.py` |
| `clasp/modeling.py` | `src/clasp/models/loading.py` |
| `clasp/rewards.py` | `src/clasp/rewards/heads.py` |
| `clasp/task_rewards.py` | `src/clasp/rewards/registry.py` |
| `clasp/reward_utils/hybrid_co_rewards.py` | `src/clasp/rewards/co_tasks.py` |
| `clasp/reward_utils/clasp.py` | `src/clasp/rewards/clasp_math.py` |
| `clasp/reward_utils/primal_dual_grpo.py` | `src/clasp/rewards/primal_dual.py` |
| `clasp/reward_utils/primal_dual_grpo_trainer.py` | `src/clasp/trainers/primal_dual_grpo.py` |
| `clasp/run_sft.py` | `src/clasp/entrypoints/train_sft.py` |
| `clasp/run_grpo.py` | `src/clasp/entrypoints/train_grpo.py` |
| `clasp/run_grpo_normal.py` | `src/clasp/entrypoints/train_grpo_standard.py` |
| `clasp/run_pipeline.py` | `src/clasp/entrypoints/train_pipeline.py` |
| `clasp/run_eval.py` | `src/clasp/entrypoints/evaluate.py` |
| `clasp/audit_data.py` | `src/clasp/entrypoints/audit_data.py` |

Dropped as unreachable from every entry point: `comparative_inference.py`,
`plot_inference_comparison.py`, `analyze_tokenization_lengths.py`, `clean_wandb.py`,
`del_wandb.py`, `unified_runner.py`, `utils.py`, the off-path root-level
`primal_dual_grpo_trainer.py` copy, and the superseded per-task reward modules
`binpack_reward.py`, `jssp_reward.py`, `vrp_tsp_reward.py`.

Three real defects were fixed during the port. All three reproduce in the research tree,
and each one blocked a documented entry point:

1. **GRPO could not start.** `train_grpo.py` passed both `generation_batch_size` and
   `steps_per_generation` to `GRPOConfig`, which TRL ≥ 0.24 rejects as mutually exclusive,
   so every GRPO run that did not set one of them explicitly aborted. Only
   `generation_batch_size` is forwarded now; TRL recomputes `steps_per_generation` from it
   with the identical formula, so the effective schedule is unchanged.

2. **`clasp-pipeline` could not start.** `_make_sft_args` / `_make_grpo_args` built a fresh
   `Namespace` by hand, omitting 23 attributes that `run_sft` and `run_grpo` read and that
   the pipeline parser does not expose (`deduplicate_train_rows`, `optim`, `weight_decay`,
   `lr_scheduler_type`, `grpo_beta`, the `generation_*` knobs, …). The pipeline died with
   `AttributeError` before finishing its first SFT step. Both builders now seed their
   namespace from the corresponding stage parser's own defaults and let the explicit
   pipeline values override them, so the list cannot rot as stage flags are added.

3. **Unhelpful eval-batch failure.** TRL's eval-batch divisibility rule surfaced deep
   inside `GRPOConfig` without naming the flag to change. `run_grpo` now checks it directly,
   gated on `eval_strategy != "no"` — TRL only enforces it when evaluation is enabled, so
   checking unconditionally at CLI-parse time would have rejected valid runs that have no
   eval split.
