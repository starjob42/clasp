# Local TSP K=8 development launcher

`scripts/run_local_tsp_k8_development.py` runs one isolated, create-once CLASP-dense
development trajectory on physical GPU 3 (the local 48 GB RTX A6000). It does not call,
modify, or satisfy the remote/publication launch protocol.

## Evidence status

Every contract produced by this launcher is marked
`development_only_not_publication_eligible`. Results cannot support a publication or SOTA
claim because this lane:

- starts from Qwen2.5-7B-**Instruct**, not the approved non-instruction Qwen2.5-7B base;
- runs with local PEFT 0.19.1, outside the publication constraint `peft>=0.18,<0.19`;
- runs on one RTX A6000 with `generation_batch_size=8`, rather than the matched four-A100
  trajectory; and
- covers TSP only and never loads confirmation data.

The launcher uses only a copied RL **train** file. `data/data_rl/tsp/eval` must not exist.
Evaluation for training is deterministically carved from the copied train source with seed 42.

## Required local artifacts

The launcher has fixed paths and SHA-256 manifests; paths cannot be overridden.

```text
models/local_tsp_sft500k_seed42_instruct/
data/data_rl/tsp/train/train_rl.json
/home/lamsade/habgaryan/phd_project/my_idea_grpo/seacove_models/base/Qwen2.5-7B-Instruct/
/home/lamsade/habgaryan/phd_project/my_idea_grpo/.venv/bin/python
```

The SFT copy must be the completed 500,000-row, step-31,250, seed-42 TSP specialist.
Its adapter hash is
`9bf9ea50a5f7e9ff57399ea9bcca487785867d95bc9fdc11dadb7c723cef5173`.
The copied 3,200-row RL train file hash is
`90c5a48909ac9df191c727237474ec262db4a9c654f3555a19b744304dc0a288`.
The launcher verifies all SFT and base-model files, not only these headline hashes.

To stage the train-only copy from the audited local source:

```bash
mkdir -p data/data_rl/tsp/train
rsync -avP \
  /home/lamsade/habgaryan/phd_project/hybrid_gnn_llm/data_rl/tsp/train/train_rl.json \
  data/data_rl/tsp/train/train_rl.json
sha256sum data/data_rl/tsp/train/train_rl.json
```

Do not copy `tsp/eval` into this data root.

## Modes

| Mode | Source cap | Actual train/eval split | Steps | Output directory |
|---|---:|---:|---:|---|
| `memory_gate` | 160 | 152 / 8 | 1 | `outputs/tsp_k8_development_20260814/contracted_memory_gate_v1` |
| `dense40` | all 3,200 | 3,040 / 100 | 40 | `outputs/tsp_k8_development_20260814/contracted_clasp_dense40_seed42_v1` |

`max_eval_samples_per_task` is enforced after the deterministic train-derived split. The
`memory_gate` caps the source at 160 to produce exactly 152 train and 8 evaluation rows;
`dense40` uses 3,040 train rows and caps the 160-row split at 100 evaluation rows.

The successful earlier gate at `outputs/tsp_k8_development_20260814/k8_memory_gate` predates
this launcher. It remains valid development evidence, but this launcher deliberately does not
adopt, resume, mutate, or describe it as a contracted run.

## Verify and execute

Dry-run verification performs all artifact hashes (including the 15 GB base), checks the exact
audited dependency versions, resolves the training CLI, confirms the target output is absent,
and prints the fully expanded command. It does not require or touch a GPU:

```bash
../my_idea_grpo/.venv/bin/python \
  scripts/run_local_tsp_k8_development.py dense40
```

Execute only after reviewing that output:

```bash
../my_idea_grpo/.venv/bin/python \
  scripts/run_local_tsp_k8_development.py dense40 \
  --execute --acknowledge-development-only
```

For execution, the launcher additionally requires physical GPU index 3 to be an RTX A6000 with
at most 1,024 MiB already used. It sets `CUDA_VISIBLE_DEVICES=3`, records the observed device,
atomically creates `local_development_run_contract.json` with mode `0440`, and launches training.

## Create-once safety

An output path is permanently single-use. If it exists for any reason—even if it contains only
a partial contract or incomplete checkpoint—the launcher refuses to run. It also removes inherited
resume, distributed, W&B, `CLASP_*`, `PD_*`, `GENERATION_*`, and `GRPO_*` state before applying the
fixed environment. There is no resume or overwrite option. Preserve a failed run for audit and
define a reviewed new launcher contract/path instead of deleting or reusing it.

After the contracted dense40 trajectory is complete, use the separate create-once procedure in
[`local_tsp_dense40_posttrain_evaluation.md`](local_tsp_dense40_posttrain_evaluation.md). That lane
validates the full checkpoint state and contract hashes before it stages historical evaluation data;
it never writes to or resumes this training directory.
