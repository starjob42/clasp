# Frozen model-initialization registry

Publication training must not choose a base model or specialist SFT start after seeing RL
results. Before the advancement policy is frozen and before any publication RL job starts,
create one five-task registry with the exact non-instruction Qwen2.5-7B bytes, every matched
500k specialist SFT artifact, seed/settings receipt, and registered SFT-data lineage:

```bash
python scripts/freeze_model_initialization_registry.py \
  --base-model-path /immutable/models/Qwen2.5-7B \
  --forbidden-source-registry /immutable/contracts/forbidden_source_registry.json \
  --base-sft-campaign /immutable/contracts/base_sft_campaign.json \
  --specialist tsp,/immutable/base_sft/runs/tsp/seed42,/immutable/data_sft/tsp/train,42 \
  --specialist cvrp,/immutable/base_sft/runs/cvrp/seed42,/immutable/data_sft/cvrp/train,42 \
  --specialist op,/immutable/base_sft/runs/op/seed42,/immutable/data_sft/op/train,42 \
  --specialist mis,/immutable/base_sft/runs/mis/seed42,/immutable/data_sft/mis/train,42 \
  --specialist jssp,/immutable/base_sft/runs/jssp/seed42,/immutable/data_sft/jssp/train,42 \
  --output /immutable/contracts/model_initialization_registry.json
```

The destination is create-once and read-only. The freezer rejects an instruction/chat base,
the wrong Qwen architecture/size, a missing task, a non-seed-42 or non-500k SFT run, an
incomplete step-31250 finalization receipt, a mismatched adapter base, changed model bytes,
and SFT data not present in the frozen forbidden-source registry. It also reopens the
all-five-task base-SFT campaign, rehashes its immutable source/base/data inputs, requires
every specialist path to be that campaign's exact task cell, and requires every cell's
create-once run contract and optimizer-resumable final checkpoint. The run contract is part
of the specialist model-tree fingerprint, so the initialization registry cryptographically
binds the pre-training campaign rather than merely trusting a directory name.
It additionally proves that `bias=all` is scope-equivalent on the exact base
weight index and rejects any specialist whose safetensors inventory lacks one
of the 84 trained Q/K/V biases or 392 Jiang-LoRA tensors.  Legacy
`bias=lora_only` checkpoints are not eligible even if their other lineage
fields match.

Copy `registry_sha256` into the new advancement policy as
`model_initialization_registry_sha256` before launching jobs. Each publication launch must
set both:

```bash
export SOTA_MODEL_INITIALIZATION_REGISTRY=/immutable/contracts/model_initialization_registry.json
export SOTA_BASE_MODEL_PATH=/immutable/models/Qwen2.5-7B
```

`prepare_sota_run.py` rehashes the actual base and task specialist, checks the full registry,
and embeds its portable compact identity in `run_contract.json` under
`model_initialization_registry`. The compact identity preserves the registry self-hash,
base-tree hash, all five specialist-tree hashes, matched-SFT contract hash, seed, SFT source
identity, and forbidden-registry/tree hashes. The advancement auditor and confirmation
campaign must require exact equality to that identity.

No production registry is checked in. The historical v1/v2 snapshots and smoke advancement
policies predate this gate and remain stale; they cannot authorize a confirmation claim.
