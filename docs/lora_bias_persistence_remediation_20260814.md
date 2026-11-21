# LoRA bias persistence remediation (2026-08-14)

## Publication decision

The queued base-SFT v1 campaign and all earlier CLASP adapters configured with
`bias=lora_only` are not valid representations of their terminal training
policies.  They may be retained as explicitly labelled development evidence,
but they must not initialize a publication RL run or support a SOTA claim.  A
fresh source snapshot, base-SFT campaign v7, output root, submission ledger, and
model-initialization registry are required.

The future publication policy is `CLASP_LORA_BIAS=all`, guarded by
`scripts/audit_lora_persistence.py`.  This is not a capacity change for the
frozen Qwen2.5-7B architecture: its weight index contains exactly 84 bias
tensors, all in the 28 layers' targeted `q_proj`, `k_proj`, and `v_proj`
modules.  Thus `all` and the intended `lora_only` trainable scope are identical
for this architecture.

## Root cause and reproduced impact

PEFT 0.18.0 (the Jean Zay environment) and PEFT 0.19.1 (the local environment)
both make the targeted Q/K/V biases trainable under `bias=lora_only`, but their
adapter exporter omits those biases.  A tiny-Qwen round trip reproduced a
nonzero reload error under `lora_only` and exact reload under `all`.  The local
40-step CLASP checkpoint independently contained optimizer state for 84 bias
tensors but its saved adapter contained zero bias tensors.

For Qwen2.5-7B with Jiang-style rank-64 attention-and-MLP LoRA, the exact
trainable and serialized inventory is:

| Component | Tensors | Parameters |
| --- | ---: | ---: |
| LoRA A/B, 7 modules × 28 layers | 392 | 161,480,704 |
| Q/K/V biases, 3 modules × 28 layers | 84 | 129,024 |
| Total | 476 | 161,609,728 |

Changing only `adapter_config.json` cannot repair an affected checkpoint.  The
trained bias values are absent, and Adam moments do not reconstruct parameter
values.  The development-only converter in
`scripts/convert_legacy_saved_policy.py` can faithfully re-encode the *saved*
policy with frozen base Q/K/V biases under `bias=all`; it explicitly cannot
recover the historical terminal biases or create publication evidence.
Evaluating either the old adapter or that re-encoding measures the saved policy,
not the terminal in-memory training policy.

## Enforced gates

`scripts/audit_lora_persistence.py` is dependency-free and performs these
checks before a model can enter the publication chain:

1. The base must have the exact frozen Qwen2.5-7B dimensions and a safetensors
   weight index whose complete bias set equals the 84 expected targeted Q/K/V
   keys.  Any additional base bias makes `bias=all` unsafe and fails closed.
2. The adapter config must use `bias=all`, rank/alpha 64, dropout 0, and exactly
   the seven Jiang target modules.
3. `adapter_model.safetensors` must contain exactly 84 expected bias tensors and
   392 expected LoRA tensors with exact names and shapes.  Its payload layout
   must be contiguous, non-overlapping, and untruncated.
4. Every serialized bias value receives a SHA-256 hash.  The final adapter audit
   receipt is create-once, read-only, and must reopen identically.
5. Every optimizer-resumable base-SFT checkpoint and the finalized root adapter
   are audited.  A single malformed or bias-missing `checkpoint-*` directory
   prevents fallback to another checkpoint.
6. The finalized root must carry `sft_reload_equivalence.json`, produced by a
   separate post-training process. Its fresh PEFT trainable closure must equal
   the pre-save checkpoint-fidelity closure exactly, so a serializer or loader
   that drops any trained Q/K/V bias fails even if the config still says
   `bias=all`.
7. Two independent fresh loads must also agree bitwise across the complete
   runtime state and fixed bfloat16 CUDA probe logits, log-probabilities, and
   argmax decisions. The five-task initialization registry binds this receipt
   independently of the persistence and checkpoint-fidelity identities.

The v7 base-SFT campaign contract cryptographically binds this persistence
policy.  The model-initialization freezer and RL run-contract builder repeat the
base-scope and adapter-inventory audits, and the future RL worker also uses
`CLASP_LORA_BIAS=all` so continued saves preserve the same biases.

## Mapped legacy `lora_only` surfaces

The following are intentionally preserved as historical/development paths and
must not be confused with the v7 publication chain:

- `scripts/run_local_tsp_k8_development.py` and its sealed evaluator contract
  validate the existing local development adapter.  The completed receipt
  remains immutable development-only evidence.
- `scripts/train_sft500k_specialist_a100.slurm` and
  `scripts/validate_sft500k_specialist.py` are the historical
  Qwen2.5-7B-Instruct lane.
- `scripts/continue_sft100k_a100.slurm` and
  `scripts/validate_sft100k_a100_continuation.py` resume a historical unified
  trajectory.  Switching its policy midway would not recover missing learned
  biases or produce a matched trajectory.
- Historical adapter configs, old run contracts, frozen remote source v1, and
  already-created campaign/ledger artifacts remain immutable evidence of what
  was actually run.  They are not rewritten or silently relabelled.
- `tests/test_trainer_correctness.py` uses a synthetic `lora_only` object to
  exercise reference-anchor semantics; it is not a publication launch policy.

## Implemented source-level closure

Training now has mutually exclusive publication/development fidelity modes.
Before optimization, every rank proves exact closure between trainable model
state and PEFT-exported state, checks the Qwen2.5-7B inventory, and agrees on a
common digest.  Every retained checkpoint and final root receives a create-once
`checkpoint_fidelity.json`; strict resume rejects the newest malformed or
unreceipted checkpoint instead of falling back. Publication beta-positive GRPO
does not merge: it independently loads the exact same base-plus-adapter state,
freezes the explicit reference, and writes `exact_peft_kl_reference.json` only
after full state/storage, tokenizer, logits/log-probabilities, prepared-runtime,
and distributed-consensus verification. The older merged fixed-KL v2 path is
development-only and cannot authorize a publication trajectory.

These hooks prevent recurrence; they do not repair historical values.  Until
the fresh bias-safe v7 SFT campaign completes, the correct claim status remains
**no publication checkpoint and no SOTA claim**.
