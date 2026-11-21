# Legacy saved-policy conversion (development only)

## Evidence boundary

`scripts/convert_legacy_saved_policy.py` re-encodes the policy that can actually
be loaded from a legacy `bias=lora_only` adapter. It does **not** reconstruct
the terminal in-memory SFT policy.

The affected PEFT serializer saved the LoRA A/B tensors but omitted the trained
Q/K/V biases. Loading that artifact therefore deploys its LoRA tensors on the
unchanged Q/K/V biases from the designated base model. The converter loads
exactly that deployable policy and saves it as `bias=all`, including those
unchanged base biases. This makes the same saved policy stable under another
save/reload cycle, but the historically trained bias values remain lost.

Every receipt permanently records:

- `recovers_historical_terminal_biases=false`;
- `converted_saved_policy_fidelity=true`;
- `terminal_sft_policy_fidelity=false`;
- `publication_eligible=false`, `sota_eligible=false`, and the combined
  `publication_sota_eligible=false` gate.

The converted adapter is suitable only for controlled development comparisons
against the legacy saved policy. It must not initialize a publication
trajectory or support a SOTA claim.

## Fail-closed conversion proof

The converter requires all of the following before it creates evidence:

1. The legacy adapter config must be local LoRA with `bias=lora_only`, include
   Q/K/V targets, and designate the supplied base by the same absolute,
   non-symlink path.
2. The legacy safetensors artifact must contain a complete paired LoRA A/B
   inventory and zero bias tensors. Unexpected modules-to-save and DoRA state
   are outside this converter's scope.
3. The supplied Qwen base may have biases only in targeted Q/K/V projections;
   otherwise `bias=all` is not scope-equivalent and conversion fails.
4. The loaded legacy policy's Q/K/V biases must be bitwise equal to the frozen
   base biases, and its LoRA tensors must be bitwise equal to the source
   safetensors payload.
5. Saving with `bias=all` must produce exactly the same canonical LoRA plus
   Q/K/V-bias state. A fresh base and the converted adapter are then reloaded.
   Every canonical tensor must agree bit-for-bit.
6. On a frozen six-token prompt, the legacy and converted reloads must produce
   bitwise-identical logits and continuation-token log-probabilities using
   eager attention and evaluation mode.
7. The base and legacy source trees are hashed before and after conversion and
   must not change. The output config, weights, full tensor inventory, and all
   other output files are bound into a self-hashed, read-only receipt.

The converter uses `clasp.models.checkpoint_fidelity` for canonical PEFT keys
and dependency-light inspection of the converted config/safetensors bytes. It
intentionally does not create `checkpoint_fidelity.json`: that protocol proves
closure with an in-memory *trained* state, which cannot be established for the
lost legacy terminal biases.

## Create-once use

A full Qwen2.5-7B development conversion was completed after the paired
evaluation receipt was sealed.  It used the fresh output path below and FP32,
matching the serialized legacy LoRA tensor dtype:

```bash
PYTHONPATH=src ../my_idea_grpo/.venv/bin/python \
  scripts/convert_legacy_saved_policy.py \
  --base-model-path /home/lamsade/habgaryan/phd_project/my_idea_grpo/seacove_models/base/Qwen2.5-7B-Instruct \
  --legacy-adapter-path models/local_tsp_sft500k_seed42_instruct \
  --output-dir experiments/sota_20260814/local_development/tsp_sft_saved_policy_bias_all_v1 \
  --device cuda:0 \
  --dtype float32
```

The verified artifact contains 392 LoRA tensors and 84 frozen-base Q/K/V bias
tensors (161,609,728 parameters total).  Its receipt hash is
`c2eb6b8714d1db4c2872f6003b1a4f73adf9d73e824109665cd7f9376cc5edb0`;
the receipt binds `recovers_historical_terminal_biases=false`,
`publication_eligible=false`, and `sota_eligible=false`.

The output path must not already exist. The tool never updates, merges into, or
reuses an existing output directory. It also verifies that the legacy source
tree is byte-identical after the run. On success, all output files are
read-only and the receipt is
`legacy_saved_policy_conversion_receipt.json`.

An existing conversion can be checked without loading the model:

```bash
PYTHONPATH=src ../my_idea_grpo/.venv/bin/python \
  scripts/convert_legacy_saved_policy.py \
  --verify-only \
  --output-dir experiments/sota_20260814/local_development/tsp_sft_saved_policy_bias_all_v1
```

Verification recomputes the self-hash, claim flags, output tensor inventory,
checkpoint-fidelity inspection, and the complete base/source/output content
manifests. It rejects a writable receipt, a changed base or source, a missing
converted bias, or any other byte-level artifact drift.

## Test coverage

`tests/test_convert_legacy_saved_policy.py` constructs a two-layer tiny Qwen
model, deliberately changes the trainable Q/K/V biases before triggering the
affected `lora_only` save behavior, and confirms that conversion preserves the
resulting deployable saved policy rather than claiming to preserve those lost
terminal values. It also rejects:

- an existing output directory without touching its sentinel content;
- a base path different from the one recorded by the legacy adapter;
- a converted safetensors artifact with one required bias removed; and
- a receipt whose publication flag is changed without recomputing its hash.
