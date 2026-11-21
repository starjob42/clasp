# SOTA campaign advancement and checkpoint selection

These tools separate three decisions that must not be conflated:

1. `audit_sota_advancement.py` decides whether a completed development stage is healthy enough to advance.
2. `select_sota_checkpoint.py` selects a checkpoint using only a frozen development rule.
3. Neither tool authorizes a SOTA/publication claim. Confirmation remains a later, one-shot evaluation on a hash-frozen unseen benchmark.

Both commands are scheduler-free and read-only with respect to run/model directories. Their only write is an atomic, self-hashed JSON receipt at the requested output path. Receipts are create-once, group-readable/non-writable (`0440`); an existing path is never replaced. A failed or incomplete audit returns exit code 2 and still writes a `HOLD` receipt when the output path is new.

The campaign is currently **BLOCKED before confirmation**: the IID reference-solver contract remains unresolved, so no production reference set, schema-v3 frozen manifest, or SOTA evidence exists. The immutable v1 training-smoke and v2 evaluation-smoke source snapshots remain historical artifacts; v2 is stale relative to the current selection and identity gates. Do not patch either snapshot. A new source snapshot is required before producing confirmation assets.

## Advancement audit

The checked 40-step policies are immutable historical smoke evidence and are not launch
authorizations for the current pipeline: they predate the mandatory model-initialization
registry and full prior-stage receipt chain. Do not edit or re-hash them. Freeze a new
policy only after the real registry exists. It must declare a nonnegative `stage_index` and
an exact chronological `prior_stage_receipt_chain`; its length must equal the index.

Run the audit only after all eight jobs have reached a terminal state:

```bash
python scripts/audit_sota_advancement.py \
  --policy /path/to/new/frozen_stage_policy.json \
  --run-root /path/to/create-once/campaign/root \
  --prior-stage-receipt /path/to/stage-0-ADVANCE.json \
  --prior-stage-receipt /path/to/stage-1-ADVANCE.json \
  --output /path/outside/run/root/current_stage_advancement_receipt.json
```

Omit `--prior-stage-receipt` only for `stage_index=0`. For later stages, repeat it
in exact chronological order for the complete chain. The launcher takes the same ordered
paths as a JSON array in `SOTA_PRIOR_STAGE_RECEIPTS_JSON`; a missing, writable, `HOLD`,
tampered, reordered, or pathname-swapped receipt blocks launch.

The auditor fails closed unless all of the following hold:

- every expected task/method/seed run exists, and no expected run is omitted from the denominator;
- every prior receipt is a stable, regular, non-writable snapshot with a valid file hash and self-hash, an `ADVANCE` decision, all scheduled runs `PASS`, and the exact chain prefix declared by the next policy;
- `run_contract.json` is a regular non-writable file with a valid internal model/base/data/source fingerprint receipt, plus exact `sft-train` and `rl-train` bindings to the pre-frozen forbidden-source registry;
- task, method, every seed scheduled by that policy (seed 42 for the 40-step smoke; three seeds for the long-horizon campaign), K, world size, horizon, logging/evaluation/save cadence, strict-resume setting, and method-specific CLASP/dual switches match the frozen policy;
- contracts within each task/seed group are byte-semantically equal after removing only the prespecified method-specific fields;
- the final model, completed `stage_status.json`, target checkpoint, root trainer state, and root log history agree;
- every retained checkpoint is optimizer-resumable: model, optimizer, scheduler, every rank RNG file, and the required primal-dual/BoN state are nonempty and structurally valid;
- every `beta>0` run has one live, byte-identical
  `exact_peft_kl_reference.json` at the root and every retained checkpoint, plus
  valid `exact_peft_kl_reference_step1.json` and
  `exact_peft_kl_reference_post_step.json` receipts at the root and every
  retained checkpoint. The complete lineage is bound to the contract's exact
  same-adapter source, beta, dtype, world size, fixed final horizon,
  tensor/function hashes, optimizer exclusion, policy-change evidence, and
  rank consensus. `beta=0` sets all three exact-KL identities to null and
  forbids all three sidecars at both locations;
- checkpoint history is a prefix of the completed root history, and training/evaluation log steps have no gap at the contracted cadence;
- every numeric contract/state/log value is finite;
- loss is finite, at least one gradient and method-appropriate advantage are nonzero, and method-appropriate objective credit is observed;
- no group auxiliary is reported skipped, no objective-quality fallback is used, and no mixed-budget/global-max publication ablation appears;
- CLASP recovery/controller health flags are present and active on every training log row.

The resume result is deliberately named **strict resume-readiness**. It proves that every retained checkpoint contains the complete state the launcher requires and that terminal histories are continuous. Byte-for-byte equivalence between separate uninterrupted and interrupted/resumed GPU executions is a different, stronger experiment; this receipt does not mislabel resume-readiness as that experiment.

For a 400-step policy, `development_performance_gate` may additionally name a baseline, challenger, required task set, maximum 0.02 absolute pass@8 regression, and minimum improved-task count. The supplied aggregate must be self-hashed schema-v3 output from `aggregate_sota.py`, be labeled `development_only`, retain every scheduled seed, and use K=8. A task improves only if every scheduled seed improves pass@8 or conditional gap. Confirmation evidence is never an input to this gate.

## Frozen development checkpoint selection

The fixed selection rule is:

1. maximize development pass@K;
2. retain every checkpoint within 0.01 absolute of the best pass@K;
3. minimize conditional strict-verifier gap to the frozen development reference
   within that band ("optimality gap" only for certified references);
4. break exact gap ties by choosing the earlier checkpoint step.

The policy must be frozen before checkpoint development results are opened. It declares the complete Cartesian product of tasks, methods, training seeds, and checkpoint steps. `checkpoint_selection_policy.template.json` remains deliberately invalid until the strongest matched baseline and full checkpoint schedule are known after the pilot. At that point, replace every placeholder, set `status` to `frozen_before_development_results`, and compute `policy_sha256` from canonical JSON after removing that field.

Selection evidence is one JSON object with `schema_version: 2`, `status: "complete"`, `evidence_scope: "development_only"`, the campaign ID, K, a `records` list, and `evidence_sha256`. Every scheduled record contains:

```json
{
  "task": "tsp",
  "method": "clasp_dense",
  "train_seed": 42,
  "checkpoint_step": 400,
  "candidate_budget": 8,
  "instances": 200,
  "feasible_instances": 190,
  "pass_at_k": 0.95,
  "conditional_gap_n": 190,
  "conditional_gap_mean": 0.0123,
  "evaluation_fingerprint": "<64 lowercase hex>",
  "protocol_fingerprint": "<64 lowercase hex>",
  "checkpoint_artifact_sha256": "<64 lowercase hex>",
  "checkpoint_fidelity_identity_sha256": "<64 lowercase hex>",
  "checkpoint_fidelity_receipt_sha256": "<64 lowercase hex>",
  "checkpoint_fidelity_receipt_file_sha256": "<64 lowercase hex>",
  "grpo_beta": 0.01,
  "fixed_kl_reference": "<path-free exact-source identity; null iff beta=0>",
  "exact_peft_kl_runtime": "<path-free live runtime identity; null iff beta=0>",
  "exact_peft_kl_post_step": "<path-free root/selected post-step lineage; null iff beta=0>",
  "run_contract_sha256": "<64 lowercase hex>",
  "data_manifest_sha256": "<64 lowercase hex>",
  "strict_verifier_manifest_sha256": "<64 lowercase hex>"
}
```

The selector checks exact pass and gap denominators, the entire scheduled grid,
shared protocol/data/verifier/run-contract identity within a checkpoint group,
the same exact-source/runtime chronology across all checkpoints in that run,
and distinct evaluation/checkpoint/fidelity fingerprints. The evidence builder
reopens both root and checkpoint runtime receipts and requires byte identity.
It additionally requires every
method and training seed for a task to use the same development protocol, data,
strict verifier, candidate budget, and instance denominator. Undefined
conditional gap in the best-pass band causes `HOLD`; it is never silently
treated as zero or infinity.

```bash
python scripts/select_sota_checkpoint.py \
  --policy experiments/sota_20260814/checkpoint_selection_policy.json \
  --evidence results/development/checkpoint_evidence.json \
  --output results/development/checkpoint_selection_receipt.json
```

The receipt preserves every candidate's disposition and the exact evidence and policy hashes. The selector rejects any evidence labeled confirmation/test, never edits a “best checkpoint” pointer, and never copies or deletes model files. Each selected row binds `checkpoint_artifact_sha256` to the immutable training `run_contract_sha256`, task, method, training seed, and step. During confirmation, the evaluator independently resolves the checkpoint and training contract and rejects a mismatch. Confirmation evaluators must receive the selected artifact explicitly and must not perform best-checkpoint mutation or selection.

## Confirmation handoff

The handoff is one-way:

```text
development runs
  -> completed advancement receipt
  -> policy frozen before checkpoint results
  -> development-only selection evidence
  -> self-hashed SELECTED checkpoint receipt
  -> approved references + new IID version
  -> pre-inference schema-v3 benchmark manifest
  -> one-shot confirmation details
  -> publication aggregate
```

After the reference gate is resolved and a fresh immutable source snapshot is verified, a GRPO confirmation launch must name all identities explicitly:

```bash
# TEMPLATE ONLY: TSP specialist shown; current confirmation assets are unresolved.
clasp-eval \
  --evaluation_purpose confirmation \
  --benchmark_manifest /path/to/new/preinference_iid_manifest_v3.json \
  --campaign_manifest /path/to/frozen_campaign_manifest.json \
  --checkpoint_selection_receipt results/development/checkpoint_selection_receipt.json \
  --checkpoint_path /path/to/exact/selected/checkpoint-N \
  --stage grpo --data_root /path/to/new/assembled_iid \
  --tasks tsp --split eval \
  --eval_start_index 0 --max_eval_samples_per_task 0 \
  --no-prefer_subset_files \
  --seed 271828 --dtype float16 \
  --num_return_sequences 8 --generation_batch_size 8 \
  --do_sample --temperature 0.7 --top_p 0.9 --top_k 50 \
  --repetition_penalty 1.0 --max_new_tokens 3000 --max_prompt_length 29768 \
  --force_no_reason_prompt --save_results --save_details --resume \
  --inference_root /path/to/create-once/confirmation-output
```

Confirmation evaluates every frozen row and forbids subset preference, caps, offsets, size filtering, mutable best-run updates, post-result checkpoint choice, and `--no-resume` replacement of an earlier attempt. Resume is valid only for an identical fingerprint in the same preserved output root. Preserve the completed run config, summary, per-task CSV, and details JSONL together. The details keep every candidate plus raw `instance_sha256`, semantic instance, benchmark-row, reference-output, locator, verifier, protocol, and experiment identities.

Publication reporting is per task and per training seed first. Then report mean and standard deviation across the three prespecified training seeds and hierarchical intervals that resample training seeds before paired instances. Decoding seeds remain sensitivity runs. A failed or missing scheduled seed stays visible; it is not removed from the denominator.

## Computing policy/evidence self-hashes

The hash is SHA-256 over UTF-8 JSON serialized with sorted keys and separators `(',', ':')`, after removing the self-hash field (`policy_sha256` or `evidence_sha256`). The commands report the independently observed digest on any mismatch. Store the frozen policy and later selection/advancement receipts alongside the job ledger; do not place receipts inside immutable run directories.
