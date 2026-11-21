# Fail-closed SOTA campaign aggregation

`scripts/aggregate_sota.py` is the publication boundary. It reads completed artifacts and
writes one atomic aggregate receipt; it never submits, cancels, repairs, rescales, or mutates
an evaluation job.

## Frozen inputs

Confirmation requires four independent, hash-bound contracts:

1. the schema-v3 independent benchmark manifest (`--benchmark-manifest`);
2. the frozen lane policy (`--lane-contracts`, defaulting to
   `experiments/sota_20260814/lane_contracts.json`); and
3. a campaign manifest (`--campaign-manifest`) frozen before confirmation inference; and
4. the reviewed external-context registry (`--targets`).

`targets.json` is an external-context registry rather than a leaderboard threshold. Its LLMCoSolver values apply only to the historical first 100
released rows under the author protocol. Its FALCON values are context from an unreproduced
preprint system with BOPO, grammar-constrained generation, and task-specific repair. Every
external row is fail-closed as comparison- and strict-beat-ineligible, so aggregation cannot
turn any of these point estimates into a SOTA flag or target-gap outcome.

The new IID benchmark has a separate reference-solver contract. TSP/CVRP remain
gap-to-heuristic; OP/MIS/JSSP may use optimality-gap wording only after every retained row
passes the task's native certificate rule. The external OP row is cross-protocol invalid and
does not prevent a future raw-Euclidean new-IID OP reference from being certified exact.

Start from `experiments/sota_20260814/campaign_manifest.template.json`. Populate `runs` with
the exact Cartesian product of methods, training seeds, and tasks. Every row is derived from
immutable artifacts and contains:

- `method`, `training_seed`, and `task`;
- `training_run_contract_sha256`;
- `checkpoint_step` and `checkpoint_artifact_sha256`; and
- all checkpoint-fidelity identity/receipt/file hashes, the exact GRPO beta,
  the path-free prelaunch `fixed_kl_reference`, post-Trainer step-zero
  `exact_peft_kl_runtime`, and post-optimization reference-immutability
  `exact_peft_kl_post_step` lineage (all three null iff beta is zero); and
- `checkpoint_selection_receipt_sha256`; and
- deterministic `confirmation_attempt_id` over the exact campaign/grid-cell hashes.

The manifest also freezes the ordered `baseline` and `challenger` roles and requires
`methods` to be exactly that two-method pair. CLI roles must match; they cannot be selected or
reversed after confirmation outcomes are observed.

It additionally freezes one safe canonical shared `confirmation_attempt_ledger` root.
Each evaluator cell creates and locks one content-addressed claim there and writes the same
read-only marker beside its outputs. Aggregation reopens both objects, verifies their
self-hashes, campaign/experiment/attempt identities, and refuses copied manifests, swapped
cells, missing markers, or writable claims.

Set `status` to `frozen_before_confirmation`, record the canonical `frozen_at_utc`, remove draft placeholders, and compute
`campaign_manifest_sha256` over canonical JSON with that field removed. The aggregator
rejects missing/extra methods, seeds, tasks, run rows, and CLI labels that disagree with the
embedded training contract.

Every training contract must also match the campaign's pre-training authorization and
model-initialization registry: campaign/stage/index/advancement-policy identity, the SHA-256
of the complete ordered prior-stage `ADVANCE` receipt chain, exact Qwen2.5-7B base bytes,
and the approved task-specialist matched-SFT bytes/lineage. The campaign embeds the same
exact receipt identities and recomputed chain digest; an omitted, reordered, duplicated,
or foreign-campaign stage fails confirmation aggregation. Equality merely between the two
compared methods is insufficient.

Confirmation run arguments use four fields:

```text
--run METHOD,TRAINING_SEED,TASK,PATH
```

`PATH` must resolve to exactly one completed task artifact. A method/seed/task row cannot be
silently inferred from its directory name.

## Typed lanes

- `raw_n8` accepts only native `record_sha256` artifacts. Any repair/rescore fingerprint or
  repair view is an error.
- `repair_n8` accepts only content-bound repair artifacts with the native K=8 provenance and
  exactly eight reported candidates.
- `repair_adaptive_n8_64` is explicitly unsupported by the current frozen policy because no
  prespecified producer exists. It is neither a required confirmation lane nor claim-eligible,
  and the aggregator rejects it. A future policy may enable it only after binding a real
  producer contract that emits per-instance stopping, K, token, runtime, and resource receipts;
  post-hoc truncation of K=64 outputs cannot stand in for adaptive inference.

All lanes enforce the same exact Alpaca no-reasoning prompt, decoder constants, source
representation switches, and strict shared verifier declared in `lane_contracts.json`.
Repair values never enter the raw table.

The output always records `claim_status=NO_SOTA_CLAIM`. Each task summary retains the source
point only under `external_context` with `comparison_status=CONTEXT_ONLY_NOT_COMPARABLE` and
`strict_beat_claim_eligible=false`; the terminal summary likewise prints
`external_context=NOT_COMPARABLE` rather than an above/below marker.

## Integrity and inference

For native, rescore, and repair schemas, aggregation recomputes every available object
self-hash, details-file SHA/size/count, summary aggregate projection, experiment fingerprint,
protocol fingerprint, verifier/data/code manifests, repair/rescore fingerprint material, and
embedded run-contract identity. Legacy artifacts are rejected for confirmation. Development
inspection can opt into legacy input only with both `--development-only` and
`--allow-legacy-provenance`; that output is not confirmatory evidence.

For beta-positive native artifacts, aggregation reopens the step-zero
`exact_peft_kl_reference.json`, step-one
`exact_peft_kl_reference_step1.json`, and final/checkpoint
`exact_peft_kl_reference_post_step.json` receipts at both the run root and
evaluated checkpoint. It requires the complete fixed-horizon lineage to equal
evaluator provenance, selection, campaign, attempt identity, and the final
aggregate receipt. Repair/rescore producers perform the same live
two-location checks before writing, then embed a self-hashed snapshot of the
exact parsed native config and canonical runtime/post-step receipt payloads.
This keeps derived artifacts relocatable: aggregation validates the embedded
byte, semantic, verifier, tensor/function, policy-change, optimizer-exclusion,
and rank-consensus identities without assuming the native config is beside the
derived summary. If an exact co-located config is present, its bytes are
rehashed as an additional check. Missing or stale evidence, a different fixed
source/verifier, or mismatched beta, dtype, world size, fixed horizon, tensors,
probes, consensus, self-hash, or file hash is fatal. Beta-zero artifacts must
not carry any of the three sidecars, embedded snapshots, or exact-KL
identities.

Local comparisons require exactly equal paired keys. Effects are averaged over training seeds
and intervals use a hierarchical seed/instance bootstrap. Within-seed McNemar/sign tests are
descriptive diagnostics. Primary exact tests exhaustively sign-flip the three paired
training-seed effects for each task, yielding exactly one p-value per task and outcome family;
Holm correction is then applied across the exact five-task family. Pooled instance metrics are retained only under
`pooled_for_descriptive_use_only` and never receive a p-value.

## Building development checkpoint evidence

Do not hand-author the input to `select_sota_checkpoint.py`. Once the checkpoint-selection
policy is frozen, derive the entire scheduled grid from completed native development artifacts:

```bash
python scripts/build_checkpoint_evidence.py \
  --policy experiments/sota_20260814/checkpoint_selection_policy.json \
  --details /path/to/run_details__checkpoint100.jsonl \
  --details /path/to/run_details__checkpoint200.jsonl \
  --output /path/to/checkpoint_evidence.json
```

Repeat `--details` for every policy row. The builder derives task, method, training seed, and
checkpoint step from the embedded run contract and resolved checkpoint—not from CLI labels.
It rehashes the on-disk run contract and every inference checkpoint file, validates native
config/summary/details closure, rejects confirmation evidence, and refuses any missing or
extra policy-grid row. Its only write is a new atomic, read-only evidence receipt.
