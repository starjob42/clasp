# CLASP confirmatory SOTA protocol

Frozen: 2026-08-14. This document separates model development from the evidence used for a headline claim. Any departure must be recorded before viewing the affected held-out results.

> **Current decision: BLOCKED; no SOTA claim.** The five-task reference-solver contract is still `BLOCKED_UNRESOLVED_DO_NOT_USE`. The existing IID v1 draft has no production references, and no schema-v3 benchmark manifest or confirmation run exists. Source snapshots v1 and v2 are immutable historical smoke snapshots; v2 predates the current identity/selection gates and is stale for confirmation. The completed TSP/CVRP 500k specialists and the queued OP/MIS/JSSP specialists use Qwen2.5-7B-Instruct, so they are historical development assets and cannot satisfy the frozen non-instruction Qwen2.5-7B initialization contract. Preserve all of them unchanged. A separate all-five-task base-model SFT campaign and a new immutable source snapshot are required before publication RL training.

## Claim boundary

The primary claim concerns direct language-model solution generation on the five supported LLMCoSolver tasks: TSP, CVRP, OP, MIS, and JSSP. It is not a claim against exact solvers, task-specific neural solvers, or LLM-designed heuristics.

Three leaderboards are reported separately:

1. **Raw N=8:** eight unmodified model completions, selected only by the shared strict validator. LLMCoSolver SFT+FOARL is historical first-100 author-protocol context, not a target for the new IID benchmark.
2. **Deterministic repair N=8:** the same eight model completions followed by the versioned `clasp-repair` operator. Raw and repaired records, actions, and repair time are all retained. This is a hybrid system, not raw CLASP and not a reproduction of FALCON.
3. **Adaptive repair (future/unsupported):** no prespecified producer currently exists, so this lane is excluded from required confirmation and all campaign claims. A future frozen producer must report its stopping rule, actual candidate count, generated tokens, GPU time, and wall time. Post-hoc truncation of K=64 outputs is not adaptive inference.

The frozen external point estimates, source URLs, publication status, and caveats live in `targets.json`. Every current external value is context-only and claim-ineligible. LLMCoSolver's Table 1 values are restricted to the released historical `eval[0:100]` author protocol; FALCON is an unmatched, unreproduced preprint system using BOPO, grammar constraints, and task-specific repair. ACCORD has an official code release and is relevant end-to-end constraint-satisfying LLM context, but its custom state-transition representation, dynamic routing, model family, task distributions, and inference budget are unmatched. It therefore has no numeric beat flag unless reproduced locally under a frozen comparison contract. None of these sources supplies paired outputs on the new IID denominator, so the aggregator does not compute “matches,” “beats,” target-gap success, or SOTA flags from their reported values. Statistical comparisons use locally rescored per-instance outputs.

The existing `METHOD=grpo` lane is not an LLMCoSolver/FOARL reproduction: it uses three strict reward heads and `loss_type=dr_grpo`, while the pinned official code uses exactly two ordered optimality/feasibility heads and the one-iteration sequence-mean GRPO loss. The official `bias=lora_only` implementation also obtains KL logits through `disable_adapter()` on the live policy; under the pinned stack its trainable Q/K/V base biases remain active in that path and are not persisted by adapter save. Keep that operational reproduction separate from the scientifically corrected, separately loaded frozen step-zero SFT reference. The corrected lane fixes reference immutability and persistence but must not be described as exact official KL semantics. Until a baseline variant and its complete data/model/budget contract are frozen, the campaign baseline remains unresolved.

External reference terminology never determines the new IID denominator. The separate reference-solver contract permits only gap-to-heuristic wording for TSP/CVRP and requires native row-level optimality certificates for OP/MIS/JSSP. In particular, new-IID OP can eventually be exact even though the released historical OP protocol is cross-protocol invalid.

## Data and leakage control

- Training uses only each task's `train/train_rl.json` (and the declared SFT training set).
- All 300 rows in the existing `eval/test.json` files are **development/replication data**. Historical jobs evaluated the full 300 and later audits repeatedly inspected `[200, 300)`, so no subset may be relabeled as unseen.
- Development and checkpoint selection may use indices `[0, 200)`. The established `[200, 300)` slice is retained only to reproduce the published-style 100-instance table; it is not confirmatory evidence.
- Before confirmation, generate a **new version** of the independent IID set with the current generator and at least 1,000 instances per task. The immutable v1 unlabeled archive predates the schema-v2 generation receipt and OP displayed-budget fix; it is historical evidence, not an asset to patch or label in place. Solve/reference the new version with declared task solvers, verify every reference under the strict validator, reject semantic overlap with SFT/RL/development data, and publish its schema-v3 manifest hash before opening model outputs.
- The executable two-phase generation, reference-ingestion, and freeze procedure is specified in `docs/independent_iid_benchmark.md`. The checked-in reference-solver contract is intentionally blocked until every solver version, binary/source hash, license, seed, status rule, and quality claim is resolved; smoke references are never confirmation data.
- Add standard OOD benchmark families where licenses and solver settings permit. They are reported separately from the IID headline.
- Dataset paths, file hashes, selected row hashes, validator code hash, model artifact hash, prompt template, and decoding configuration are part of each evaluation fingerprint. Resume is allowed only for an identical fingerprint.
- Every candidate is rescored by the repository's strict task validator. If an external method can be run, its raw completions are rescored by the same validator before a matched local comparison.

## Fixed raw-generation contract

- Model family: Qwen2.5-7B, with the exact base/SFT/RL artifact chain recorded.
- Prompt: Alpaca no-reasoning for the Jiang-matched specialist checkpoints; evaluation must fail closed if training prompt metadata is absent unless an explicit override is supplied.
- Candidates: `K=8` in one generation call.
- Sampling: seed `271828`, float16, temperature `0.7`, top-p `0.9`, top-k `50`, repetition penalty `1.0`.
- Limits: maximum 3,000 new tokens and maximum 29,768 prompt tokens.
- Selection: feasible first, then best verified objective. Declared objectives never determine feasibility or selection.
- No parsing fix, local search, regeneration, or repair is permitted in the raw lane.
- Raw confirmation must declare `--evaluation_purpose confirmation`, pass the frozen
  `--benchmark_manifest`, `--campaign_manifest`, and (for GRPO)
  `--checkpoint_selection_receipt`, evaluate every row with
  `--max_eval_samples_per_task 0`, and use `--no-prefer_subset_files`. Confirmation always
  saves the completed config, summary, CSV, and candidate-level details JSONL.

Sensitivity runs vary decoding seeds (`271829`, `271830`) without changing the primary contract. They quantify decoding variance; they do not inflate the number of independent benchmark instances.

## Frozen non-instruction base-specialist prerequisite

Publication trajectories use five newly trained task specialists initialized from one
byte-frozen **non-instruction** Qwen2.5-7B base. They do not reuse, overwrite, resume from,
or relabel the historical Qwen2.5-7B-Instruct specialist tree. All five tasks must be
trained again because a LoRA adapter's base-model identity is part of the model.

The base-SFT v7 campaign is frozen before its output root exists. Its create-once receipt
binds an explicit mode-sealed input trust root, the immutable source
snapshot, the full base-model tree and architecture signature,
the complete five-task forbidden-source registry, an independent 20-source
task-by-reservation-class authority registry, each exact 500k SFT source, the fixed
seed/hyperparameters, and the canonical fresh output root. The freezer rejects a chat
template, a writable or symlinked base tree, a partial task matrix, mutable registry or
source receipts, a non-500k SFT source, and an existing output root. Each allocation
rehashes those inputs and validates a per-task create-once run receipt before fresh or
resumed training. Every component below the trust root is reopened descriptor-relative
without following links and is bound by type, mode, device, inode, ownership, link count,
and self-hash. The current owner-controlled anchor is not administrative immutability:
its owner can chmod or rename it. The launch gate therefore requires the anchor and
direct parent to be non-writable at freeze/run time and repeats path, inode, mode, and
content validation on every allocation. An administrator-owned directory or equivalent
ACL would strengthen this boundary but is not claimed here.
Campaign protocol v7 additionally binds the serialization-safe `bias=all`
policy.  The freezer proves from the exact Qwen weight index that all 84 base
biases are targeted Q/K/V biases; every checkpoint/root must then contain those
84 biases and the expected 392 rank-64 Jiang-LoRA tensors.  PEFT `lora_only`
artifacts are rejected because PEFT 0.18.0/0.19.1 omit their trained biases on
save.  See `docs/lora_bias_persistence_remediation_20260814.md`.

Before tokenization, the five clean-corpus task receipts must be merged to the
create-once `sft_cleanliness_audit.json`. The v7 freezer requires exactly
500,000 unique demonstrations per task, zero reserved-evaluation overlap,
strictly valid labels, the live cleanliness implementation hash, and exact
equality between each audited clean output hash and the corresponding sole
registered `sft-train` file. Each detailed task receipt must remain read-only at
`data_sft/cleanliness_receipts/{task}.json`; merged validation live-reopens and
rehashes all five detailed receipts. The submitter and every worker allocation
reopen that merged identity before any scheduler or training mutation.

The clean receipts do not get to define what “reserved” means. Before the
campaign is frozen, `freeze_sft_reserved_source_registry.py` creates a
read-only authority over exactly 20 immutable JSON files: every Cartesian pair
of the five tasks and `sft_eval`, `rl_eval`, `rl_train`, and
`iid_confirmation`. Each clean-task receipt must match that authority
class-for-class in path, bytes, size, and row count. The `rl_eval` and
`rl_train` copies are independently content-matched to the ordinary forbidden
registry. All five IID copies must match the retained schema-v1 unlabeled IID
generation receipt by file hash and row count. That legacy receipt contains no
pre-IID forbidden-source registry binding; v7 records this limitation explicitly
instead of making a false temporal claim. The independently replayed clean-corpus
receipts provide direct zero-overlap evidence. The ordinary 15-tree base-SFT
registry therefore uses a distinct post-IID/pre-specialist-SFT protocol and embeds
the same retained receipt identity. This prevents a cleanliness receipt from
assigning trusted labels to incomplete or convenient reservation inputs.

Before the data tree is sealed, `scripts/audit_sft_token_lengths.py` streams all
2,500,000 SFT rows through the exact Qwen tokenizer and the production
Alpaca/no-reason task transformation.  Its create-once
`sft_token_length_audit.json` proves per-task row counts and the complete token
length histogram, and requires zero rows above 20,000 tokens, zero
prompt/completion token-prefix mismatches, and zero empty supervised
completions.  The campaign and every allocation reopen this receipt and its
tokenizer, registry, and five source bindings; the 20,000-token setting alone
is not evidence that no row was truncated. Each row contributes a compact,
ordered digest of its source locator, prompt/full lengths, boundary result, and
limit result; full token-ID arrays are compared transiently but are never JSON
serialized. The five task shards can therefore run independently and are
revalidated and merged without retokenizing.

The campaign also binds one create-once
`clasp-sealed-remote-environment-v1` receipt produced from the exact four-A100
allocation, immutable source snapshot, module stack, and Python prefix used by
the worker. The producer hashes the resolved Python executable, Torch/CUDA
build, every visible installed distribution, and the complete Python-prefix
tree (including transitive packages, native libraries, scripts, metadata, and
safe internal symlinks), plus the homogeneous A100 8.0 accelerator/driver
identity. The prefix, its direct parent, every prefix entry, the receipt, and
its create-once parent must be non-writable. Broken or prefix-escaping
symlinks, editable or system-site escapes, duplicate/missing distributions, inherited
`PYTHONPATH`, and a four-GPU non-A100 or mixed allocation fail closed. The
campaign freezer reopens this receipt; every worker allocation reopens it
before any run inspection/write, again inside the same `srun` step immediately
before `torch.distributed.run`, and again before finalization receipts. A hash
projection in the campaign and per-cell run contract never substitutes for
the live receipt. No real receipt or replacement v7 campaign is checked in: this
template remains **HOLD** until an operator creates and verifies the dedicated
read-only environment in an authorized allocation.

The freezer also publishes one separate create-once five-run authority containing the
campaign, all five task run contracts, and a self-hashed authority receipt. Its exact
entry set is transactionally published with no-replace semantics; files are `0440`, the
root is `0550`, and parent/root inode identities are bound. Each mutable run-directory
copy is byte-identical but must be a distinct inode. Submission ledgers and results stay
outside the authority.

Every 500-step checkpoint is retained through step 31,250 with
`save_total_limit=64`. After Trainer, checkpoint fidelity, optimizer, scheduler, trainer
state, and all four rank RNG files complete their distributed save, the worker
synchronizes all ranks, fsyncs the files and directory hierarchy, seals them
`0440`/`0550`, and atomically creates `base_sft_resume_fidelity.json`. The receipt binds
exact bytes and modes plus step/horizon/world-size, campaign/environment, and the
authoritative run contract. The newest checkpoint must be complete; fallback to an older
checkpoint is forbidden. Immediately before Trainer load, each rank reopens the receipt,
retains an `O_RDONLY|O_DIRECTORY|O_NOFOLLOW` descriptor, and supplies
`/proc/self/fd/<n>` to Trainer. Resume then proves the full loaded adapter trainable
closure, unwrapped optimizer and scheduler state/layout, trajectory-critical Trainer
state, and exact per-rank RNG restore before the first resumed update. Matrix submission
requires 1 TiB free and each allocation rechecks a 200 GiB cell floor.

Publication eligibility additionally requires a read-only, self-hashed
`checkpoint_fidelity.json` beside every admitted SFT, retained RL checkpoint,
and finalized run root. The receipt closes the exact serialized trainable-state
inventory, adapter configuration and weights, and the Qwen2.5-7B structural
invariant. Run-contract schema v2 and model-initialization-registry protocol v2
embed a path-free `clasp-checkpoint-fidelity-identity-v1` projection containing
the receipt semantic/file hashes and the adapter, tensor-inventory, and
publication-invariant hashes. Every consumer reopens the live receipt at its
artifact boundary; an embedded identity alone never substitutes for live
verification. Missing, writable, incomplete, mismatched, or self-hash-invalid
receipts fail closed. Historical adapters and explicit legacy saved-policy
conversions remain development-only and cannot enter selection or a SOTA claim.
The advancement audit additionally requires the finalized run root to have the
same fidelity identity as `checkpoint-<target_step>`. A development-selected
nonterminal checkpoint remains independently bound to its own live identity;
it is never assumed to equal the finalized root.

The final SFT root also requires `sft_reload_equivalence.json`.  After Trainer
and the persistence audit exit, a separate process independently reloads the
saved adapter twice.  The first load's complete trainable closure must equal
the pre-save/serialized `checkpoint_fidelity.json` closure exactly—including
all 84 Q/K/V biases and all 392 attention+MLP LoRA tensors—and both loads must
have bitwise-identical full state and fixed-probe bfloat16 CUDA logits,
log-probabilities, and argmax decisions.  The five-task model-initialization
registry embeds a compact identity for this receipt and refuses any specialist
without it.

Any publication trajectory with `beta>0` instead continues the exact approved
`bias=all` SFT adapter as the policy and independently loads the same immutable
base-plus-adapter state as an explicit frozen TRL reference. Merging is forbidden:
it changes the numerical object and can introduce dtype-dependent drift. Before
the run contract is created, `CLASP_KL_REFERENCE_MODEL_PATH` must equal the policy
initialization adapter and `CLASP_KL_REFERENCE_RECEIPT` must equal that adapter's
authoritative `checkpoint_fidelity.json`. The contract carries the path-free
`clasp-exact-peft-kl-source-identity-v2`, including immutable adapter/base/tokenizer
trees, saved-FP32 LoRA dtype evidence, checkpoint-fidelity hashes, and the exact
verifier implementation.

After Trainer construction but before optimization, the runtime verifier proves
distinct policy/reference objects and storage, bitwise-equal full tensor state,
zero trainable reference tensors, exact fixed-probe FP32 logits/log-probabilities/
argmax, prepared-model identity, and distributed rank consensus. It writes the
read-only `exact_peft_kl_reference.json` receipt at the run root and copies the
same bytes into every retained checkpoint and finalized root before checkpoint
fidelity is sealed. Development evidence, selection receipts, campaign cells,
confirmation attempt identities/provenance, advancement audits, and aggregates
all carry and live-reopen the same path-free source, step-zero runtime, and
post-optimization identities. After optimization starts, beta-positive runs
also preserve `exact_peft_kl_reference_step1.json` and
`exact_peft_kl_reference_post_step.json` at the root and every retained
checkpoint. Validation reopens all three sidecars at both the root and selected
checkpoint and compares beta, dtype, world size, fixed final horizon,
source/verifier identity, receipt self/file hashes, tensor/functional hashes,
optimizer exclusion, policy-change evidence, and rank consensus. `beta=0`
requires all three exact-KL identities to be null and rejects all three
sidecars at both locations; `beta>0` requires the complete lineage exactly.
Development merge receipts and legacy conversions are never publication- or
SOTA-eligible.

```bash
set -euo pipefail
# TEMPLATE ONLY. First publish a new immutable source snapshot and make a verified
# copy of the non-instruction base artifact read-only. Never point these commands at
# the historical Instruct output root or mutable source checkout.
# HOLD: an operator must first create a dedicated Python environment at SEALED_ENV,
# verify its packages, remove caches, and make the prefix and its direct parent
# non-writable. These commands must run inside an authorized four-A100 allocation
# after loading exactly arch/a100 and miniforge/24.9.0. ENV_RECEIPT_DIR must not exist.
IMMUTABLE_INPUT_TRUST_ROOT=/lustre/fsn1/projects/rech/vas/uag72kz/clasp_base_sft500k_clean_20260821_anchor_v1/inputs
SOURCE_ROOT="${IMMUTABLE_INPUT_TRUST_ROOT}/source-snapshot"
BASE_MODEL_PATH="${IMMUTABLE_INPUT_TRUST_ROOT}/Qwen2.5-7B-base_86e6e1ec"
RELEASED_SFT500K_ROOT="${IMMUTABLE_INPUT_TRUST_ROOT}/released_sft500k_v1"
FORBIDDEN_SOURCE_REGISTRY="${IMMUTABLE_INPUT_TRUST_ROOT}/forbidden-source-registry.json"
SFT_RESERVED_SOURCE_ROOT="${IMMUTABLE_INPUT_TRUST_ROOT}/reserved_sources_v1"
SFT_RESERVED_SOURCE_REGISTRY="${IMMUTABLE_INPUT_TRUST_ROOT}/sft_reserved_source_registry.json"
IID_GENERATION_RECEIPT="${SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/generation_receipt.json"
BASE_SFT_RESERVED_SFT_EVAL_CVRP="${SFT_RESERVED_SOURCE_ROOT}/sft_eval/cvrp/test.json"
BASE_SFT_RESERVED_SFT_EVAL_JSSP="${SFT_RESERVED_SOURCE_ROOT}/sft_eval/jssp/eval.json"
BASE_SFT_RESERVED_SFT_EVAL_MIS="${SFT_RESERVED_SOURCE_ROOT}/sft_eval/mis/test.json"
BASE_SFT_RESERVED_SFT_EVAL_OP="${SFT_RESERVED_SOURCE_ROOT}/sft_eval/op/eval.json"
BASE_SFT_RESERVED_SFT_EVAL_TSP="${SFT_RESERVED_SOURCE_ROOT}/sft_eval/tsp/test.json"
BASE_SFT_RESERVED_RL_EVAL_CVRP="${SFT_RESERVED_SOURCE_ROOT}/rl_eval/cvrp/test.json"
BASE_SFT_RESERVED_RL_EVAL_JSSP="${SFT_RESERVED_SOURCE_ROOT}/rl_eval/jssp/test.json"
BASE_SFT_RESERVED_RL_EVAL_MIS="${SFT_RESERVED_SOURCE_ROOT}/rl_eval/mis/test.json"
BASE_SFT_RESERVED_RL_EVAL_OP="${SFT_RESERVED_SOURCE_ROOT}/rl_eval/op/test.json"
BASE_SFT_RESERVED_RL_EVAL_TSP="${SFT_RESERVED_SOURCE_ROOT}/rl_eval/tsp/test.json"
BASE_SFT_RESERVED_RL_TRAIN_CVRP="${SFT_RESERVED_SOURCE_ROOT}/rl_train/cvrp/train_rl.json"
BASE_SFT_RESERVED_RL_TRAIN_JSSP="${SFT_RESERVED_SOURCE_ROOT}/rl_train/jssp/train_rl.json"
BASE_SFT_RESERVED_RL_TRAIN_MIS="${SFT_RESERVED_SOURCE_ROOT}/rl_train/mis/train_rl.json"
BASE_SFT_RESERVED_RL_TRAIN_OP="${SFT_RESERVED_SOURCE_ROOT}/rl_train/op/train_rl.json"
BASE_SFT_RESERVED_RL_TRAIN_TSP="${SFT_RESERVED_SOURCE_ROOT}/rl_train/tsp/train_rl.json"
BASE_SFT_RESERVED_IID_CONFIRMATION_CVRP="${SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/cvrp/test.json"
BASE_SFT_RESERVED_IID_CONFIRMATION_JSSP="${SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/jssp/test.json"
BASE_SFT_RESERVED_IID_CONFIRMATION_MIS="${SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/mis/test.json"
BASE_SFT_RESERVED_IID_CONFIRMATION_OP="${SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/op/test.json"
BASE_SFT_RESERVED_IID_CONFIRMATION_TSP="${SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/tsp/test.json"
SFT_DATA_ROOT="${IMMUTABLE_INPUT_TRUST_ROOT}/data_sft"
SEALED_ENV=/path/to/new/read-only-parent/publication-sft-environment
SEALED_PYTHON="$(readlink -f "${SEALED_ENV}/bin/python")"
ENV_RECEIPT_DIR=/path/to/new/sealed-remote-environment-v1
REMOTE_ENVIRONMENT_RECEIPT="${ENV_RECEIPT_DIR}/remote_environment.json"
RUN_CONTRACT_AUTHORITY_ROOT=/path/to/create-once/base-sft-v7-authority
BASE_SFT_OUTPUT_ROOT=/path/to/fresh/base-specialist-v7-output
module purge
module load arch/a100
module load miniforge/24.9.0
conda activate "${SEALED_ENV}"
test ! -e "${ENV_RECEIPT_DIR}"
test ! -w "${SEALED_ENV}" && test ! -w "$(dirname "${SEALED_ENV}")"
export PATH="${SEALED_ENV}/bin:/usr/bin:/bin"
export PYTHONPATH="${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS
while IFS='=' read -r inherited_name _; do
  case "${inherited_name}" in
    CLASP_*|PD_*|GENERATION_*|GRPO_*|PYTORCH_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|WANDB_*|NCCL_*) unset "${inherited_name}" ;;
  esac
done < <(env)
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/freeze_remote_environment.py" \
  --output "${REMOTE_ENVIRONMENT_RECEIPT}"

# Build every final clean corpus create-once. RELEASED_* names are the five
# audited original 500k files. Replacement argument arrays contain ordered
# (--replacement-json, --replacement-receipt) pairs from sealed solver-backed
# shards; use empty arrays only when the independent full replay found no
# shortfall. Never guess a deficit or reuse a shard. The builder itself requires
# exactly 500,000 unique, strict-valid, reservation-disjoint demonstrations.
RELEASED_CVRP_TRAIN="${RELEASED_SFT500K_ROOT}/cvrp/train_cvrp.json"
RELEASED_JSSP_TRAIN="${RELEASED_SFT500K_ROOT}/jssp/training_jssp.json"
RELEASED_MIS_TRAIN="${RELEASED_SFT500K_ROOT}/mis/mis_train.json"
RELEASED_OP_TRAIN="${RELEASED_SFT500K_ROOT}/op/train_op.json"
RELEASED_TSP_TRAIN="${RELEASED_SFT500K_ROOT}/tsp/train_tsp.json"
for released_source in \
  "${RELEASED_CVRP_TRAIN}" "${RELEASED_JSSP_TRAIN}" \
  "${RELEASED_MIS_TRAIN}" "${RELEASED_OP_TRAIN}" \
  "${RELEASED_TSP_TRAIN}"; do
  test -f "${released_source}" && test ! -w "${released_source}"
done
: "${SFT_CLEAN_TASK_RECEIPT_DIR:=${SFT_DATA_ROOT}/cleanliness_receipts}"
declare -a CVRP_REPLACEMENT_ARGS JSSP_REPLACEMENT_ARGS MIS_REPLACEMENT_ARGS
declare -a OP_REPLACEMENT_ARGS TSP_REPLACEMENT_ARGS
# Example for each nonempty array (repeat pairs in ascending shard order):
# CVRP_REPLACEMENT_ARGS=(
#   --replacement-json /sealed/replacements/cvrp/shard-000/replacements.json
#   --replacement-receipt /sealed/replacements/cvrp/shard-000/replacement_receipt.json
# )
for task in cvrp jssp mis op tsp; do
  for source_class in sft_eval rl_eval rl_train iid_confirmation; do
    variable="BASE_SFT_RESERVED_${source_class^^}_${task^^}"
    source_path="${!variable:?set ${variable} to its canonical reserved JSON file}"
    test -f "${source_path}"
  done
done
mkdir -p \
  "${SFT_DATA_ROOT}/cvrp/train" "${SFT_DATA_ROOT}/jssp/train" \
  "${SFT_DATA_ROOT}/mis/train" "${SFT_DATA_ROOT}/op/train" \
  "${SFT_DATA_ROOT}/tsp/train" "${SFT_CLEAN_TASK_RECEIPT_DIR}"
test ! -e "${SFT_DATA_ROOT}/cvrp/train/train_cvrp.json"
test ! -e "${SFT_CLEAN_TASK_RECEIPT_DIR}/cvrp.json"
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/build_clean_sft500k_corpus.py" \
  --task cvrp --source-json "${RELEASED_CVRP_TRAIN}" \
  --reserved-source "sft_eval,${BASE_SFT_RESERVED_SFT_EVAL_CVRP}" \
  --reserved-source "rl_eval,${BASE_SFT_RESERVED_RL_EVAL_CVRP}" \
  --reserved-source "rl_train,${BASE_SFT_RESERVED_RL_TRAIN_CVRP}" \
  --reserved-source "iid_confirmation,${BASE_SFT_RESERVED_IID_CONFIRMATION_CVRP}" \
  "${CVRP_REPLACEMENT_ARGS[@]}" \
  --output-json "${SFT_DATA_ROOT}/cvrp/train/train_cvrp.json" \
  --receipt "${SFT_CLEAN_TASK_RECEIPT_DIR}/cvrp.json" --expected-rows 500000
test ! -e "${SFT_DATA_ROOT}/jssp/train/training_jssp.json"
test ! -e "${SFT_CLEAN_TASK_RECEIPT_DIR}/jssp.json"
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/build_clean_sft500k_corpus.py" \
  --task jssp --source-json "${RELEASED_JSSP_TRAIN}" \
  --repair-jssp-terminal-makespan \
  --reserved-source "sft_eval,${BASE_SFT_RESERVED_SFT_EVAL_JSSP}" \
  --reserved-source "rl_eval,${BASE_SFT_RESERVED_RL_EVAL_JSSP}" \
  --reserved-source "rl_train,${BASE_SFT_RESERVED_RL_TRAIN_JSSP}" \
  --reserved-source "iid_confirmation,${BASE_SFT_RESERVED_IID_CONFIRMATION_JSSP}" \
  "${JSSP_REPLACEMENT_ARGS[@]}" \
  --output-json "${SFT_DATA_ROOT}/jssp/train/training_jssp.json" \
  --receipt "${SFT_CLEAN_TASK_RECEIPT_DIR}/jssp.json" --expected-rows 500000
# Stop unless the read-only JSSP receipt reports source_rows=500000,
# accepted_source_rows=500000, source_repairs.repaired_rows=6633,
# replacement_rows=0, additional_replacements_required=0, and clean_rows=500000.
# Any different repair count or any nonrepairable/replacement shortfall requires
# a new audit; do not add a JSSP replacement shard to make the command pass.
"${SEALED_PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1], encoding="utf-8")); observed={"source_rows":r["source_rows"],"accepted_source_rows":r["accepted_source_rows"],"repaired_rows":r["source_repairs"]["repaired_rows"],"replacement_rows":r["replacement_rows"],"additional_replacements_required":r["additional_replacements_required"],"clean_rows":r["clean_rows"]}; expected={"source_rows":500000,"accepted_source_rows":500000,"repaired_rows":6633,"replacement_rows":0,"additional_replacements_required":0,"clean_rows":500000}; sys.exit(0 if observed == expected else f"unexpected JSSP recovery receipt: {observed!r}")' \
  "${SFT_CLEAN_TASK_RECEIPT_DIR}/jssp.json"
test ! -e "${SFT_DATA_ROOT}/mis/train/mis_train.json"
test ! -e "${SFT_CLEAN_TASK_RECEIPT_DIR}/mis.json"
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/build_clean_sft500k_corpus.py" \
  --task mis --source-json "${RELEASED_MIS_TRAIN}" \
  --reserved-source "sft_eval,${BASE_SFT_RESERVED_SFT_EVAL_MIS}" \
  --reserved-source "rl_eval,${BASE_SFT_RESERVED_RL_EVAL_MIS}" \
  --reserved-source "rl_train,${BASE_SFT_RESERVED_RL_TRAIN_MIS}" \
  --reserved-source "iid_confirmation,${BASE_SFT_RESERVED_IID_CONFIRMATION_MIS}" \
  "${MIS_REPLACEMENT_ARGS[@]}" \
  --output-json "${SFT_DATA_ROOT}/mis/train/mis_train.json" \
  --receipt "${SFT_CLEAN_TASK_RECEIPT_DIR}/mis.json" --expected-rows 500000
test ! -e "${SFT_DATA_ROOT}/op/train/train_op.json"
test ! -e "${SFT_CLEAN_TASK_RECEIPT_DIR}/op.json"
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/build_clean_sft500k_corpus.py" \
  --task op --source-json "${RELEASED_OP_TRAIN}" \
  --reserved-source "sft_eval,${BASE_SFT_RESERVED_SFT_EVAL_OP}" \
  --reserved-source "rl_eval,${BASE_SFT_RESERVED_RL_EVAL_OP}" \
  --reserved-source "rl_train,${BASE_SFT_RESERVED_RL_TRAIN_OP}" \
  --reserved-source "iid_confirmation,${BASE_SFT_RESERVED_IID_CONFIRMATION_OP}" \
  "${OP_REPLACEMENT_ARGS[@]}" \
  --output-json "${SFT_DATA_ROOT}/op/train/train_op.json" \
  --receipt "${SFT_CLEAN_TASK_RECEIPT_DIR}/op.json" --expected-rows 500000
test ! -e "${SFT_DATA_ROOT}/tsp/train/train_tsp.json"
test ! -e "${SFT_CLEAN_TASK_RECEIPT_DIR}/tsp.json"
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/build_clean_sft500k_corpus.py" \
  --task tsp --source-json "${RELEASED_TSP_TRAIN}" \
  --reserved-source "sft_eval,${BASE_SFT_RESERVED_SFT_EVAL_TSP}" \
  --reserved-source "rl_eval,${BASE_SFT_RESERVED_RL_EVAL_TSP}" \
  --reserved-source "rl_train,${BASE_SFT_RESERVED_RL_TRAIN_TSP}" \
  --reserved-source "iid_confirmation,${BASE_SFT_RESERVED_IID_CONFIRMATION_TSP}" \
  "${TSP_REPLACEMENT_ARGS[@]}" \
  --output-json "${SFT_DATA_ROOT}/tsp/train/train_tsp.json" \
  --receipt "${SFT_CLEAN_TASK_RECEIPT_DIR}/tsp.json" --expected-rows 500000

# Merge the five completed task receipts into the authoritative data-root
# receipt before the token audit and trust-root seal.
: "${SFT_CLEAN_TASK_RECEIPT_DIR:?set the canonical task-receipt directory}"
test ! -e "${SFT_DATA_ROOT}/sft_cleanliness_audit.json"
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/build_clean_sft500k_corpus.py" \
  --data-root "${SFT_DATA_ROOT}" \
  --merge-task-receipt "${SFT_CLEAN_TASK_RECEIPT_DIR}/cvrp.json" \
  --merge-task-receipt "${SFT_CLEAN_TASK_RECEIPT_DIR}/jssp.json" \
  --merge-task-receipt "${SFT_CLEAN_TASK_RECEIPT_DIR}/mis.json" \
  --merge-task-receipt "${SFT_CLEAN_TASK_RECEIPT_DIR}/op.json" \
  --merge-task-receipt "${SFT_CLEAN_TASK_RECEIPT_DIR}/tsp.json" \
  --receipt "${SFT_DATA_ROOT}/sft_cleanliness_audit.json"
test ! -w "${SFT_DATA_ROOT}/sft_cleanliness_audit.json"

# Freeze the independent reservation authority. The 20 explicit variables
# above name the one canonical read-only JSON file below each corresponding
# reserved_sources_v1/CLASS/TASK subtree. The registry rejects missing, duplicate,
# writable, non-JSON, out-of-subtree, or content-changing inputs.
test ! -e "${SFT_RESERVED_SOURCE_REGISTRY}"
reserved_source_args=()
for task in cvrp jssp mis op tsp; do
  for source_class in sft_eval rl_eval rl_train iid_confirmation; do
    variable="BASE_SFT_RESERVED_${source_class^^}_${task^^}"
    source_path="${!variable:?set ${variable} to its canonical reserved JSON file}"
    test ! -w "${source_path}"
    reserved_source_args+=(--source "${task},${source_class},${source_path}")
  done
done
test ! -w "${IID_GENERATION_RECEIPT}"
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/freeze_sft_reserved_source_registry.py" \
  --immutable-input-trust-root "${IMMUTABLE_INPUT_TRUST_ROOT}" \
  "${reserved_source_args[@]}" \
  --iid-generation-receipt "${IID_GENERATION_RECEIPT}" \
  --output "${SFT_RESERVED_SOURCE_REGISTRY}"
test ! -w "${SFT_RESERVED_SOURCE_REGISTRY}"

# Now that the final clean files and retained IID receipt exist, freeze the
# distinct post-IID/pre-specialist-SFT 15-tree registry. It must never claim
# frozen_before_confirmation_generation: the retained schema-v1 IID receipt has
# no pre-IID forbidden-registry binding.
# Its RL sources intentionally point to the same canonical files as the
# authority's rl_train and rl_eval classes; campaign v7 content-cross-checks
# both registries. sft_eval remains a distinct authority-only class.
test ! -e "${FORBIDDEN_SOURCE_REGISTRY}"
forbidden_source_args=()
for task in cvrp jssp mis op tsp; do
  rl_train_variable="BASE_SFT_RESERVED_RL_TRAIN_${task^^}"
  rl_eval_variable="BASE_SFT_RESERVED_RL_EVAL_${task^^}"
  forbidden_source_args+=(
    --source "${task},sft-train,${SFT_DATA_ROOT}/${task}/train"
    --source "${task},rl-train,${!rl_train_variable}"
    --source "${task},historical-eval,${!rl_eval_variable}"
  )
done
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/freeze_forbidden_source_registry.py" \
  "${forbidden_source_args[@]}" \
  --base-sft-post-iid-generation-receipt "${IID_GENERATION_RECEIPT}" \
  --output "${FORBIDDEN_SOURCE_REGISTRY}"
test ! -w "${FORBIDDEN_SOURCE_REGISTRY}"

# Run one shard per task while SFT_DATA_ROOT is still writable, after the
# forbidden-source registry is final but before sealing the trust root. The
# shard directory is external audit evidence and must be fresh and canonical.
test ! -e "${SFT_DATA_ROOT}/sft_token_length_audit.json"
: "${SFT_TOKEN_AUDIT_SHARD_DIR:?set a fresh canonical shard directory}"
test ! -e "${SFT_TOKEN_AUDIT_SHARD_DIR}"
mkdir -m 0750 "${SFT_TOKEN_AUDIT_SHARD_DIR}"
token_audit_pids=()
for task in cvrp jssp mis op tsp; do
  "${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/audit_sft_token_lengths.py" \
    --data-root "${SFT_DATA_ROOT}" \
    --tokenizer-path "${BASE_MODEL_PATH}" \
    --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}" \
    --batch-size 256 --max-seq-length 20000 \
    --expected-rows-per-task 500000 --task "${task}" \
    --output "${SFT_TOKEN_AUDIT_SHARD_DIR}/${task}.json" &
  token_audit_pids+=("$!")
done
for token_audit_pid in "${token_audit_pids[@]}"; do
  wait "${token_audit_pid}"
done
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/audit_sft_token_lengths.py" \
  --data-root "${SFT_DATA_ROOT}" \
  --tokenizer-path "${BASE_MODEL_PATH}" \
  --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}" \
  --merge-shard-dir "${SFT_TOKEN_AUDIT_SHARD_DIR}" \
  --max-seq-length 20000 --expected-rows-per-task 500000 \
  --output "${SFT_DATA_ROOT}/sft_token_length_audit.json"
test ! -w "${SFT_DATA_ROOT}/sft_token_length_audit.json"

# Mode-seal the complete input trust root only after the audit receipt exists.
# This owner-controlled seal is mutation detection, not administrative
# immutability: the following campaign gate requires every component and the
# direct parent to be non-writable, binds path/inode/content identity, and
# refuses to create or repair the receipt itself.
test ! -e "${RUN_CONTRACT_AUTHORITY_ROOT}"
test ! -e "${BASE_SFT_OUTPUT_ROOT}"
test ! -w "${IMMUTABLE_INPUT_TRUST_ROOT}"
test ! -w "$(dirname "${IMMUTABLE_INPUT_TRUST_ROOT}")"
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/freeze_base_sft_campaign.py" \
  --campaign-id qwen25-base-specialists-500k-v7 \
  --run-contract-authority-root "${RUN_CONTRACT_AUTHORITY_ROOT}" \
  --immutable-input-trust-root "${IMMUTABLE_INPUT_TRUST_ROOT}" \
  --source-root "${SOURCE_ROOT}" \
  --base-model-path "${BASE_MODEL_PATH}" \
  --forbidden-source-registry "${FORBIDDEN_SOURCE_REGISTRY}" \
  --reserved-source-registry "${SFT_RESERVED_SOURCE_REGISTRY}" \
  --remote-environment-receipt "${REMOTE_ENVIRONMENT_RECEIPT}" \
  --sft-data-root "${SFT_DATA_ROOT}" \
  --output-root "${BASE_SFT_OUTPUT_ROOT}" \
  --output "${RUN_CONTRACT_AUTHORITY_ROOT}/base_sft_campaign.json"

CLASP_PROJECT_ROOT="${SOURCE_ROOT}" \
BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT="${IMMUTABLE_INPUT_TRUST_ROOT}" \
BASE_SFT_CAMPAIGN_CONTRACT="${RUN_CONTRACT_AUTHORITY_ROOT}/base_sft_campaign.json" \
BASE_SFT_FORBIDDEN_SOURCE_REGISTRY="${FORBIDDEN_SOURCE_REGISTRY}" \
BASE_SFT_RESERVED_SOURCE_REGISTRY="${SFT_RESERVED_SOURCE_REGISTRY}" \
BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT="${REMOTE_ENVIRONMENT_RECEIPT}" \
BASE_SFT_PYTHON="${SEALED_PYTHON}" \
BASE_SFT_BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
BASE_SFT_DATA_ROOT="${SFT_DATA_ROOT}" \
BASE_SFT_OUTPUT_ROOT="${BASE_SFT_OUTPUT_ROOT}" \
  bash "${SOURCE_ROOT}/scripts/submit_sft500k_base_specialists_a100.sh"
```

The submitter has a fixed sorted matrix of TSP/CVRP/OP/MIS/JSSP, two sequential
20-hour allocations per task, seed 42, and a create-once submission ledger. It carries
only explicit contract variables into Slurm. A failed partial submission is audited;
the output root is never silently reused. Only five finalized, optimizer-resumable,
campaign-bound specialists may be admitted to `freeze_model_initialization_registry.py`.

## Training matrix and gates

All trainable methods start from the same task-specialist SFT artifact, use the same task rows/order, LoRA policy, optimizer, rollout settings, fixed 3,000-token completion cap, total optimizer-step horizon, and seeds. Adaptive completion length is disabled for every core comparison and may appear only as a separately named systems ablation with token/time accounting. The core methods are:

- standard GRPO;
- strong scalarized dual-GRPO;
- CLASP with dense lexicographic set credit and active component dual;
- CLASP exact leave-one-out credit (estimator ablation).

The experiment is staged to avoid spending the full allocation on broken or clearly dominated configurations:

| Stage | Data visible | Runs | Advancement rule |
|---|---|---|---|
| invariant smoke | synthetic/local only | all methods, no GPU claim | all tests, state round-trip, and reward-order invariants pass |
| 40-step GPU smoke | development only | TSP and CVRP, seed 42 | finite loss/advantages, nonzero objective credit, strict resume-readiness, no group/aux skips |
| 400-step pilot | development only | all five tasks when SFT exists, seed 42 | no >2 pp pass@8 regression and improvement in feasibility or gap on at least three tasks |
| 3,200-step selection | existing 300-row development pool only | CLASP and strongest matched baseline, seeds 42/43/44 | choose checkpoint by a prespecified lexicographic development score |
| published-slice replication | existing `[200,300)` | one frozen checkpoint per task/seed/method | comparison table only; explicitly marked previously observed |
| confirmation | new hash-frozen IID/OOD sets | one frozen checkpoint per task/seed/method | run once, aggregate without retuning |

Development checkpoint selection first maximizes pass@8, then minimizes the
conditional strict-verifier gap to the task's frozen development reference among
checkpoints within one percentage point of the best pass@8. Ties use the earlier
checkpoint. This is called an optimality gap only where the reference is certified
optimal (OP/MIS/JSSP); no test metric enters the rule.

## Development to confirmation: mandatory order

1. Before any publication trajectory or development evidence is produced, run `freeze_forbidden_source_registry.py` on the complete authoritative SFT-train, RL-train, and historical-eval trees for all five tasks and publish its registry/tree hashes. Every training contract verifies its actual SFT/RL trees against this receipt.
2. From a new immutable source snapshot, freeze and complete the separate all-five-task non-instruction base-specialist campaign above. Do not alter or wait to repurpose the historical Instruct campaign. Audit every final SFT adapter, optimizer checkpoint, RNG state, data/source/base identity, and run receipt; then freeze the five-task model-initialization registry.
3. Create a new initial 40-step advancement policy only after the initialization-registry hash exists. It must declare `stage_index=0`, an empty prior-stage receipt chain, the exact model-initialization registry hash, the new source-snapshot ID, and a never-used output root. Complete the prespecified development grid and audit it with `audit_sota_advancement.py`. A `HOLD` or missing seed remains in the denominator and cannot be bypassed. Every non-initial advancement policy must embed the complete chronological identities of all prior `ADVANCE` receipts, and the auditor/launcher must reopen and validate those immutable receipts before authorizing the next stage. The resulting chain digest and stage index remain bound in every training contract and in the later confirmation campaign.
4. Before opening checkpoint-level development results, freeze the complete checkpoint-selection policy with the full registry snapshot and exact per-task `historical-eval` identities embedded. Run `select_sota_checkpoint.py` only on evidence whose data files exactly equal those registered identities and whose `evidence_scope=development_only`. Its create-once `SELECTED` receipt records the policy/evidence self-hashes, exact checkpoint step and artifact hash, all three checkpoint-fidelity identity/receipt/file hashes, the exact GRPO beta, prelaunch exact-source identity, post-Trainer step-zero exact-runtime identity, and full post-optimization reference-immutability lineage (all three exact-KL fields null exactly when beta is zero), immutable training `run_contract_sha256`, data manifest, verifier, and evaluation fingerprint. For beta-positive runs the lineage reopens the root and selected-checkpoint step-one/post-step receipts, binds their file and semantic hashes to the fixed horizon and policy closure, and requires changed selected-checkpoint trainables even though the warmup schedule may leave the step-one policy unchanged.
5. Resolve the reference-solver contract and create a fresh source snapshot. Reuse the same pre-results registry when generating the new IID version, attach evidence-bearing references, and freeze it with `freeze_benchmark.py --forbidden-source-registry <receipt>`. Receipt schema v2 and manifest schema v3 bind the solver contract, strict verifier, ordered semantic instances, normalized benchmark rows, reference outputs, and the exact pre-generation forbidden-source file matrix. Partial non-empty trees are invalid. Publish the manifest hash before any inference.
6. Invoke the evaluator explicitly; no launcher or evaluator selects a checkpoint during confirmation:

```bash
# TEMPLATE ONLY: TSP specialist shown; do not run while the gate remains BLOCKED.
clasp-eval \
  --evaluation_purpose confirmation \
  --benchmark_manifest /path/to/new/preinference_iid_manifest_v3.json \
  --campaign_manifest /path/to/frozen_campaign_manifest.json \
  --checkpoint_selection_receipt /path/to/development/checkpoint_selection_receipt.json \
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

The evaluator rejects a mismatched manifest self-hash, campaign grid cell, deterministic
attempt identity, purpose/data selection, selection receipt, training contract, checkpoint
step, checkpoint artifact, or checkpoint-fidelity identity. The selection receipt,
confirmation campaign cell, evaluator provenance, and aggregate must agree on the
checkpoint-fidelity semantic receipt hash and exact receipt-file hash. The frozen campaign
names one shared canonical create-once
attempt-ledger root; each grid cell has one durable locked claim and an identical read-only
output marker. Deleting or relocating a failed output cannot create a second attempt.
Development and replication commands must not carry confirmation manifests or receipts.
Confirmation uses resume only for an identical fingerprint and must never use `--no-resume`
to replace a prior confirmation attempt. Confirmation forbids mutable best-run updates and
any row/size/subset filtering.

Only after every scheduled seed has a completed artifact set, build the raw table through
the manifest-gated publication path. Confirmation requires the schema-v2 campaign manifest
and all 80 exact `METHOD,SEED,TASK,PATH` cells (two methods x eight seeds x five tasks).
The loop below constructs that exact Cartesian grid; a missing, duplicate, extra, or legacy
method ID is rejected:

```bash
run_args=()
for method in grpo_z clasp_canonical; do
  for seed in 42 43 44 45 46 47 48 49; do
    for task in cvrp jssp mis op tsp; do
      run_args+=(--run "${method},${seed},${task},/path/to/${method}/seed${seed}/${task}/run_details__....jsonl")
    done
  done
done

python scripts/aggregate_sota.py \
  --benchmark-manifest /path/to/new/preinference_iid_manifest_v3.json \
  --campaign-manifest /path/to/frozen_campaign_manifest.json \
  --targets experiments/sota_20260814/targets.json \
  --lane raw_n8 \
  "${run_args[@]}" \
  --baseline grpo_z --challenger clasp_canonical \
  --output /path/to/create-once/raw_n8_confirmation_aggregate.json
```

Run deterministic repair in its separately named output; never combine its records with
`raw_n8`. The adaptive lane remains disabled until a future frozen producer contract exists.

## Primary outcomes and statistics

For every task and each of the eight prespecified training seeds (42 through 49), report:

- pass@8 with Wilson 95% interval;
- candidate-level feasibility;
- conditional gap to the frozen reference with an instance bootstrap interval
  (called conditional optimality gap only for rows accepted by the separate
  reference-solver certificate contract);
- median and tail gap (P90), generated tokens, wall time, and peak GPU memory;
- parse-failure and named constraint-failure rates.

For local method comparisons, pair on `(task, training_seed, decoding_seed, example_index)`.
The sole primary estimand is the paired-seed mean challenger-minus-baseline difference in
normalized feasible-gap success-curve AUC: integrate
`I(strictly feasible and verified_gap <= t)` over the frozen `t in [0,0.10]` range and
divide by `0.10`; infeasible records score zero. For each task, exhaustively sign-flip the
eight paired seed effects in a two-sided exact test, then apply Holm across exactly the five
task hypotheses. A significant negative mean is labeled
`difference_detected_against_clasp`; primary superiority support requires both Holm rejection
and a positive paired mean.

Pass@8 is a separate secondary noninferiority gate with margin `0.01`. For every task, test
`H0: Delta <= -0.01` at the boundary by exhaustively applying the one-sided sign-flip test to
`d_seed + 0.01`, Holm-adjust the five task p-values, and also require observed
`mean(d_seed) >= -0.01`. The overall gate passes only if all five tasks satisfy both
requirements. Candidate feasibility, conditional verified gap, lexicographic wins, eligible
external-target success, and compute remain secondary/descriptive. Report every seed
individually and mean/standard deviation over eight seeds. Within-seed McNemar/sign tests and
pooled instances are descriptive only; decoding seeds are sensitivity analyses, not extra
training replicates. The hierarchical bootstrap resamples training seeds first and paired
instances second.

A raw-lane headline cannot be generated from `targets.json`. Any later manuscript-level
“new SOTA” claim requires all of the following external evidence in addition to the local
aggregate:

1. the frozen raw contract and strict validator were used;
2. a separately frozen, locally rescored historical first-100 author-protocol replication with per-instance outputs, plus the new IID comparison; author-reported aggregate point estimates alone are insufficient;
3. the frozen primary directional decision supports the challenger and the separate all-task pass@8 noninferiority gate passes;
4. failures, exclusions, and all scheduled seeds are reported;
5. the claim names the exact historical replication and IID protocols rather than saying “all combinatorial optimization.”

If only some tasks satisfy these conditions, the result is described task by task. A repaired-system claim follows the same rule in its own lane and never substitutes repaired values into the raw table.

## Required ablations

- dense lexicographic credit versus exact leave-one-out;
- active component dual versus fixed/raw recovery weights;
- task-level set dual versus no set dual;
- SFT versus standard GRPO versus scalarized dual-GRPO;
- K at train time (`4/8/16`, development only) while test K remains 8;
- raw output versus deterministic repair, with repair success, objective delta, and runtime broken out;
- specialist versus unified SFT initialization after all matched 500k specialists finish.

## Reproducibility artifacts

Each run must preserve the submitted script, resolved arguments/environment, Slurm job metadata, software versions, source snapshot hash, dataset manifest, model hashes, immutable training contract and its self-hash, training logs, every checkpoint-selection decision, raw completions, strict rescoring output, aggregation command, and machine-readable summary. The independent-test manifest and overlap audit must predate test inference. Failed and preempted jobs remain in the ledger rather than disappearing from the denominator.

The completed raw artifact set is the matching `run_config__*.json`, `run_summary__*.json`, per-task CSV, and `run_details__*.jsonl`; a summary alone is not evidence. Config and summary are self-hashed, and the completed config binds the details artifact hash, size, and record count. Config and detail records bind experiment/protocol fingerprints, strict-verifier implementation, model/checkpoint artifact, data manifest, benchmark manifest, checkpoint-selection receipt, and source-row locators. Per-row integrity keeps the raw representation `instance_sha256` separate from `semantic_instance_sha256`, `benchmark_row_sha256`, and `reference_output_sha256`. Rescore, deterministic-repair, and any future adaptive-repair records/summaries must add their own self-hashes and a distinct implementation fingerprint while preserving an auditable link to the unchanged raw details.
