# Remote publication launch and immutable source workflow

> **CURRENT HOLD / SUPERSESSION (2026-08-21):** Every base-specialist SFT
> instruction below that refers to the v3 or v7 campaign, four A100s,
> `train_sft500k_base_specialist_a100.slurm`, or
> `submit_sft500k_base_specialists_a100.sh` is superseded and retained only as
> historical context. The current candidate is the fresh, one-H100-per-task
> Jiang-f638 v8 specialist lane (`clasp-jiang-f638-base-specialist-sft-campaign-v8`).
> It matches f638's Qwen2.5-7B **base** specialist topology, pinned software
> stack and Unsloth/legacy-TRL path, exact prompt bytes and EOS behavior, one
> explicit seed-42 data shuffle, 500,000 rows/task, one epoch, 20,000-token
> context, batch 4 with accumulation 4, AdamW-8bit schedule, and rank/alpha-64
> attention+MLP LoRA on one H100. Its declared claim-bearing corrections are a
> clean/decontaminated 500,000-example corpus per task, serialization-safe
> `bias="all"`, fresh `trainer.train()` rather than the broken unconditional
> resume call, and a strict contextual response-only mask rather than the
> released full-example fallback.
>
> The exact local execution surfaces are
> `src/clasp/entrypoints/train_sft_jiang_f638.py`,
> `scripts/freeze_remote_environment.py`,
> `scripts/smoke_jiang_f638_sft_environment_h100.slurm`,
> `scripts/freeze_base_sft_campaign.py`,
> `scripts/run_jiang_f638_sft_in_sealed_environment.py`,
> `scripts/train_sft500k_base_specialist_h100.slurm`,
> `scripts/submit_sft500k_base_specialists_h100.sh`, and
> `scripts/validate_base_sft500k_specialist.py`, with the environment pinned by
> `requirements.jiang-f638-sft.txt`. See
> `docs/jiang_llmcosolver_f638_training_audit_20260821.md` for the exact
> match/correction boundary. **No v8 source snapshot, sealed H100 receipt, or
> frozen campaign authority has yet authorized a run; no job is launched or
> authorized by this notice or by the checked-in scripts.**

The superseded body below was last updated on 2026-08-15. This is an
operational runbook, not evidence of SOTA performance and not authorization to
submit duplicate jobs.

## HOLD: canonical B=9/K=8 A100 campaign

`scripts/train_clasp_sota_a100.slurm` now describes the new matched-objective
training matrix, but this repository update does **not** submit or release any
job. The only runnable method identifiers in a fresh canonical source snapshot
are:

| `SOTA_METHOD` | Credit/controller difference |
|---|---|
| `grpo_z` | `z_G(r_i)` through the same strict constrained evaluator, with every multiplier fixed at zero |
| `dual_scalar_z` | `z_G(r_i-p_i)` with the legacy projected EMA component controller (scalarize, then standardize) |
| `dual_separate_z` | `z_G(r_i)-z_G(p_i)` with the same positive aggregate component penalty and controller (standardize separately) |
| `clasp_canonical` | exact canonical size-eight subset credit, bounded failed-set recovery, and signed failure@8 controller |
| `clasp_canonical_fixed_lambda` | the identical canonical utility with a preregistered frozen `SOTA_CANONICAL_FIXED_LAMBDA` and controller learning rate zero |
| `clasp_canonical_no_recovery` | the canonical estimator/controller with recovery scale exactly zero |
| `set_null` | the same frozen symmetric K-set utility with credit computed against the task's independently constructed and replay-verified fixed-null receipt |

The old `grpo`, `dual_grpo`, `clasp_dense`, and `clasp_exact` identifiers remain
historical labels only and the new launcher rejects them. The exact method ID
`set_null` is now formula-matched and fail-closed on the task's frozen receipt
under `experiments/sota_20260814/fixed_null_empty_v1`; the aliases
`clasp_set_null` and `clasp_fixed_null` remain rejected. The campaign manifest
must bind the approved receipt hash for every task, and every SET-NULL run must
bind the matching task receipt. Legacy Frontier-CDF uses a different
construction and is not this matched `SET-NULL` baseline.

`dual_separate_z` is now a distinct formula-bound trainer branch. It consumes
the existing positive aggregate component penalty
`p_i=sum_j lambda_j w_j m_ij max(0,c_ij-epsilon_j)` and computes exactly
`z_G(r_i)-z_G(p_i)`. It does not sum separately standardized component costs,
and it does not reuse `dual_scalar_z`'s `z_G(r_i-p_i)` transform. The CLI rejects
any attempt to combine this branch with `pd_group_scalarization` or CLASP.

Every runnable arm is hard-bound to rollout `B=9`, deployment `K=8`, global
generation batch 36, effective optimization batch 36, the same data order per
training seed, prompt/completion caps 29,768/3,000, decoder, optimizer, strict
ground-truth reward check, `co_feasibility_lock_all` verifier profile, zero
structure reward, infeasible objective credit disabled, `loss_type=grpo`, and
the explicitly acknowledged clipped token-level practical surrogate. This is
the same practical surrogate across arms; it is not the unclipped unbiased
estimator in the theorem. Feasibility warmup, replay, adaptive length, Pareto
anchor, auxiliary losses, hard-group rescaling, legacy BoN recovery, and
component-dual recovery are all disabled. The shared optimizer LR warmup ratio
remains 0.03; “warmup disabled” here means the CLASP feasibility-credit
replacement, not the common optimizer schedule.

The canonical arms also set every legacy component-dual learning rate and
initial component multiplier to zero. Thus their only operative multiplier is
the task-level failure@8 multiplier. The main and no-recovery arms use set-dual
learning rate 0.05; the fixed-lambda arm uses exactly zero and therefore cannot
change its lambda (its failure EMA may still be logged). The no-recovery arm
changes only `rho` from 1 to 0 and leaves the failure@8 controller active.
`CLASP_BON_SUCCESS_CREDIT` is omitted because that legacy switch is not read by
the canonical branch.

The deployment failure target is frozen at `delta=0.005` (pass@8 target 99.5%)
as a stringent reliability target, not selected from test outcomes. A new
advancement policy must contain `canonical_target_failure_rate: 0.005`; when it
includes the fixed-lambda arm it must also contain an exact
`canonical_fixed_lambda_by_task` map. The launcher compares both fields before
creating any run directory.

All arms now use `clasp.entrypoints.train_grpo` and the same evaluator-backed
reward manager. In particular, `grpo_z` no longer enters the standard
three-reward-head path. That manager always calls the task objective with
`apply_constraint_gate=True`; the launcher also passes
`--no-pd_soft_objective_for_infeasible`, so every infeasible candidate has
`r_i=0`. `CLASP_FEASIBILITY_GATED=0` on `grpo_z` only disables a legacy CLASP
advantage branch and does not affect this evaluator gate. With structure weight,
component multiplier initialization/learning rates, set multiplier, and every
recovery path all zero, the executed coefficient is exactly
`z_G(r_i)=(r_i-mean_G(r))/(std_G(r)+1e-4)`.

For `dual_scalar_z`, let
`p_i=sum_j lambda_j w_j m_ij max(0,c_ij-epsilon_j)`. The executed coefficient is
`z_G(r_i-p_i)`, not `z_G(r_i)-mu z_G(c_i)`. Its legacy controller computes the
masked minibatch gap `g_j=E[m_ij(c_ij-epsilon_j)]`, applies EMA
initialization `e_j=g_j` followed by `e_j <- 0.9 e_j + 0.1 g_j`, and updates
`lambda_j <- clip(lambda_j+0.05 e_j,0,100)`. Under
`co_feasibility_lock_all`, operative feasibility heads have `epsilon_j=0` and
nonnegative costs, so those multipliers are one-sided/monotone. Diagnostic
heads with `epsilon_j=1` receive no positive policy penalty and remain at zero
from the frozen zero initialization. This is a projected component-penalty
baseline, not a signed residual controller.

For `dual_separate_z`, the same controller constructs the same already
aggregated positive penalty `p_i` before standardization. The policy coefficient
is
`(r_i-mean_G(r))/(std_G(r)+1e-4) - (p_i-mean_G(p))/(std_G(p)+1e-4)`.
A zero-spread reward or penalty term contributes exactly zero. Both terms are
group centered, so the resulting coefficient is group centered as well. The
formula is receipted as
`group_z_reward_minus_group_z_positive_component_penalty_v1`; the component
controller and every non-formula setting match `dual_scalar_z`.

For each canonical size-eight subset `S`, the frozen symmetric utility is
`U(S)=max_{i in S, feasible} r_i + rho I[failure] R(S) - lambda I[failure]`,
with failed-set recovery `R` bounded in `[-1,0]`; the policy coefficient is the
exact `K=8` times contain-minus-exclude canonical credit. The main and
no-recovery arms update the sole set multiplier by projected signed failure@8
residual, `lambda <- clip(lambda+0.05*(EMA_failure-0.005),0,100)` (the first
EMA observation is the batch failure rate; later observations use
`0.9*previous+0.1*batch`). The fixed-lambda arm has
`CLASP_BON_DUAL_LR=0`, so it never updates lambda. The no-recovery arm retains
that controller and changes only `rho` from one to zero.

The run contract independently validates the method profile and rejects any
drift in rollout count, deployment count, token surrogate, reward gate,
lengths, batches, or disabled systems components. It records the exact source
tree, data/model registries, tracked research environment, package versions,
and a live identity derived from `SOTA_REMOTE_ENVIRONMENT_RECEIPT`. The launcher
requires `SOTA_SEALED_ENV`, refuses a writable prefix or parent, and invokes
`run_grpo_in_sealed_environment.py` inside the one `srun` step. That wrapper
revalidates the complete sealed prefix and the homogeneous four-A100 identity
immediately before it execs distributed training.

### Remaining blockers before any 4xA100 submission

1. Freeze a **new** immutable source snapshot containing the canonical trainer,
   launcher, sealed-step wrapper, contract validator, and tests. No existing v1,
   v2, or historical v3 snapshot contains this complete method.
2. Produce a real read-only remote-environment receipt for the exact sealed
   interpreter and verify it on an allocated homogeneous 4xA100 node. There is
   still no such checked-in receipt; version strings from a login shell are not
   a substitute.
3. Finish and validate all five new non-instruction-base specialist SFTs, then
   freeze the five-task model-initialization registry and its exact base/SFT
   checkpoint receipts. Historical Instruct specialists remain ineligible.
4. Freeze the complete forbidden-source/data registry and a fresh advancement
   policy whose `expected_methods` are exactly the six identifiers above,
   whose `contract_requirements.num_generations` is 9, whose
   `canonical_target_failure_rate` is 0.005, and whose
   `canonical_fixed_lambda_by_task` contains exactly the five active tasks. Its source snapshot,
   artifact root, seeds, stage index, prior receipt chain, save/eval cadence,
   and model-registry hash must match the launch exactly.
5. Select and preregister the fixed lambda using validation data only. The
   fixed-lambda arm fails closed when `SOTA_CANONICAL_FIXED_LAMBDA` is absent;
   lambda zero is permitted only when it is the explicit frozen choice.
6. Run a held smoke/preflight that demonstrates B=9 memory headroom at the
   29,768/3,000 caps, exact save/resume fidelity for the canonical controller,
   and nonzero finite canonical credit. A source audit alone cannot establish
   A100 memory safety.
7. Update and test the confirmation-evaluation, checkpoint-selection, and
   publication-aggregation method allowlists/config masks for the new method
   identifiers before opening confirmatory data. The training launcher and
   advancement-policy validator recognize the matrix; the older downstream
   evaluation surface still names only historical arms and is therefore a
   validation blocker.
8. Implement the same-utility fixed-null trainer if `SET-NULL` is required by
   the final attribution table. Until then it is an explicitly missing
   ablation, not a Frontier-CDF result.
Keep every job held while any item above is unresolved. This runbook contains
no `sbatch` command for the new matrix and authorizes no scheduler mutation.

> **Superseded launch warning (17:59 UTC):** the base-SFT v1 source/campaign uses
> PEFT `bias=lora_only`, whose exporter omits the trained Q/K/V biases in both
> the remote PEFT 0.18.0 and local PEFT 0.19.1 environments.  Do not release or
> reuse v1 for publication.  Follow
> `docs/lora_bias_persistence_remediation_20260814.md`
> only as historical remediation context, then freeze a fresh v3 source and
> campaign using `bias=all`, exact checkpoint/resume fidelity, and the sealed
> five-run authority described below.

## Current launch gate

- TSP and CVRP historical 500k specialist SFT artifacts are finalized at step
  31,250 and contain root adapter weights, but both resolve to
  `Qwen2.5-7B-Instruct`. They do **not** satisfy the new non-instruction base
  contract.
- OP, MIS, and JSSP historical specialist directories do not yet exist. Their
  first SFT jobs (`873904`, `873906`, `873908`), continuations, and validation
  dependencies are pending. Their stored worker also defaults to
  `Qwen2.5-7B-Instruct`, so their eventual outputs remain in the historical
  Instruct lane; do not alter, cancel, or repurpose those jobs.
- A complete non-instruction base artifact exists at
  `/lustre/fsn1/projects/rech/vas/uag72kz/jiang_reproduction/models/unsloth_Qwen2.5-7B_86e6e1ec`.
  Its Qwen architecture, absence of a chat template, pinned upstream revision,
  four safetensor shards, index, config, and tokenizer hashes were independently
  rechecked against `model_receipt.json`. Its current tree is owner-writable,
  however, so publication use requires a verified read-only copy at a new
  canonical path.
- Historical source snapshots v1 and v2 must remain unchanged. v2 predates the
  current integrity gates and is not a launch source for current results.
- No v3 source snapshot, frozen 15-tree forbidden-source registry, frozen
  base-SFT campaign, base-specialist output root, or model-initialization
  registry currently exists remotely.
- Jobs `965221` through `965228` are already pending for the old v1 TSP/CVRP
  40-step smoke. Never silently submit the same task/method/seed tuples while
  those jobs are pending or running. Their outputs identify v1 code and cannot
  validate the corrected v3 implementation.
- The reference-solver/benchmark gate remains blocked. A successful training
  smoke advances development only; it cannot authorize confirmation inference
  or a SOTA claim.

The publication launcher now requires a new explicit artifact root, a verified
read-only `SOURCE_MANIFEST.sha256`, a self-hashed forbidden-source registry, and
a self-hashed five-task model-initialization registry. The advancement policy
must contain the exact `model_initialization_registry_sha256`; the launch must
also provide the exact approved non-instruction base path. The launcher verifies
that the actual base, task specialist, SFT tree, and RL train tree equal their
frozen receipts. Its immutable run contract hashes every source-snapshot file
except the manifest itself and the writable Slurm `logs/` directory. It retains
every scheduled checkpoint and permits clean job boundaries only at steps that
are both saved and evaluated.

For every `beta>0` cell, the launcher uses the approved `bias=all` specialist
adapter itself for policy initialization and for the separately loaded frozen
reference. `CLASP_KL_REFERENCE_MODEL_PATH` must equal that adapter and
`CLASP_KL_REFERENCE_RECEIPT` must equal its internal
`checkpoint_fidelity.json`; merged-model reference receipts are not accepted.
The post-Trainer `exact_peft_kl_reference.json` must then exist byte-identically
at the run root and every retained checkpoint. Advancement, development-evidence,
selection, confirmation, and aggregation reopen it and reject any source,
verifier, beta/dtype/world-size, tensor, functional, rank-consensus, self-hash,
or file-hash mismatch. A zero-beta cell must contain no such receipt.

## Create a fresh base-SFT source snapshot without touching v1 or v2

Run the following from the repository root on the local workstation. Choose a
unique staging name. Every `test ! -e` is deliberate; do not replace it with a
delete or overwrite operation.

```bash
set -euo pipefail
REMOTE_HOST=uag72kz@jean-zay.idris.fr
SNAPSHOT_ID=correct_clasp_base_sft_20260815_v3
REMOTE_PARENT=/lustre/fsn1/projects/rech/vas/uag72kz
REMOTE_STAGE="${REMOTE_PARENT}/.${SNAPSHOT_ID}.staging_$(date -u +%Y%m%dT%H%M%SZ)_$$"
REMOTE_FINAL="${REMOTE_PARENT}/${SNAPSHOT_ID}"
REMOTE_RECEIPT="${REMOTE_PARENT}/${SNAPSHOT_ID}.receipt.json"

ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "test ! -e '${REMOTE_STAGE}' && test ! -e '${REMOTE_FINAL}' && test ! -e '${REMOTE_RECEIPT}' && mkdir -m 2750 '${REMOTE_STAGE}'"

rsync -avP \
  --exclude='__pycache__/' --exclude='*.py[co]' \
  README.md pyproject.toml src scripts tests docs \
  "${REMOTE_HOST}:${REMOTE_STAGE}/"

# Transfer only the freezer's exact static experiments/ allowlist. Never transfer
# the full experiments tree: it contains local models and runtime evidence.
STATIC_EXPERIMENT_FILES=(
  experiments/sota_20260814/PROTOCOL.md
  experiments/sota_20260814/campaign_manifest.template.json
  experiments/sota_20260814/checkpoint_selection_policy.template.json
  experiments/sota_20260814/forbidden_source_registry.template.json
  experiments/sota_20260814/iid_generator_contract.json
  experiments/sota_20260814/lane_contracts.json
  experiments/sota_20260814/llmcosolver_f638_public_data_manifest.json
  experiments/sota_20260814/llmcosolver_f638_public_data_manifest_v2.json
  experiments/sota_20260814/llmcosolver_f638_semantic_contract.json
  experiments/sota_20260814/llmcosolver_f638_semantic_contract_v2.json
  experiments/sota_20260814/llmcosolver_f638_stack_probe.json
  experiments/sota_20260814/reference_solver_contract.template.json
  experiments/sota_20260814/targets.json
  experiments/sota_20260814/tsp_fcdf_beta_pilot.template.json
)
for source_file in "${STATIC_EXPERIMENT_FILES[@]}"; do
  test -f "${source_file}"
done
rsync -avP --relative "${STATIC_EXPERIMENT_FILES[@]}" \
  "${REMOTE_HOST}:${REMOTE_STAGE}/"

ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "/linkhome/rech/genlam01/uag72kz/.conda/envs/new_unsloth_env/bin/python \
   '${REMOTE_STAGE}/scripts/freeze_source_snapshot.py' \
   --staging-root '${REMOTE_STAGE}' --final-root '${REMOTE_FINAL}' \
   --receipt '${REMOTE_RECEIPT}' --snapshot-id '${SNAPSHOT_ID}'"
```

The freezer refuses an existing destination/receipt, links, special files,
caches, artifact-directory components at any depth, nested model weights or
adapters, checkpoints, logs, raw generations, repair/runtime outputs, any
non-allowlisted `experiments/` entry, an incomplete launch surface, a nonempty
root log directory, or a changing file. It emits a self-hashed receipt, sets
source files to `0440`, source directories to `0550`, the root to `2550`, and
leaves only the root `logs/` writable at `0770` before atomic publication.

Verify the remote result with manifest, mode, and read-only import checks:

```bash
ssh -o BatchMode=yes "${REMOTE_HOST}" "
  set -eu
  cd '${REMOTE_FINAL}'
  sha256sum -c SOURCE_MANIFEST.sha256
  test -z \"\$(find . -path ./logs -prune -o -perm /0222 -print)\"
  test \"\$(stat -c %a .)\" = 2550
  test \"\$(stat -c %a logs)\" = 770
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${REMOTE_FINAL}/src' \
    /linkhome/rech/genlam01/uag72kz/.conda/envs/new_unsloth_env/bin/python -c \
    'import clasp.entrypoints.train_grpo, clasp.entrypoints.train_grpo_standard'
"
```

After the read-only block succeeds, copy the create-once remote receipt into
the local evidence directory without overwriting anything:

```bash
test ! -e experiments/sota_20260814/source_snapshot_base_sft_v3.json
rsync -avP "${REMOTE_HOST}:${REMOTE_RECEIPT}" \
  experiments/sota_20260814/source_snapshot_base_sft_v3.json
```

## HOLD: freeze the exact remote environment before the base-SFT campaign

The existing mutable `new_unsloth_env` is not itself a publication identity.
Do not freeze a campaign against package version strings copied from a login
shell. An operator must first create a dedicated environment at a never-used
path, verify/install the final stack, remove runtime caches, and seal every
prefix entry plus the prefix's direct parent against writes. Creating/copying
that environment, changing permissions, and obtaining the required four-A100
allocation are administrative actions and require operator authorization; this
runbook does not authorize or perform them.

Only after the new source snapshot and sealed environment both exist, load
exactly `arch/a100` and `miniforge/24.9.0`, activate that environment by its
absolute prefix, set `PATH`, `PYTHONPATH`, and `PYTHONNOUSERSITE` exactly as in
the base-SFT block of `experiments/sota_20260814/PROTOCOL.md`, and run from an
authorized four-A100 allocation:

```bash
set -euo pipefail
SOURCE_ROOT=/path/to/new/immutable/source-snapshot
SEALED_ENV=/path/to/new/read-only-parent/publication-sft-environment
SEALED_PYTHON="$(readlink -f "${SEALED_ENV}/bin/python")"
ENV_RECEIPT_DIR=/path/to/new/sealed-remote-environment-v1
REMOTE_ENVIRONMENT_RECEIPT="${ENV_RECEIPT_DIR}/remote_environment.json"

module purge
module load arch/a100
module load miniforge/24.9.0
conda activate "${SEALED_ENV}"
export PATH="${SEALED_ENV}/bin:/usr/bin:/bin"
export PYTHONPATH="${SOURCE_ROOT}/src:${SOURCE_ROOT}/scripts"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS
while IFS='=' read -r inherited_name _; do
  case "${inherited_name}" in
    CLASP_*|PD_*|GENERATION_*|GRPO_*|PYTORCH_*|HF_*|TRANSFORMERS_*|TOKENIZERS_*|BITSANDBYTES_*|UNSLOTH_*|ACCELERATE_*|WANDB_*|NCCL_*) unset "${inherited_name}" ;;
  esac
done < <(env)
test ! -e "${ENV_RECEIPT_DIR}"
test ! -w "${SEALED_ENV}"
test ! -w "$(dirname "${SEALED_ENV}")"
test "${SLURM_GPUS_ON_NODE:-0}" = 4
"${SEALED_PYTHON}" "${SOURCE_ROOT}/scripts/freeze_remote_environment.py" \
  --output "${REMOTE_ENVIRONMENT_RECEIPT}"
```

The producer itself rechecks the complete prefix tree, all visible installed
distributions, Python/Torch/native build identity, loader/import paths, producer
bytes, and four homogeneous A100 compute-capability-8.0 devices. It rejects a
writable prefix or parent, broken symlinks, symlinks that escape the sealed
prefix, editable/system-site escapes, missing/duplicate packages, or a
non-A100/mixed allocation. Strictly internal regular-directory symlinks are
accepted and their resolved target trees are hash-covered. It creates
the previously absent receipt directory, writes one self-hashed `0440` receipt,
and seals the directory `0550`; it will not reuse an existing directory.

No real remote-environment receipt or replacement base-SFT v7 campaign exists in
this repository. Keep all launch templates on **HOLD** until the operator has
frozen the real receipt. Pass its canonical path to both
`freeze_base_sft_campaign.py --remote-environment-receipt` and the submitter as
`BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT`; pass the sealed interpreter as
`BASE_SFT_PYTHON`. The campaign freezer and submitter rehash the non-accelerator
identity. Every batch allocation reopens the live package/runtime/A100 identity
before touching a run, inside the same `srun` step before training, and before
finalization-side receipt writes.

Before any SFT submission, create the complete forbidden-source registry and
the independent 20-file reserved-source authority, publish a verified read-only
copy of the audited non-instruction base, execute all five clean-corpus build
commands in `PROTOCOL.md`, run the all-row token audit below while the SFT data
root is still writable, mode-seal the complete input trust root, and run
`freeze_base_sft_campaign.py` exactly as documented in `PROTOCOL.md`. Then use only
`submit_sft500k_base_specialists_a100.sh`; it fixes the complete five-task matrix and
creates a new output root and immutable submission ledger. The historical
`submit_sft500k_specialists_a100.sh` remains the Instruct-lane controller and must not be
used for the base campaign.

```bash
set -euo pipefail
BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT=/lustre/fsn1/projects/rech/vas/uag72kz/clasp_base_sft500k_clean_20260821_anchor_v1/inputs
BASE_SFT_BASE_MODEL_PATH="${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT}/Qwen2.5-7B-base_86e6e1ec"
BASE_SFT_DATA_ROOT="${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT}/data_sft"
BASE_SFT_FORBIDDEN_SOURCE_REGISTRY="${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT}/forbidden-source-registry.json"
BASE_SFT_CLEAN_TASK_RECEIPT_DIR="${BASE_SFT_DATA_ROOT}/cleanliness_receipts"
BASE_SFT_RESERVED_SOURCE_ROOT="${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT}/reserved_sources_v1"
BASE_SFT_RESERVED_SFT_EVAL_CVRP="${BASE_SFT_RESERVED_SOURCE_ROOT}/sft_eval/cvrp/test.json"
BASE_SFT_RESERVED_SFT_EVAL_JSSP="${BASE_SFT_RESERVED_SOURCE_ROOT}/sft_eval/jssp/eval.json"
BASE_SFT_RESERVED_SFT_EVAL_MIS="${BASE_SFT_RESERVED_SOURCE_ROOT}/sft_eval/mis/test.json"
BASE_SFT_RESERVED_SFT_EVAL_OP="${BASE_SFT_RESERVED_SOURCE_ROOT}/sft_eval/op/eval.json"
BASE_SFT_RESERVED_SFT_EVAL_TSP="${BASE_SFT_RESERVED_SOURCE_ROOT}/sft_eval/tsp/test.json"
BASE_SFT_RESERVED_RL_EVAL_CVRP="${BASE_SFT_RESERVED_SOURCE_ROOT}/rl_eval/cvrp/test.json"
BASE_SFT_RESERVED_RL_EVAL_JSSP="${BASE_SFT_RESERVED_SOURCE_ROOT}/rl_eval/jssp/test.json"
BASE_SFT_RESERVED_RL_EVAL_MIS="${BASE_SFT_RESERVED_SOURCE_ROOT}/rl_eval/mis/test.json"
BASE_SFT_RESERVED_RL_EVAL_OP="${BASE_SFT_RESERVED_SOURCE_ROOT}/rl_eval/op/test.json"
BASE_SFT_RESERVED_RL_EVAL_TSP="${BASE_SFT_RESERVED_SOURCE_ROOT}/rl_eval/tsp/test.json"
BASE_SFT_RESERVED_RL_TRAIN_CVRP="${BASE_SFT_RESERVED_SOURCE_ROOT}/rl_train/cvrp/train_rl.json"
BASE_SFT_RESERVED_RL_TRAIN_JSSP="${BASE_SFT_RESERVED_SOURCE_ROOT}/rl_train/jssp/train_rl.json"
BASE_SFT_RESERVED_RL_TRAIN_MIS="${BASE_SFT_RESERVED_SOURCE_ROOT}/rl_train/mis/train_rl.json"
BASE_SFT_RESERVED_RL_TRAIN_OP="${BASE_SFT_RESERVED_SOURCE_ROOT}/rl_train/op/train_rl.json"
BASE_SFT_RESERVED_RL_TRAIN_TSP="${BASE_SFT_RESERVED_SOURCE_ROOT}/rl_train/tsp/train_rl.json"
BASE_SFT_RESERVED_IID_CONFIRMATION_CVRP="${BASE_SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/cvrp/test.json"
BASE_SFT_RESERVED_IID_CONFIRMATION_JSSP="${BASE_SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/jssp/test.json"
BASE_SFT_RESERVED_IID_CONFIRMATION_MIS="${BASE_SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/mis/test.json"
BASE_SFT_RESERVED_IID_CONFIRMATION_OP="${BASE_SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/op/test.json"
BASE_SFT_RESERVED_IID_CONFIRMATION_TSP="${BASE_SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/tsp/test.json"
test ! -e "${BASE_SFT_DATA_ROOT}/sft_cleanliness_audit.json"
"${BASE_SFT_PYTHON}" "${CLASP_PROJECT_ROOT}/scripts/build_clean_sft500k_corpus.py" \
  --data-root "${BASE_SFT_DATA_ROOT}" \
  --merge-task-receipt "${BASE_SFT_CLEAN_TASK_RECEIPT_DIR}/cvrp.json" \
  --merge-task-receipt "${BASE_SFT_CLEAN_TASK_RECEIPT_DIR}/jssp.json" \
  --merge-task-receipt "${BASE_SFT_CLEAN_TASK_RECEIPT_DIR}/mis.json" \
  --merge-task-receipt "${BASE_SFT_CLEAN_TASK_RECEIPT_DIR}/op.json" \
  --merge-task-receipt "${BASE_SFT_CLEAN_TASK_RECEIPT_DIR}/tsp.json" \
  --receipt "${BASE_SFT_DATA_ROOT}/sft_cleanliness_audit.json"
test ! -w "${BASE_SFT_DATA_ROOT}/sft_cleanliness_audit.json"

BASE_SFT_RESERVED_SOURCE_REGISTRY="${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT}/sft_reserved_source_registry.json"
BASE_SFT_IID_GENERATION_RECEIPT="${BASE_SFT_RESERVED_SOURCE_ROOT}/iid_confirmation/generation_receipt.json"
test ! -e "${BASE_SFT_RESERVED_SOURCE_REGISTRY}"
reserved_source_args=()
for task in cvrp jssp mis op tsp; do
  for source_class in sft_eval rl_eval rl_train iid_confirmation; do
    variable="BASE_SFT_RESERVED_${source_class^^}_${task^^}"
    source_path="${!variable:?set ${variable} to its canonical reserved JSON file}"
    test -f "${source_path}" && test ! -w "${source_path}"
    reserved_source_args+=(--source "${task},${source_class},${source_path}")
  done
done
test ! -w "${BASE_SFT_IID_GENERATION_RECEIPT}"
"${BASE_SFT_PYTHON}" "${CLASP_PROJECT_ROOT}/scripts/freeze_sft_reserved_source_registry.py" \
  --immutable-input-trust-root "${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT}" \
  "${reserved_source_args[@]}" \
  --iid-generation-receipt "${BASE_SFT_IID_GENERATION_RECEIPT}" \
  --output "${BASE_SFT_RESERVED_SOURCE_REGISTRY}"
test ! -w "${BASE_SFT_RESERVED_SOURCE_REGISTRY}"

test ! -e "${BASE_SFT_FORBIDDEN_SOURCE_REGISTRY}"
forbidden_source_args=()
for task in cvrp jssp mis op tsp; do
  rl_train_variable="BASE_SFT_RESERVED_RL_TRAIN_${task^^}"
  rl_eval_variable="BASE_SFT_RESERVED_RL_EVAL_${task^^}"
  forbidden_source_args+=(
    --source "${task},sft-train,${BASE_SFT_DATA_ROOT}/${task}/train"
    --source "${task},rl-train,${!rl_train_variable}"
    --source "${task},historical-eval,${!rl_eval_variable}"
  )
done
"${BASE_SFT_PYTHON}" "${CLASP_PROJECT_ROOT}/scripts/freeze_forbidden_source_registry.py" \
  "${forbidden_source_args[@]}" \
  --base-sft-post-iid-generation-receipt "${BASE_SFT_IID_GENERATION_RECEIPT}" \
  --output "${BASE_SFT_FORBIDDEN_SOURCE_REGISTRY}"
test ! -w "${BASE_SFT_FORBIDDEN_SOURCE_REGISTRY}"

test ! -e "${BASE_SFT_DATA_ROOT}/sft_token_length_audit.json"
: "${BASE_SFT_TOKEN_AUDIT_SHARD_DIR:?set a fresh canonical shard directory}"
test ! -e "${BASE_SFT_TOKEN_AUDIT_SHARD_DIR}"
mkdir -m 0750 "${BASE_SFT_TOKEN_AUDIT_SHARD_DIR}"
token_audit_pids=()
for task in cvrp jssp mis op tsp; do
  "${BASE_SFT_PYTHON}" "${CLASP_PROJECT_ROOT}/scripts/audit_sft_token_lengths.py" \
    --data-root "${BASE_SFT_DATA_ROOT}" \
    --tokenizer-path "${BASE_SFT_BASE_MODEL_PATH}" \
    --forbidden-source-registry "${BASE_SFT_FORBIDDEN_SOURCE_REGISTRY}" \
    --batch-size 256 --max-seq-length 20000 \
    --expected-rows-per-task 500000 --task "${task}" \
    --output "${BASE_SFT_TOKEN_AUDIT_SHARD_DIR}/${task}.json" &
  token_audit_pids+=("$!")
done
for token_audit_pid in "${token_audit_pids[@]}"; do
  wait "${token_audit_pid}"
done
"${BASE_SFT_PYTHON}" "${CLASP_PROJECT_ROOT}/scripts/audit_sft_token_lengths.py" \
  --data-root "${BASE_SFT_DATA_ROOT}" \
  --tokenizer-path "${BASE_SFT_BASE_MODEL_PATH}" \
  --forbidden-source-registry "${BASE_SFT_FORBIDDEN_SOURCE_REGISTRY}" \
  --merge-shard-dir "${BASE_SFT_TOKEN_AUDIT_SHARD_DIR}" \
  --max-seq-length 20000 --expected-rows-per-task 500000 \
  --output "${BASE_SFT_DATA_ROOT}/sft_token_length_audit.json"
test ! -w "${BASE_SFT_DATA_ROOT}/sft_token_length_audit.json"
```

After the owner mode-seals the complete input trust root and its parent, freeze
v7 and submit with the same explicit reservation authority. This mode seal is
not administrative immutability; every launch repeats path/inode/mode/content
checks, and an administrator-owned or ACL-protected anchor would be stronger:

```bash
set -euo pipefail
: "${BASE_SFT_RUN_CONTRACT_AUTHORITY_ROOT:?set a fresh authority root}"
: "${BASE_SFT_OUTPUT_ROOT:?set a fresh specialist output root}"
BASE_SFT_CAMPAIGN_CONTRACT="${BASE_SFT_RUN_CONTRACT_AUTHORITY_ROOT}/base_sft_campaign.json"
test ! -e "${BASE_SFT_RUN_CONTRACT_AUTHORITY_ROOT}"
test ! -e "${BASE_SFT_OUTPUT_ROOT}"
test ! -w "${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT}"
test ! -w "$(dirname "${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT}")"
"${BASE_SFT_PYTHON}" "${CLASP_PROJECT_ROOT}/scripts/freeze_base_sft_campaign.py" \
  --campaign-id qwen25-base-specialists-500k-v7 \
  --run-contract-authority-root "${BASE_SFT_RUN_CONTRACT_AUTHORITY_ROOT}" \
  --immutable-input-trust-root "${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT}" \
  --source-root "${CLASP_PROJECT_ROOT}" \
  --base-model-path "${BASE_SFT_BASE_MODEL_PATH}" \
  --forbidden-source-registry "${BASE_SFT_FORBIDDEN_SOURCE_REGISTRY}" \
  --reserved-source-registry "${BASE_SFT_RESERVED_SOURCE_REGISTRY}" \
  --remote-environment-receipt "${BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT}" \
  --sft-data-root "${BASE_SFT_DATA_ROOT}" \
  --output-root "${BASE_SFT_OUTPUT_ROOT}" \
  --output "${BASE_SFT_CAMPAIGN_CONTRACT}"

CLASP_PROJECT_ROOT="${CLASP_PROJECT_ROOT}" \
BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT="${BASE_SFT_IMMUTABLE_INPUT_TRUST_ROOT}" \
BASE_SFT_CAMPAIGN_CONTRACT="${BASE_SFT_CAMPAIGN_CONTRACT}" \
BASE_SFT_FORBIDDEN_SOURCE_REGISTRY="${BASE_SFT_FORBIDDEN_SOURCE_REGISTRY}" \
BASE_SFT_RESERVED_SOURCE_REGISTRY="${BASE_SFT_RESERVED_SOURCE_REGISTRY}" \
BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT="${BASE_SFT_REMOTE_ENVIRONMENT_RECEIPT}" \
BASE_SFT_PYTHON="${BASE_SFT_PYTHON}" \
BASE_SFT_BASE_MODEL_PATH="${BASE_SFT_BASE_MODEL_PATH}" \
BASE_SFT_DATA_ROOT="${BASE_SFT_DATA_ROOT}" \
BASE_SFT_OUTPUT_ROOT="${BASE_SFT_OUTPUT_ROOT}" \
  bash "${CLASP_PROJECT_ROOT}/scripts/submit_sft500k_base_specialists_a100.sh"
```

The merged cleanliness receipt is a mandatory v7 gate: each task must contain
500,000 unique demonstrations, no reserved-evaluation collision, and only
strictly valid labels, and its live output hash must equal the registered SFT
file. The subsequent token pass is a streaming audit over all five 500k
sources, not a sampled prompt check. Each row is reduced to a
source/tokenizer-bound scalar length and
alignment digest rather than a serialized token-ID vector. The task shards may
run concurrently; the merge reopens all source, tokenizer, producer, and shard
hashes and does not retokenize. The campaign freezer and submitter's campaign validation both fail if
the receipt is absent, writable, attached to different tokenizer/data bytes,
or reports any truncation/boundary failure.

The canonical freezer invocation must include
`--immutable-input-trust-root`, `--run-contract-authority-root`, and
`--reserved-source-registry`; its `--output`
is exactly `<authority-root>/base_sft_campaign.json`. Do not put that authority,
the submission ledger, or results below the mutable output root.

The current freezer emits `clasp-base-specialist-sft-campaign-v7`. It requires
one explicit immutable-input trust root containing the source snapshot, base
model, forbidden-source registry, reserved-source registry/tree, and SFT data.
The clean audit's four labeled reservation records per task must equal the
20-file authority byte-for-byte; RL train/eval are also cross-bound to the
ordinary registry. IID confirmation is bound by file hash and row count to the
retained schema-v1 generation receipt. That receipt has no pre-IID forbidden
registry binding, so the ordinary registry uses the explicit
`clasp-base-sft-post-iid-source-registry-v1` protocol and records this limitation
instead of claiming it was frozen before IID generation. Every path component from that
anchor to each input is reopened without following links, must be non-writable,
and is bound by type, mode, device, inode, ownership, link count, and a
self-hash. The current owner-controlled parent must be non-writable during
freeze and launch but can later be chmoded or renamed by its owner; the worker
therefore repeats the check inside the same `srun` allocation. The worker refuses `lora_only`, checks
every checkpoint and finalized root for 84 serialized Q/K/V biases plus 392
LoRA tensors, and creates `adapter_persistence_receipt.json`. A v1 or v2
campaign contract cannot be upgraded in place.

After the terminal save, the base worker also runs
`audit_sft_reload_equivalence.py` in a fresh process. It writes the create-once
read-only `sft_reload_equivalence.json` only after a fresh trainable PEFT load
matches the pre-save trainable closure exactly and two independent loads match
bitwise in full state and fixed-probe bfloat16 CUDA distributions. The final
validator and the all-five-task model-initialization registry require this
receipt; an adapter that merely passes `adapter_config.json` and tensor-count
checks cannot be admitted.

Before submission the campaign freezer atomically publishes one dedicated,
previously absent authority directory containing the campaign plus all five
task run contracts and a self-hashed authority receipt. Files are `0440`, the
directory is `0550`, its parent/root inode identities are bound, and every
reader performs descriptor-relative no-follow reads and an exact entry-set
rehash. The mutable run-directory contract is a distinct-inode byte copy; it
never aliases the authority with a hard link. Keep the submission ledger and
all outputs outside this authority.

Publication v3 retains every scheduled 500-step checkpoint through the 31,250
step horizon (`save_total_limit=64`). Each completed checkpoint is sealed
`0440`/`0550` only after checkpoint fidelity, trainer state, optimizer,
scheduler, and all four rank RNG files are durable. Its create-once
`base_sft_resume_fidelity.json` binds those exact bytes, global step, world
size, campaign, authoritative run contract, and remote environment. Resume
selects the newest checkpoint without fallback, retains its directory file
descriptor, passes `/proc/self/fd/<n>` to Trainer, verifies RNG immediately
after Trainer restores it, and audits the loaded adapter closure,
optimizer/scheduler state, and trajectory-critical Trainer state before the
first resumed update. The submitter requires at least 1 TiB free for the full
matrix; every allocation rechecks a 200 GiB per-cell floor.

## Retired v3 40-step launch materials

The checked-in `advancement_policy_40_step_v3.json` and its former submission
commands predate the mandatory five-task model-initialization registry. They are
historical evidence only: they do not contain
`model_initialization_registry_sha256` and cannot authorize the current
launcher. The executable submission block is intentionally omitted so that an
operator cannot mistake stale smoke assets for a publication launch.

Do not edit or re-hash that policy in place. After all five **new base-specialist** SFT
artifacts exist, create the initialization registry as described in
`docs/model_initialization_registry.md`, create a new immutable source snapshot,
and freeze a new advancement policy whose top-level registry hash exactly equals
the new receipt. The new policy must declare `stage_index` and the complete ordered
`prior_stage_receipt_chain`. Supply the corresponding canonical absolute receipt paths
to the launcher as a JSON array in `SOTA_PRIOR_STAGE_RECEIPTS_JSON` (use `[]` only for
stage zero); the launcher reopens and validates every immutable `ADVANCE` receipt before
creating a run contract. Only then construct a fresh campaign command with explicit
`SOTA_MODEL_INITIALIZATION_REGISTRY` and `SOTA_BASE_MODEL_PATH` values. Preserve
every submitted tuple and scheduler job ID in an append-only ledger; partial or
missing matrices remain failures.

For the later 3,200-step selection stage, the launch contract defaults to
400-step chunks/checkpoints, 100-step evaluation, and retention of all eight
checkpoints. Freeze the final method/checkpoint/seed selection policy before
opening checkpoint development results. Do not launch the long matrix until
all five matched SFT validators are complete and the 400-step pilot selects the
strongest baseline. Every later launcher policy must use the same official
schema plus explicit `source_snapshot_id` and `artifact_root` fields; its
`campaign_id`, `target_step`, `contract_requirements`, and expected grid must
match the submitted environment exactly. `chunk_steps` and `save_total_limit`
are also mandatory; ad hoc stop-step overrides are not supported.
