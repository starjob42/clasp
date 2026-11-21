# TSP Frontier-CDF x SFT-anchor pilot (development evidence only)

## Claim boundary

This is a preregistered **development-only** 2x2 pilot. It cannot support a
publication, confirmation, benchmark, or state-of-the-art claim. Its legacy
initialization receipt proves fidelity to the *deployable saved policy*, not to
the historical terminal in-memory SFT policy: the latter's trained Q/K/V biases
were omitted and are unrecoverable. Every contract therefore fixes
`terminal_sft_policy_fidelity=false`, `publication_eligible=false`, and
`sota_eligible=false`.

There is a second precision and parameterization boundary. A trusted relocation
proof compares the converted and relocated saved policy in FP32 without
merging. Every trajectory then loads that relocated PEFT policy in BF16 and
continues its existing nonzero SFT LoRA A/B tensors and saved base-layer biases.
This is a common runtime transformation, not exact equivalence to the FP32
source policy. The exact frozen fields are
`relocation_verification_dtype=float32`, `training_load_dtype=bfloat16`, and
`common_bfloat16_load_not_exact_fp32_source_policy_equivalence=true`. No merged
policy/reference or fresh RL LoRA is used. Exact-FP32-source-policy,
historical-terminal-SFT, publication, and SOTA claims are explicitly forbidden.

The pilot may choose a hypothesis for a later, independent experiment. It may
not choose a reported publication result.

## Question and rationale

The corrected `dense_rank` implementation provides dense nonzero credit, but
its successful-set quality term is a tie-aware rank. The distance between two
objective values does not affect the quality credit after their ordering is
fixed. That variance-reducing surrogate can therefore spend equal policy
pressure on a tiny and a large objective improvement.

This is a material hypothesis, not a post-hoc story. FALCON's primary text
([arXiv:2602.01090, section 3.4](https://arxiv.org/abs/2602.01090)) pairs the
best feasible sample against every inferior feasible sample and weights each
log-sigmoid preference term by its objective difference normalized by the mean
difference. It explicitly motivates making a 50% objective gap drive more than
a 1% gap and argues that ordinary GRPO is dominated by the best sample. BOPO
similarly studies best-anchored preference construction and
objective-difference-scaled gradients
([ICML 2025 PMLR paper](https://proceedings.mlr.press/v267/liao25a.html)). These
are related mechanisms only. They are neither matched CLASP benchmarks nor
evidence that the Frontier-CDF surrogate will win.

The F arm is also not a reproduction of BOPO or FALCON. It is the current
target-aware, fixed-K Frontier-CDF set-credit implementation plus relative
failed-group recovery. A direct best-vs-inferior objective-difference-weighted
auxiliary requires distributed full-group gradient support and belongs in a
separate later arm.

Consequently, `F-D` is a target-aware set-credit *bundle* contrast, not a pure
"magnitude awareness only" ablation: Frontier-CDF also changes target shaping,
noisy-OR set aggregation, centering, and failed-group recovery. The factorial
isolates that bundle from beta anchoring, not its internal components.

## Frozen 2x2 design

All four cells directly continue the same immutable relocated `bias=all` PEFT
adapter in BF16 and use training seeds 42, 43, and 44. The relocation changes
only the absolute base-model path and proves the relocated saved policy equal to
the converted saved policy in FP32. `train_grpo` does not merge it and does not
attach a fresh RL LoRA. This continued-LoRA geometry is scientifically distinct
from training a zero-initialized RL adapter around merged SFT weights: SFT LoRA
A/B are already nonzero, and the saved Q/K/V base-layer biases remain trainable.

| Cell | Successful-set credit | Failed-group recovery | GRPO beta |
| --- | --- | --- | ---: |
| D0 | `dense_rank` | current dense relative recovery | 0 |
| D5 | `dense_rank` | current dense relative recovery | 0.05 |
| F0 | Frontier-CDF | relative failed recovery | 0 |
| F5 | Frontier-CDF | relative failed recovery | 0.05 |

There are 12 trajectories. Every trajectory uses TSP only, K=8, the same
frozen 3,200-row RL source, optimizer and linear schedule, and exactly 400
steps. The prespecified training microbatch is one with 32 gradient-
accumulation steps, preserving effective batch 32 while reducing peak policy
activation memory; generation batch and K remain eight. Logging is every step;
save and training-time evaluation occur only at step 400. No intermediate
checkpoint can be selected and no metric can stop training early. A 3,200-step
continuation must restart at step zero under a separate contract; extending a
400-step linear schedule would compare a different learning-rate trajectory.

The Frontier-CDF settings are fixed:

- raw target gap `tsp:0.0107`;
- initial/final quality temperature 0.5/0.15, annealed over 0.6 of training;
- violation temperature 0.15;
- objective-score scale 2 and gap slope 3;
- recovery and relative-recovery scales 1;
- feasibility-violation tolerance `1e-8`;
- group failure target 0.005.

The 1.07% target is the raw model-output objective-gap target. It is not the
0.95% deterministic-repair target, and repaired results are not the primary
endpoint.

The target is retained without outcome-driven tuning, but it is already loose
relative to the sealed SFT best-of-8 baseline. On the exact 200-row development
slice, SFT has mean best-of-8 gap 0.008826100659, which is 82.4869% of the
0.0107 target and leaves only 0.001873899341 absolute headroom. Best-of-8 meets
the target on 128/200 rows (64%), including 35 exact-zero clipped gaps. This is
material set-level saturation. It is not complete candidate-level saturation:
only 455/1,434 feasible candidates (31.7294%) meet the target, 106 are clipped
to zero, and their mean gap is 0.022463381411. The target therefore remains a
fixed seven-threshold normalization scale, not an improvement claim; the
paired 0.0005 factor gate still controls advancement.

Every F trajectory must retain `best_feasible_normalized_gap_p50/p90`,
`quality_utility_success_mean`, and `quality_credit_group_mass_mean` every
optimizer step. Over the complete step-301--400 window, a trajectory is marked
saturated when median successful-group quality utility is at least 0.99 and
median quality-credit group mass is at most `1e-4`; a cell is marked saturated
when at least two of three seeds meet that rule. If F0 or F5 is cell-saturated,
the pilot is underidentified: no nonbaseline factor advances, and the result
cannot reject magnitude-aware credit. This is a terminal interpretation rule,
not an early stop. Every terminal arm must also report best-of-8 and feasible-
candidate occupancy at the unchanged 0.5, 0.75, 1, 1.25, 1.5, 2, and 3 target
multipliers. No alternative training target may be substituted inside this
2x2.

There is a separate target-reference limitation. The frozen development labels
are strictly replayed feasible tours with internally consistent raw-Euclidean
objectives, but they are heuristic references, not certified optima. Thus
"1.07%" means gap to that exact frozen heuristic reference. Frontier-CDF
inverts the exponential objective score and clips better-than-reference gaps to
zero, so it cannot distinguish improvements beyond the reference. The freezer
audits all 3,200 training references and all 200 selected development
references, records `optimality_certified=false`, and binds their outputs. This
is adequate for a matched development hypothesis test, not a benchmark/SOTA
claim.

The current fixed-null Frontier-CDF difference has a known all-infeasible
cancellation: `-lambda` appears in both full and counterfactual sets. The F arm
therefore requires relative failed recovery, which replaces only failed-group
credits with signed, centered, lower-violation credit scaled by `1+lambda`.
Successful groups retain Frontier-CDF credit. This removes that known
cancellation, but it does not guarantee a useful signal.

## SFT KL anchor and fidelity boundary

Beta 0.05 must never use TRL's raw-base/adapter-disable fallback. With biased
LoRA, disabling the adapter retains trained bias values and is not a fixed
step-zero SFT reference. D5 and F5 require:

1. the exact immutable relocated `bias=all` adapter at
   `experiments/sota_20260814/local_development/`
   `tsp_sft_saved_policy_bias_all_frozen_base_v1`;
2. its internal `clasp-development-adapter-base-relocation-v1` receipt, with
   semantic SHA-256
   `2030fd314dae6857999f24de8103f2051a28a6ed908779ac2c21218ebec8d000`
   and file SHA-256
   `3b03875fcb2729c75b147d2fa6ef66a4dc747d48e38614ca51e4a0b8ca1ab52e`;
3. relocation producer SHA
   `29ab2608db680d0a2510b28e4a07d0ea2f133ed6c606c8f2579d1a6f1b7ce640`
   and exact PEFT verifier SHA
   `9cffcbe6abfaac35e9d4f7ac2c74ed86f91aee63258c95bb8204673744ea9c06`;
   post-step verifier SHA
   `fedf7042386c33848cb7a734d75c409709031472396b68016615e3b68f19d9f9`;
   symmetric PEFT loader SHA
   `3acdec37e991a4c1c184920193d7c4771a4a41888bf9f1dc9d1401ca8625e382`;
   development-fidelity SHA
   `936adebbdb5f25f3ae5bdb030587d87d450aefcecc5e87235c30466a2c2df5da`;
   and GRPO entry-point SHA
   `a35e4a4cb0882bc221469bccb49b60a9e5717b4275169b8b5d13412975923c49`;
4. the immutable frozen base/tokenizer bound by tree SHA-256
   `ace0432b498998365629c8a4a71042b8f117ff5e782fb368a76fe7e39df0e5a0`;
5. explicit `CLASP_KL_REFERENCE_MODEL_PATH` and
   `CLASP_KL_REFERENCE_RECEIPT` values.

For D5/F5, `CLASP_KL_REFERENCE_MODEL_PATH` is exactly the same relocated adapter
used as `--model_path`, and `CLASP_KL_REFERENCE_RECEIPT` is exactly its internal
`adapter_base_relocation_receipt.json`. The entry point loads two distinct
model objects from the same immutable base+adapter: the policy remains
trainable, while the reference has zero trainable tensors and disjoint storage.
Both calls use the same explicit `autocast_adapter_dtype=True` PEFT path; the
source binding, runtime training contract, and post-step receipts all bind those
two booleans as true. Saved LoRA remains FP32 and both live LoRA inventories must
be FP32 and bitwise equal.
The reference receives the already-loaded bound tokenizer and replays the exact
policy special-token transform. Policy and reference `model.config` and
`generation_config` must all exactly equal the bound tokenizer for BOS/EOS/PAD,
and every non-null ID must be valid for both tokenizer and model vocabularies;
an equal-but-wrong model pair is rejected. The bound Qwen tokenizer deliberately
keeps BOS unset, EOS 151645, and PAD 151643.
Before optimization it requires bitwise-equal state and three fixed-probe full
distributions, log probabilities, and argmax, with all-rank digest consensus.
The runtime receipt protocol/status are
`clasp-development-exact-peft-kl-reference-v2` /
`development_exact_step_zero_peft_reference_verified`, written as
`exact_peft_kl_reference.json` at the run root and copied into checkpoints.
`sync_ref_model`, vLLM, FSDP, and DeepSpeed are forbidden.

After optimization begins, beta-positive runs also require schema-2
`clasp-development-exact-peft-kl-reference-post-step-v2` /
`development_exact_peft_reference_immutability_verified` receipts from verifier
`clasp-exact-peft-kl-reference-post-step-verifier-v2`. The authoritative step-one
file is `exact_peft_kl_reference_step1.json`; each checkpoint and final root use
`exact_peft_kl_reference_post_step.json`. These receipts bind the immutable
pre-step pair, exact run-contract horizon, live reference tensor values, zero
reference trainables, prepared object identities, optimizer exclusion, disjoint
policy/reference storage, policy-change context, all-rank consensus, and
in-process CUDA identity/peak memory.

The beta-zero cells explicitly unset both KL-reference variables and must not
emit an exact-PEFT runtime receipt. Every arm uses
`CLASP_DEVELOPMENT_FIDELITY=1` and publication fidelity is unset/zero. This
requires exact live trainable-state closure and exact Qwen2.5-7B r64
seven-module `bias=all` structure while keeping `publication_invariant=null`.
Both `checkpoints/checkpoint-400` and the final run root must contain a valid
`clasp-checkpoint-fidelity-v1` receipt before terminal evaluation. D5/F5 must
also contain valid pre-step, step-one, and post-step sidecars at both locations.
D0/F0 must contain none of those exact-KL sidecars; their immutable matrix-ledger
identity is instead cross-bound through `hyperparams.json`, exact authorized
arguments/environment, and the create-once terminal audit.

The immutable frozen base is every arm's `--tokenizer_path`; the adapter itself
must never be used as the tokenizer path. The converted adapter remains
upstream lineage, while the relocated adapter is the direct policy
parameterization. This prevents raw-foundation fallback and makes starts
matched across D0/D5/F0/F5, but does not recover omitted historical terminal
biases or establish exact equivalence to the FP32 source policy.

### Mandatory one-step beta memory gate

No 400-step cell may start until a one-step beta=0.05 gate succeeds with the
same K=8, generation batch eight, completion limit 3,000, policy microbatch one,
and accumulation 32 used by the pilot. It keeps the 400-step scheduler horizon
and uses `GRPO_STOP_AT_GLOBAL_STEP=1`, but fixes `warmup_ratio=0` for this gate:
the scientific arm's 0.03 warmup gives update one zero learning rate and could
not satisfy the policy-change check. This prespecified nonzero-LR override makes
the gate an engineering test, not a scientific arm. It must exercise
generation, reference log-probabilities, backward, one optimizer step, and
checkpoint save; its artifact may never be resumed, extended, or used as a
400-step initialization, and its metrics cannot enter cell selection.

The gate must prove global step one, a valid checkpoint-1 fidelity receipt,
the read-only create-once gate-attempt receipt bound to the exact live
environment and payload argv,
bitwise-identical root/checkpoint-1 pre-step and step-one sidecars, a valid
checkpoint-role post-step receipt, at least one changed policy-trainable tensor,
zero reference trainables, exact live-reference equality to the pre-step
inventory, stable prepared/unwrapped identities, disjoint policy/reference
storage, optimizer exclusion, no OOM/NaN/Inf, and per-rank in-process CUDA peak
allocated/reserved bytes. Core intentionally permits a zero-change step-one
receipt, so the pilot's external gate rejects it unless
`changed_trainable_tensor_count > 0`.

Because this is a hard stop below the 400-step scheduler horizon, the root has
an exact allowlist: `_claim_anchors`, `_runtime_tmp`, `checkpoints`,
`hyperparams.json`, the authoritative pre-step receipt, and the authoritative
step-one receipt. It must not contain a root model, root checkpoint-fidelity
receipt, root final post-step receipt, final trainer state, or any unrecognized
entry. `checkpoints` must contain only `checkpoint-1`, which is never resumed.
The core post-step auditor is now frozen and tested; the remaining prerequisite
is empirical: this create-once gate has not been run. Therefore the
source/template is ready to freeze, but all 12 scientific cells remain locked
until the gate PASS and full-matrix reservation closure exist.

## Frozen development evaluation and analysis

The terminal checkpoint is evaluated once on rows `[0, 200)` of one immutable
historical development source. The freezer hashes every selected instance's
canonical coordinate list and rejects any overlap with all 3,200 RL training
instances. Every arm uses evaluation sampling seed 271828 and raw best-of-8
generation. Independent per-arm checkpoint selection is prohibited.

The terminal endpoint is two-part: raw pass@8 feasibility, then mean raw
best-of-8 objective gap among feasible groups; an all-infeasible group is a
failure, not a repaired observation. Pairwise gap contrasts use jointly
feasible instances only and must report that denominator plus each arm's
failure rows. A required sensitivity assigns every failed group a gap of 1.0;
if a factor-effect direction changes, that factor cannot advance. The
prespecified contrasts are:

- Frontier effect at beta 0: `F0-D0`;
- Frontier effect at beta 0.05: `F5-D5`;
- anchor effect under dense credit: `D5-D0`;
- anchor effect under Frontier-CDF: `F5-F0`;
- interaction: `(F5-F0)-(D5-D0)`.

Report every matched-seed contrast and a seed-blocked hierarchical bootstrap
over training seeds and then paired instances. Three seeds are too few for a
strong population claim, which is another reason the evidence remains
development-only.

Cell eligibility requires mean raw gap at most 0.0107 and pass@8 feasibility at
least 0.995. Mechanism decisions use the factorial main effects. Frontier-CDF
must improve `0.5*((F0-D0)+(F5-D5))` by at least 0.0005 absolute gap, with a
pooled improvement in at least two of three seeds. The beta-0.05 anchor must do
the same for `0.5*((D5-D0)+(F5-F0))`. The interaction is reported with
uncertainty but is not a selection gate. Select the lowest-gap eligible cell
whose nonbaseline factor levels each pass; within 0.0001 choose lower beta,
then dense credit. If no nonbaseline factor passes, no new mechanism advances
and D0 remains the control.

Health stops are limited to NaN/Inf telemetry, immutable input or source drift,
missing/invalid checkpoint-fidelity evidence, a terminal step other than 400,
or detected resume/scheduler extension. Statistical underperformance is not an
early-stop reason.

Per-step telemetry is mandatory, not optional debugging output. Common arms
must retain loss, learning rate, gradient norm, candidate/group feasibility,
and advantage L2/absolute mean; KL is additionally mandatory for beta-positive
arms. F arms additionally retain target
gap and temperature, feasible-gap quantiles, nonzero/zero credit rates, null
decomposition error, failed-group relative-recovery magnitude/sign coverage,
and dual activity/at-zero rates. Missing required telemetry invalidates a
trajectory before arm selection. This is particularly important because the
same beta has a larger reward-relative effect when Frontier-CDF advantages are
smaller; the pilot estimates that interaction rather than silently rescaling
one method.

## Historical v1/v2 failures and v3 freeze workflow

The original v1 preregistration is historical and must not be modified. Its spec
and contract remain at
`experiments/sota_20260814/local_development/tsp_fcdf_beta_pilot_20260815_v1.spec.json`
and
`experiments/sota_20260814/local_development/tsp_fcdf_beta_pilot_20260815_v1.contract.json`;
its failed tree remains at `outputs/tsp_fcdf_beta_pilot_20260815_v1`. Its spec
file SHA-256 is
`cc10f2982aefbe9a3eeba05904426d521cd2a1c01c0b0f69a3e5a26c67cc46e8`;
its contract semantic/file SHA-256 values are
`284addf001628633eb56059fc6f40bd3c981bfa95e579bfb303a8d751c729827` and
`60468e72e21bc30513cf49375e4683e1173e22864918ec22a57c34312127a998`.
The immutable gate claim SHA-256 is
`5346809fbafacef5282d8e23f6060630d1bc8ab998eae5e0c07aef303eccb2f9`,
and the retained hyperparameters SHA-256 is
`fabc28f90621510081cb9b8311e5dd7ca07ca7a6947e56c4d0a154d8b1c21411`.
That gate stopped at the exact paired-config check because policy BOS was null
while reference BOS was 151643; the complementary raw/policy PAD asymmetry was
also present. It produced no checkpoint, policy update, PASS, matrix ledger, or
scientific command, and the GPU was released. V1 predates supervised transcript
and outcome receipts, so its exit details are unretained and must never be
invented.

A later, separate historical-failure receipt may safely bind those four immutable
file identities plus an exact v1 tree manifest, label the event
`unsupervised_exit_details_unretained`, and set evidence/retry/matrix/publication/
SOTA eligibility false. It should live outside the v1 output tree, be create-once
and self-hashed, and only then authorize a recursive read-only seal of the v1
tree. It must not masquerade as the v2 supervised FAILED protocol. This source
update does not create that receipt or alter v1.

V2 is also immutable and terminal. Its spec and contract remain at
`experiments/sota_20260814/local_development/tsp_fcdf_beta_pilot_20260815_v2.spec.json`
and
`experiments/sota_20260814/local_development/tsp_fcdf_beta_pilot_20260815_v2.contract.json`;
their file SHA-256 values are
`6f2402c2b578db17dc00b985059a059ba28a9001f89ea897d24f180bf667685a`
and
`2d11b2b6367ec419536cf5240519b18c04f69078252a1e23e4f62b23ddd3250f`,
and the contract semantic SHA-256 is
`614035ede9517ab2f8ebee0abe9ec2738da0978800c795dec22fb6d139d4aec1`.
The supervised child exited 1 during the pre-update exact-pair audit: the policy
LoRA tensor was FP32 while the independently loaded reference tensor was BF16 at
`base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight`. The
create-once outcome semantic/file SHA-256 values are
`3030e11af13287e743a5e79b0db95844b8af0b365d5f7a41cb52a3f8e669355a`
and
`f0915d11febe0b8d962e648835daadcf060e8005f3702128c7eaf1dbb3dad219`.
Its sealed tree has no checkpoint, optimizer update, PASS, matrix authorization,
reservation closure, or scientific cell. V2 must never be rerun, retried,
rewritten, or used to authorize `--record-memory-gate-pass`.

The causal defect was PEFT 0.19.1 loader asymmetry. The policy auto-loader used
its implicit `autocast_adapter_dtype=True`, promoting LoRA to FP32, while the
manual frozen-reference loader explicitly disabled that promotion. The corrected
loader now performs the base and PEFT stages explicitly for both objects and
passes `autocast_adapter_dtype=True` in the common adapter call. The strict gate
still requires complete key, shape, dtype, value, storage-independence, and
runtime-autocast logit equality; it is not relaxed.

The corrected source is eligible only for a separately reviewed v3 freeze. The
checked-in template remains `HOLD_TEMPLATE_NOT_FROZEN`; the recommended v3 spec,
contract, and output paths are all absent. The planned unique v3 identities are:

- spec:
  `experiments/sota_20260814/local_development/tsp_fcdf_beta_pilot_20260815_v3.spec.json`;
- contract:
  `experiments/sota_20260814/local_development/tsp_fcdf_beta_pilot_20260815_v3.contract.json`;
- output:
  `outputs/tsp_fcdf_beta_pilot_20260815_v3`;
- campaign: `tsp_fcdf_beta_20260815_v3`.

To freeze later, copy the template to a separately reviewed spec and change
only that copy to `READY_TO_FREEZE`. The exact local train, development,
sealed-SFT, Python, adapter, tokenizer, contract, and output paths are already
filled in the HOLD template. The freezer requires `schema_version=4`, protocol
`clasp-tsp-fcdf-beta-development-pilot-spec-v4`, campaign
`tsp_fcdf_beta_20260815_v3`, the absent output/contract paths, all frozen hashes,
the exact 200-row saturation audit, and zero train/development overlap.

The existing relocation can be reauthorized without rewriting it:

```bash
PYTHONPATH=src python3 scripts/relocate_development_adapter_base.py \
  --source-adapter-path /home/lamsade/habgaryan/phd_project/correct_clasp/experiments/sota_20260814/local_development/tsp_sft_saved_policy_bias_all_v1 \
  --frozen-base-model-path /home/lamsade/habgaryan/phd_project/correct_clasp/experiments/sota_20260814/local_development/Qwen2.5-7B-Instruct_frozen_v1 \
  --output-dir /home/lamsade/habgaryan/phd_project/correct_clasp/experiments/sota_20260814/local_development/tsp_sft_saved_policy_bias_all_frozen_base_v1 \
  --verify-only
```

This strict verifier rehashes the source, base, and relocated output and checks
the recorded independent FP32 state/probe proof. The beta-positive training
entry point then independently loads policy and reference, rehashes source
bindings, and creates the BF16 exact-pair runtime receipt before optimization.
A self-hashed receipt containing asserted digests is not sufficient. The sealed
SFT `inference_summary.json` and 200-record `run_details*.jsonl` must have SHA-256
`a065aeadbf0811462492e79233b3d11edbd745b92799236db492041213512967`
and
`5d6b56e64092be597d1fe4fe389b30c471cd2a2892cc6a033fd0758e319a0046`,
respectively. The freezer independently recomputes all occupancy statistics.
Any stale core/source byte, symlink, wrong receipt/tree/dtype/claim, adapter used
as tokenizer, overlap, or pre-existing destination fails closed.

Then freeze the design:

```bash
/home/lamsade/habgaryan/phd_project/my_idea_grpo/.venv/bin/python \
  scripts/freeze_tsp_fcdf_beta_pilot.py \
  --spec /absolute/path/to/reviewed-ready-spec.json \
  --output /home/lamsade/habgaryan/phd_project/correct_clasp/experiments/sota_20260814/local_development/tsp_fcdf_beta_pilot_20260815_v3.contract.json
```

The freezer creates only one self-hashed read-only contract and starts no
process. This source update deliberately does not run that command.

Before the gate, the default renderer verifies the live contract, all pinned
source/input bytes, runtime, and GPU; it emits only relocation verification and
the one create-once engineering gate. The 12 scientific commands are absent:

```bash
/home/lamsade/habgaryan/phd_project/my_idea_grpo/.venv/bin/python \
  scripts/dry_run_tsp_fcdf_beta_pilot.py \
  --contract /home/lamsade/habgaryan/phd_project/correct_clasp/experiments/sota_20260814/local_development/tsp_fcdf_beta_pilot_20260815_v3.contract.json \
  --format shell
```

Every gate, cell, audit, and evaluation rendering uses `env -i` and a complete
allowlist. Before consuming an attempt and again immediately before spawn/exec,
the same PID chain requires the exact live environment and inner payload argv to
equal the frozen gate command or ledger command. It reopens the contract,
rehashes every frozen source/input, verifies the reservation/ledger identity,
checks the pinned Python/package/CUDA versions, and binds physical GPU 3 (UUID
`GPU-025e10e6-263f-d814-6dd5-added86fc8af`, RTX A6000, 49,140 MiB, driver
535.261.03). The legitimate `/usr/bin/nvidia-smi` alternatives chain resolves
through `/etc/alternatives/nvidia--nvidia-smi` to the regular executable
`/usr/lib/nvidia/current/nvidia-smi`, size 686,384, mode 0755, SHA-256
`d2ce2e23e90b1044e8cbf9e60c93baefc7abbc2faa43275528916d652e983db3`.
The freezer records every symlink and resolved-file inode identity and rejects a
broken, cyclic, nonregular, current-user-writable, retargeted, or byte-drifted
probe. Host `root:root` and the managed user namespace's `65534:65534` overflow
view normalize to one trusted-system-owner identity, so the contract does not
spuriously drift across those two views; any other owner mapping fails. The
same-process runtime guard executes the resolved path and rehashes it before and
after the query. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is fixed.
HOME,
TMPDIR, TMP, TEMP, XDG, Hugging Face, Torch, Triton, TorchInductor, and CUDA
caches must all descend from that attempt's workspace-local `_runtime_tmp`.
`tempfile.gettempdir()` must equal it, statfs must report at least 100 GiB free,
and `/tmp` fallback is forbidden.

The memory gate is the only supervised child-process surface; there is no
general execute CLI. Before the child starts, its immutable attempt claim binds
a mode-0440 combined stdout/stderr transcript inode and one precreated outcome
inode. Every execution anchor has one retained hard-link generation guard in an
exact sealed subdirectory; each pair must remain on the same inode with link
count exactly two, so unlink/recreate and extra-link drift fail even when a
filesystem immediately recycles inode numbers. Failure manifests explicitly
exclude the two exact names of the self-referential outcome inode while retaining
same-basename files elsewhere. The claim also binds the sealed anchor directory's
full device/inode/mode/mtime/ctime identity and the containing gate-directory
entry. The create-once outcome closes over the exact post-execution versions of
both identities, which PASS rechecks. The supervisor launches a tiny audited
Linux wrapper in a new session and
process group. The wrapper sets `PR_SET_PDEATHSIG`, immediately verifies its
expected parent PID to close the fork-to-prctl race, kills its whole group if the
supervisor dies, and then execs the exact authorized payload. The transcript
records the wrapper PID, Linux start-time ticks, PGID, and session ID before any
payload output. Signal forwarding and cleanup address the entire group. Before
any zero, nonzero, signal, spawn-error, or operator-unknown receipt is persisted,
two stable inspections must prove the exact wrapper group, authorized payload,
and target-GPU compute process are absent. A normal zero receipt is required for
PASS. Every other outcome consumes the same mutually exclusive slot, recursively
seals the tree, and permanently forbids retry and matrix authority. Operator
finalization is limited to an abruptly dead supervisor, requires the transcript
flock to be released, binds the last complete self-hashed supervisor event, and
records `child_started=null` unless a complete spawn identity exists.

After the gate finishes, no human assertion can unlock cells. This create-once
transition audits/seals the complete gate tree, writes PASS outside it, creates
one pre-outcome 12-cell ledger, reserves all 12 train and 12 evaluation slots,
precreates every possible single-use first-attempt/evaluation/resume claim
inode inside read-only anchor directories, and closes those reservations before
emitting any scientific command:

```bash
/home/lamsade/habgaryan/phd_project/my_idea_grpo/.venv/bin/python \
  scripts/dry_run_tsp_fcdf_beta_pilot.py \
  --contract /home/lamsade/habgaryan/phd_project/correct_clasp/experiments/sota_20260814/local_development/tsp_fcdf_beta_pilot_20260815_v3.contract.json \
  --record-memory-gate-pass \
  --format shell
```

The ledger binds exact argv/env/output/evaluation attempts and becomes every
cell's `CLASP_TRAINING_RUN_CONTRACT_PATH`. Each training first start and each
evaluation atomically consumes one closure-bound empty anchor inode into a
read-only claim. A training retry is permitted only from the surviving newest
nonterminal checkpoint with exact receipts and one checkpoint-specific anchor;
the wrapper then sets `RESUME_FROM_CHECKPOINT` to that exact claimed path before
the second authority check. Deleted, recreated, fresh, terminal, or already
claimed artifacts fail. Evaluation has no retry: a failed/preempted attempt
stays in the denominator. Frozen anchor-directory metadata and per-file inode
identities detect deletion/recreation. The ledger is never regenerated after
any attempt.

At step 400, the create-once terminal audit exact-crossbinds root and
checkpoint-400 trainable closures, ledger identity, hyperparameters/arguments,
all 400 telemetry records, beta-specific exact-KL evidence, and the absence of
an earlier evaluation attempt. Only then can the preauthorized one-shot
evaluation claim succeed. Missing, failed, or invalid cells remain among all 12
required trajectories; evaluation attempt paths cannot be replaced or rerun.

Current verdict: **v3-ready at the source-contract layer;
HOLD_TEMPLATE_NOT_FROZEN operationally**. The recommended v3 spec, contract,
and output remain absent; no v3 gate or GPU payload has run, and no v3 matrix
command is authorized or emitted. V1 and v2 remain immutable historical
pre-update failures.
