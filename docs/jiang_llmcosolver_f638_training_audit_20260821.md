# Jiang et al. LLMCoSolver training audit (2026-08-21)

This is the reproducibility record for *Large Language Models as End-to-end
Combinatorial Optimization Solvers* (Jiang et al., NeurIPS 2025). It separates
what the paper says, what the official code executes, and the declared
departures required for the claim-bearing, serialization-safe SFT lane. That
lane is recipe-matched rather than byte-identical: it uses clean data,
`bias="all"`, a fresh `trainer.train()` call, and a fail-closed contextual
response mask. Only author/maintainer primary sources are used for claims about
the Jiang setup.

As of this audit, no corrected v8 specialist job has been submitted or is
running. The local execution contract passes its focused validation suite, but
the remote immutable source snapshot, five clean corpora, sealed exact
environment, and frozen campaign authority do not yet exist.

## Frozen primary sources

- Official repository: [Summer142857/LLMCoSolver](https://github.com/Summer142857/LLMCoSolver)
- Audited repository commit:
  [`f6388991d243c0aa91d9ee7e2fa18ed1c9e72477`](https://github.com/Summer142857/LLMCoSolver/commit/f6388991d243c0aa91d9ee7e2fa18ed1c9e72477)
- Git tree: `4017f8b4c78bf057eeadc3f5b0e2ee1c0000a77d`
- Commit author/date/message: Xia Jiang, 2025-10-24 09:36:26 +0200,
  `Update eval_utils.py`; GitHub reports a verified signature.
- `main` and remote `HEAD` both resolved to `f6388991...` on 2026-08-21.
- The repository had no tags and the GitHub releases endpoint returned no releases.
- Final conference paper:
  [NeurIPS 2025 PDF](https://papers.neurips.cc/paper_files/paper/2025/file/f0c6c93e1dbff84daf6082fef7e1a094-Paper-Conference.pdf)
- Author arXiv record:
  [arXiv:2509.16865](https://arxiv.org/abs/2509.16865), v1, submitted
  2025-09-21; no later arXiv revision was listed on the audit date.
- Base model ID named by code, without a revision:
  [`unsloth/Qwen2.5-7B`](https://huggingface.co/unsloth/Qwen2.5-7B). The
  corrected local lane freezes the audited snapshot
  [`86e6e1eccea424c7fb77d59f7d2a9125c0673bfd`](https://huggingface.co/unsloth/Qwen2.5-7B/tree/86e6e1eccea424c7fb77d59f7d2a9125c0673bfd),
  whose card/config identifies `Qwen/Qwen2.5-7B` as the base; this does not
  establish which mutable Hub revision the authors used historically.
- Official TRL version source:
  [`v0.15.2` commit `fc2b041b58f6fbe766dceaec819bc5a8f9d209da`](https://github.com/huggingface/trl/tree/fc2b041b58f6fbe766dceaec819bc5a8f9d209da)
- Official Transformers version source:
  [`v4.50.3` commit `a78e884f...`](https://github.com/huggingface/transformers/tree/v4.50.3)

The exact audited upstream file hashes are:

| File | SHA-256 |
|---|---|
| `main_train.py` | `a37901e49fcea759f9fb960e67a9c38a6f73aad4da8f79df222aa8e1bde84f13` |
| `rl_train.py` | `c6b7703e8f621e136e78724a0d4788795aeda58188f51f3351981d6409ccf583` |
| `eval.py` | `6f60157115e69fe0550fb43a54ad506281859c473b7d63eb6b14714a944f72c3` |
| `utils.py` | `88b5c9c06fad5025e8d3caf6ed24d0a0b8f97dacefc537b3a60d27c0bd512db1` |
| `rewards.py` | `879f73a0b4f6ae5d4227a8d4ce8dbf1aa5cc966b526c6d76bba5a2c29a2380c5` |
| `requirements.txt` | `b8c309502184dbb01f01c3d016c70d5d69d04ce03ec770ae7be720424a9a22b6` |
| `README.md` | `09b2bfea2a6af3d56457d72d4d34c8039d61319fa5ff5d0e17436fe34eda4050` |
| `cmd.sh` | `88702b0bd35b8c5eac74645cd715587f60284ab59c263050e5cb91c1c27cfd84` |

## Exact published SFT recipe

The paper's Appendix C.4 and the released `main_train.py` jointly specify:

| Control | Published value |
|---|---|
| Base | `unsloth/Qwen2.5-7B` (base, not Instruct) |
| Context | 20,000 tokens |
| Precision / quantization | bfloat16; no 4-bit loading |
| Training set | 500,000 examples for each task specialist |
| Epochs | 1 |
| Per-device batch | 4 |
| Gradient accumulation | 4 |
| Effective one-device batch | 16 |
| Terminal step for 500k rows | 31,250 |
| Optimizer | `adamw_8bit` |
| Learning rate | `2e-4` |
| Warm-up | 20 steps |
| Scheduler | linear |
| Weight decay | 0.01 |
| Seed | 42 for dataset shuffle, PEFT initialization, and Trainer |
| LoRA | rank 64, alpha 64, dropout 0 |
| Targets | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Gradient checkpointing | Unsloth mode (`"unsloth"`) |
| Packing | false |
| Dataset workers | 16 |
| Intended loss | response-only after a response header |
| Executed released loss | full-example LM fallback for exact Qwen/f638 text; see contradiction 1 |
| Evaluation during SFT | disabled (`evaluation_strategy="no"`) |
| Logging / checkpoint interval | 1 / 5,000 steps |
| Checkpoint retention | 50 |
| Hardware reported | one NVIDIA H100; AMD EPYC 7F72 host |
| Reported JSSP SFT resource | approximately 65 hours and 28 GB of GPU memory |
| Result-selection policy | three independent LLM experiments; best result reported |

The released data path is loaded with `split="train"`, then exactly one
`Dataset.shuffle(seed=42)`, then one batched formatting map. Trainer sampling
is also seeded at 42. The literal prompt has a trailing space on its first
line, four spaces before every subsequent non-empty line, embeds the target
directly after `### Response:`, and appends one tokenizer EOS string. These
bytes differ from the cleaner generic Alpaca prompt shown typographically in
the paper. The released collator's standalone marker tokenization does not
occur inside that literal formatted Qwen text.

The corrected lane matches the gradient-bearing recipe, not every inert side
effect of the released script. In particular, the released script loads,
formats, and passes an evaluation dataset even though
`evaluation_strategy="no"`; the corrected lane has no evaluation dataset and
records `eval_size=0`. It also uses frozen local paths, different W&B naming,
a nested periodic-checkpoint directory, explicit terminal save, and additional
pre/post-save audits. None of those choices adds an optimizer update or an
evaluation-triggered model-selection decision, but they are reasons not to
describe the execution as byte-identical.

The official dependency pins relevant to training are:

```text
torch==2.7.1
transformers==4.50.3
trl==0.15.2
peft==0.17.0
unsloth==2025.3.19
unsloth_zoo==2025.3.17
accelerate==1.7.0
datasets==3.6.0
bitsandbytes==0.46.1
safetensors==0.6.1
xformers==0.0.31.post1
wandb==0.18.0
```

Import order is material. `unsloth` is imported before TRL in the official
program and patches the legacy `SFTTrainer` surface. Vanilla TRL 0.15.2's
constructor does not independently expose every legacy keyword passed by the
released program. The exact Unsloth 2025.3.19 compatibility wrapper maps
`tokenizer` to `processing_class`, promotes the four legacy SFT keywords into
an `SFTConfig`, and then calls TRL. The corrected lane deliberately retains
that released call path and audits every recipe-relevant effective setting
before training.

The released 177-line `requirements.txt` is not an executable environment by
itself: the historical cluster runtime receipt
`jiang_reproduction/envs/llmcosolver_f6388991_py311_plus_transitives_v1/runtime_receipt.json`
proves eight additional runtime dependencies (`annotated-types==0.7.0`,
`cffi==1.17.1`, `distro==1.9.0`, `platformdirs==4.3.8`,
`pycparser==2.22`, `pydantic==2.11.7`, `pydantic-core==2.33.2`, and
`typing-inspection==0.4.1`). In particular, omitting cffi makes the datasets
import fail at `_cffi_backend`. These pins are checked in separately as
`requirements.jiang-f638-runtime-closure.txt`; its exact bytes and parsed
inventory are bound into the v4 environment receipt. The sealed environment therefore admits
exactly those 177 released distributions, the eight separately declared and
receipt-proven runtime-closure additions, and the construction bootstrap
`pip==25.1.1`. Any other visible distribution is rejected. Every installed
file payload and the complete read-only prefix remain hash-bound.

## Released dataset schema and identity

The README links an official
[SFT Drive folder](https://drive.google.com/drive/folders/1bE1coGUa00gfuMkPXnfvldi1-WHGNnEb)
and an official
[RL Drive folder](https://drive.google.com/drive/folders/1VN9crftdW7DTsMQupbc06u6PzRT-Bwnx).
All released training files are top-level JSON arrays. Required SFT fields are
strings.

| Task | SFT rows | Common keys | Exact duplicate full rows | Local bytes | SHA-256 |
|---|---:|---|---:|---:|---|
| CVRP | 500,000 | `instruction,input,output,num_nodes,vehicle_capacity` | 2,220 | 2,709,936,196 | `9981b841c42376275d8950faa600d1a0710fd27c4e902ed655617f3d0620d0ac` |
| JSSP | 500,000 | `instruction,input,output,n,m` | 0 | 2,180,079,958 | `f21e82f01f07fdff960fda6a003f0f3a7c1eefbeeae7f00db8198195133ccd6a` |
| MIS | 500,000 | `instruction,input,output,num_nodes` | 30 | 1,975,922,309 | `b5c873c3803a0b99d4aba99b9bcc5f34ecf28f345c029810f172362c309fcb31` |
| OP | 500,000 | `instruction,input,output,num_nodes,start_node,max_route_length` | 0 | 2,618,731,138 | `01e7f08cbea12cff92b2533f39c5cc9f73caf3ddb2c578ef2da34544f853959b` |
| TSP | 500,000 | `instruction,input,output,num_nodes` | 0 | 3,451,489,028 | `e7f1936c0f641112f43b36dcb25d79d361cb2c43a183cecbf7c2cc93a8e07c1a` |

These SFT hashes identify the files frozen locally and matching the Drive
filenames and observed content lengths. The historical acquisition did not
retain a direct-download receipt, so exact Drive-byte identity must not be
claimed from these hashes alone. In contrast, all ten RL files were acquired
directly by Drive item ID and locally sealed. RL training sizes are CVRP 2,000,
JSSP 3,200, MIS 3,200, OP 5,000, and TSP 3,200. RL rows add a structured
`instance` field: an array for CVRP/JSSP/OP/TSP and an object for MIS.

The output grammars in the released SFT files are:

- TSP: `Route: [...], Objective: ...`
- CVRP: `Routes: [[...], ...], Objective: ...`
- OP: `Route: [...], Objective: ...`
- MIS: `Set: [...], Objective: ...`
- JSSP: `Schedule: [[...], ...], Makespan: ...`

## Paper/code/data contradictions

1. **The released “response-only” collator does not find its marker.** The
   formatter emits the contextual bytes `    ### Response:{output}`, while
   the collator searches for the standalone tokenization of `### Response:`.
   With the exact Qwen tokenizer, indentation changes the first token and the
   colon can merge with the first output subword (for example `:Set`). The
   standalone ID sequence never occurs. The public collator therefore warns
   and takes `fallback_strategy="full_example"`, supervising prompt and target
   rather than only the response. To implement the stated response-only
   intent without dropping the first output subword, the corrected lane finds
   the contextual header `    ### Response` (without the colon) and begins
   supervision at the immediately following token.

2. **The released fresh/resume call is internally inconsistent.**
   `main_train.py` unconditionally calls
   `trainer.train(resume_from_checkpoint=True)`. Under the published
   Transformers 4.50.3 `Trainer` semantics, this resolves the last checkpoint
   and raises `ValueError` when a fresh output has none. The repository does
   not document or test an Unsloth patch that changes this behavior, so the
   README's fresh command is not an established executable contract. The
   corrected fresh lane calls `trainer.train()`.

   The released `cmd.sh` reference to `checkpoint-31250` is consistent with
   Transformers 4.50.3: its default flow callback forces a save whenever
   `global_step >= max_steps`, even when the terminal step is off the regular
   5,000-step cadence. Thus the expected Trainer inventory is checkpoints at
   5,000-step intervals through 30,000 plus `checkpoint-31250`. The released
   program still contains no separate root-level terminal `save_model()` call.

   The released SFT constructor is executable only because import order installs
   Unsloth 2025.3.19's compatibility wrapper. That wrapper maps `tokenizer` to
   `processing_class`, copies the `TrainingArguments` values into `SFTConfig`,
   and promotes `dataset_text_field`, `max_seq_length`, `dataset_num_proc`, and
   `packing` before invoking TRL 0.15.2. The corrected lane uses this literal
   released constructor surface and fail-closes if the wrapper's effective
   config differs, including the otherwise easy-to-misreport effective defaults
   `per_device_eval_batch_size=8` and `eval_steps=None`.

3. **The released bias setting and exact Unsloth post-patch disagree.** The
   paper says only LoRA matrices are trainable, while code requests
   `bias="lora_only"`. PEFT would initially expose the targeted Q/K/V biases,
   but Unsloth 2025.3.19 subsequently runs its generic training preparation,
   which re-freezes every parameter except LoRA A/B (and optional magnitude
   vectors). Merely changing the PEFT config to `bias="all"` is therefore not
   sufficient. The corrected lane uses `bias="all"` and then explicitly
   re-enables and audits exactly the 84 Q/K/V `base_layer.bias` tensors before
   the first optimizer step. All base biases on this Qwen2.5-7B snapshot are
   exactly those targeted biases, and PEFT serializes them in the adapter. This
   is a deliberate trainability/persistence correction, not byte-identical
   execution of the released program.

4. **“500,000 instances” does not mean 500,000 unique demonstrations.** The
   released CVRP file contains 2,220 exact duplicate rows and MIS contains 30.
   Clean 500k replacements are necessary for the planned controlled result.

5. **Author evaluation is not globally held out.** Exact instance overlap is
   299/300 for the union of CVRP training sources versus RL eval, 300/300 for
   MIS, and 299/300 for OP. SFT train versus SFT eval is 300/300 for CVRP and
   299/300 for OP. JSSP and TSP have zero observed overlap. A new disjoint IID
   test is required for an ICLR claim; it is a contamination correction rather
   than exact reproduction of the released split.

6. **JSSP label wording differs.** The paper's illustrative output uses
   `Objective: 466`; released JSSP data and the public evaluator use
   `Makespan:`. Training must follow the actual released/verified data grammar.

7. **CVRP capacity generation differs.** The paper says capacity is the mean
   demand. The public generator samples `r` from a triangular distribution
   over 3, 6, and 25 and uses `int(r * mean(customer demands))`.

8. **JSSP's upper processing-time endpoint differs.** The paper states integer
   times in `[1,100]`; the generator uses NumPy `randint(1,100)`, hence 1--99.

9. **OP closure semantics differ.** The paper says an OP path need not return
   to the depot, but the shared public distance helper unconditionally adds
   the last-to-first edge. Generator/reward route-length behavior therefore
   closes the path.

10. **The repository does not publish a canonical 500k generation command.**
   Environment `__main__` snippets are small/demo configurations and, for TSP,
   the helper weights size selection in proportion to node count rather than
   sampling sizes uniformly. Exact released-data RNG provenance cannot be
   reconstructed from one documented command.

11. **The paper's RL clipping epsilon is not established by public code.** The
    paper lists epsilon 0.1. `rl_train.py` parses epsilon but comments it out
    when constructing `GRPOConfig`; official TRL 0.15.2 has no epsilon config
    field or clipped-ratio term in its published GRPO loss. Reproducing epsilon
    0.1 would require behavior not established by the released code/runtime.

12. **Paper and README decoding controls differ.** The paper describes BoN
    with top-p 0.7 and `N=8`; the README example labels `--temperature 0.7`,
    while code defaults should be inspected directly for any exact comparison.

## Missing artifacts

The official repository links data but no trained checkpoint. On 2026-08-21,
the repository had no release asset or tag, and Hugging Face API searches for
the paper/repository name and author account returned no author model or
dataset repository. There is therefore no Jiang adapter/model card whose
weights can be used to validate a reproduced checkpoint bitwise. The frozen
base-model revision used by the corrected lane, code commit, data identities,
training receipts, and fresh save/reload equality evidence are the available
reproducibility anchors. Because the released command did not pin a model
revision, the corrected lane's frozen revision is a controlled choice rather
than evidence of the authors' historical base bytes.

The current five-cell campaign freezes the only released SFT seed, 42, once per
task. It does not implement the paper's underspecified “three independent LLM
experiments; best result reported” rule: neither the other seeds nor the exact
selection stage/metric are published. For an ICLR comparison, independent
seeds should be reported as variation rather than recreated through an
unverifiable post-hoc best-of-three selection.

## Implemented corrected entrypoint

The dedicated worker entrypoint is
[`train_sft_jiang_f638.py`](../src/clasp/entrypoints/train_sft_jiang_f638.py).
It preserves the f638 prompt, import order, fixed hyperparameters, one dataset
shuffle, and intended response-only objective. It retains the released
Unsloth-patched trainer call and adds the declared corrections and
certification gates:

- fresh output and `trainer.train()` with no resume path;
- the literal legacy `SFTTrainer` constructor, followed by an exact audit of
  the effective `SFTConfig` produced by Unsloth's compatibility wrapper;
- `bias="all"` plus exact live trainable-to-PEFT closure preflight;
- explicit post-Unsloth restoration of exactly 84 targeted Q/K/V biases;
- a second identical 392-LoRA/84-bias closure audit after trainer construction,
  before the first optimizer step;
- an all-500k-row exactly-once contextual raw/tokenized header and
  no-truncation audit;
- a collator that raises instead of falling back to full-example loss;
- response supervision beginning at the token immediately after the
  contextual `    ### Response` header, preserving any `:output` crossing
  token;
- Trainer checkpoints at steps 5,000 through 30,000 plus the forced terminal
  `checkpoint-31250` under `RUN_DIR/checkpoints/checkpoint-N`, with an exact
  `checkpoint_fidelity.json` receipt at every save;
- a separate terminal adapter at step 31,250, hyperparameters, trainer logs,
  and `stage_status.json` under `RUN_DIR`; `stage_status.json` is written before
  the independent reload audit so a certification interruption can be safely
  recovered without another optimizer update; the completed-run sealer also
  requires this root adapter to be byte-identical to the forced
  `checkpoint-31250` adapter; and
- optional `--post-save-reload-check`, which runs the existing independent
  two-load, full-state and fixed-probe equality certification.

Clean-data enforcement sits at the v8 campaign boundary rather than inside the
standalone formatter: the worker and validator require the frozen 500,000-row
per-task cleanliness and all-row token-audit receipts before the entrypoint is
executed.

The fixed worker CLI is:

```bash
python -m clasp.entrypoints.train_sft_jiang_f638 \
  --task tsp \
  --model-path /frozen/Qwen2.5-7B \
  --data-root /clean/sft500k \
  --output-dir /runs/tsp/seed42 \
  --cache-dir /scratch/cache \
  --report-to wandb \
  --post-save-reload-check
```

There are intentionally no CLI switches for rank, alpha, target modules,
batching, learning rate, scheduler, epochs, context, shuffle seed, or save
interval. Runtime drift is rejected before Unsloth import.

## Matched specialist evaluation contract

The released `eval.py` uses a second byte-sensitive prompt that ends exactly at
the indented `    ### Response:` colon, with no newline, EOS, or chat template.
The evaluator now recognizes `prompt_style="jiang_f638"` directly instead of
silently falling back to generic Alpaca. The separate released FOARL prompt,
which adds the `<reasoning></reasoning>` sentence, is also bound for future RL;
evaluation remains explicitly no-reason as in `eval.py`.

Direct Jiang evaluation inherits `max_new_tokens=3000`, permits prompts up to
the loaded Qwen 131,072-token context minus that completion budget, and refuses
to truncate. The exact representation profile preserves released MIS `Set:`
labels and disables every optional CLASP rewrite. With either checked-in
launcher, set:

```bash
EVALUATION_REPRESENTATION_PROFILE=jiang_f638_sft
```

The default `native` profile is retained for historical corrected lanes. A
profile/environment conflict now aborts rather than changing the evaluated
instruction grammar. Decoding must still be declared explicitly for a paper
comparison: released `eval.py` defaults BoN to `N=8`, temperature 0.7, and
top-p 0.9, whereas the paper reports top-p 0.7.
