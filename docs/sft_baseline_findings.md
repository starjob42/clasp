# SFT baselines vs Jiang et al. (LLMCoSolver) — protocol, gaps, and what to do about it

Date: 2026-08-04. Cluster: Jean Zay A100 (`gpu_p5`, `qos_gpu_a100-dev`, account `vas@a100`).
Repo: `correct_clasp` (seacove `~/phd_project/correct_clasp`, Jean Zay `$SCRATCH/correct_clasp`).

## 1. Why A100 and not H100

H100 is spent: **24,400 / 20,000 GPU-h (122%)**. A100 is at 5,464 / 15,000 (36%), ~9.5k h
left. All evaluation below runs on A100 `qos_gpu_a100-dev` (2 h wall, 10 submitted jobs
per user). At 100 instances × BoN=8 a task costs 18–45 min on one A100, so every job fits
the dev wall with room to spare and no sharding is needed.

## 2. Evaluation protocol (pinned, not defaulted)

`scripts/evaluate_a100.slurm`. Every generation setting is pinned on the command line,
because the repo's argparse defaults (**bfloat16, T=0.8, top_p=0.95**) match no published
number and would produce a table comparable to nothing.

| Setting | Value | Why |
| --- | --- | --- |
| dtype | `float16` | matches the exact protocol behind the Jiang rows and our frontier table |
| BoN | `N=8`, `generation_batch_size=8` | K=8 in one `generate` call |
| sampling | `T=0.7, top_p=0.9, top_k=50, repetition_penalty=1.0` | released-code lane |
| `max_new_tokens` | `3000` | released-code lane |
| `max_prompt_length` | **`29768` (pinned)** | see §3 — the default silently differs per family |
| seed | `271828` | the seed used for the recorded SFT frontier table |
| slice | `eval[200:300]`, 100 instances/task | see §4 |
| data | `data_rl/<task>/eval/test.json` | passed explicitly; `--stage sft` would otherwise pick the SFT root |

A **single seed (271828) is used for all runs**, which is right for comparability but means
checkpoint differences and sampling noise are not separable within this table. Combined with
n=100 (95% interval on feasibility ≈ ±9 pp), that is the basis for the "do not rank the
checkpoints" instruction in §9.

Metrics are the Jiang-comparable pair, recomputed from raw per-instance records:

- **feasibility** = `best_of_n_instance_feasibility_rate` — instances with ≥1 feasible
  candidate out of 8. *Not* `candidate_feasibility_rate`, which is per-sample and runs
  ~35 points lower; reporting it would make the table meaningless as a comparison.
- **gap** = `mean_best_feasible_gap` — mean over instances where a feasible solution was
  found (conditional). Verified in `_aggregate_records`: infeasible instances are excluded,
  not counted as zero.

## 3. Two defects found in the eval path before running anything

**(a) The checkpoint alias silently changes the prompt template.**
`_nearest_run_dir` (evaluate.py:796) uses `os.path.abspath`, **not** `realpath`. The
convenience alias `$WORK/clasp/models/sft/checkpoint-18400` is a symlink into the real run
directory, but abspath keeps the alias path, whose parent is `.../models/sft` — no
`hyperparams.json`. Stage hyperparameters come back empty, `_stage_uses_no_reason({})`
returns `False`, and the model is evaluated with the **reasoning** template although it was
trained with `alpaca_no_reason`. There is no CLI flag to force `no_reason` back on
(`--force_reasoning_prompt` only turns it off). Mitigation: always pass the original run
path, `.../<run>/checkpoints/checkpoint-N`. The launcher preflight now fails closed if
`hyperparams.json` is not reachable, and the job log prints the resolved template.

**(b) `--max_prompt_length 0` gives the two SFT families different prompt budgets.**
The fallback is `max_seq_length - max_new_tokens`: **5192** for the 50k family
(`max_seq_length=8192`) versus **17000** for the 100k family (`20000`). Measured on the
eval slice, 5192 **truncates 8 MIS prompts** (max MIS prompt = 7898 tokens) — and MIS is
one of the two tasks where we trail Jiang. Truncation guarantees infeasibility, so the
50k family would have been penalised on exactly the metric under study. Pinned to 29768
for every checkpoint; at that budget nothing truncates on any task.

## 4. Slice choice

`--eval_start_index 200 --max_eval_samples_per_task 100` → `eval[200:300]`.

- `eval[0:100]` is Jiang's direct comparison slice and is **algorithmically quarantined**;
  it is kept for one final read, not for iteration.
- `eval[100:150]` is the checkpoint-selection slice.
- `eval[200:300]` is where the existing SFT frontier numbers live, so it doubles as a
  regression check.

**This is not the Jiang first-100 row.** Same frozen benchmark, different rows. Fine as a
development baseline; not a headline claim.

## 5. Pre-flight audits (both clean)

- **Reference-solution audit.** Every row's own ground-truth `output`, scored by our reward
  heads on `eval[200:300]`: **100% feasible, gap 0.000e+00, max gap 0.000e+00** on all five
  tasks. So any gap we measure is the model, not the dataset. This matters most for CVRP,
  whose eval rows 114–299 are known to disagree with `data/cvrp/instances.pkl` — `clasp-eval`
  reads `instance_json`/`instance` from the row itself and never touches the pickle, and the
  rows are internally consistent.
- **Index integrity.** 300 rows per task, only `test.json` (no `*_subset.json` shadowing),
  **zero exact duplicates**, so the loader's dedup does not shift `eval[200:300]` away from
  raw file rows 200–299.
- **Base model.** Both SFT families sit on **byte-identical** Qwen2.5-7B-Instruct weights
  (verified by md5 of `config.json`, `generation_config.json`, and shard 1). No base-model
  confound between our own checkpoints.

## 6. Harness regression gate (passed on all five tasks)

`checkpoint-18400`, exact protocol, A100, ~18 min/task:

| Task | measured feas / gap | recorded frontier | |
| --- | ---: | ---: | :-- |
| TSP | 100% / 2.426% | 100% / 2.385% | MATCH |
| CVRP | 65% / 6.122% | 64% / 5.715% | MATCH |
| OP | **94% / 1.372%** | 94% / 1.331% | MATCH |
| MIS | 73% / 1.727% | 73% / 1.756% | MATCH |
| JSSP | 100% / 13.520% | 100% / 10.876% | MATCH (gap still settling at partial n) |

Feasibility reproduces exactly on every task across a different GPU and dependency stack.
This is what licences the rest of the matrix.

**Provenance note on this row.** A stray drainer process resubmitted the `sft50k_18400`
array after it had already completed; the duplicate was cancelled ~60 s in, but because
`--resume` defaults on and rewrites `run_details__*.jsonl` atomically, it had already
touched the **mis** and **jssp** detail files (mtimes 20:58:03, inside the duplicate's
20:57:13–20:58:12 window; tsp/cvrp/op predate it). Rather than assume the resume logic
reused records correctly, both tasks were re-run into a **fresh inference root** where
`--resume` had nothing to reuse. The clean runs reproduce the originals exactly:

| Task | original | clean re-run | per-instance agreement |
| --- | --- | --- | --- |
| MIS | 72% / 1.727% | 72% / 1.727% | 100/100 feasibility, 100/100 `feasible_candidates` |
| JSSP | 100% / 10.838% | 100% / 10.838% | 100/100 feasibility, 100/100 `feasible_candidates` |

Both rows are therefore uncontaminated. Jobs 587267 / 587268; results under
`inference/sft_baseline_verify_20260804/`.

## 7. Checkpoints under test — and the confounds between them

| Label | Data | Steps / epochs | LoRA | Notes |
| --- | --- | --- | --- | --- |
| `sft50k_16025` | 50k/task | 16025 / 4.32 ep | attn-only, bias `none` | seq 8192 |
| `sft50k_18350` | 50k/task | 18350 / 4.95 ep | attn-only, bias `none` | seq 8192 |
| `sft50k_18400` | 50k/task | 18400 / 4.96 ep | attn-only, bias `none` | paper-era init |
| `sft100k_attn_31250` | 100k/task | 31250 / **1.00 ep** | attn-only, bias `lora_only` | seq 20000 |
| `sft100k_mlp_1950` | 100k/task | 1950 / **0.06 ep** | attn+MLP, bias `lora_only` | barely trained |
| ~~500k/task~~ | — | — | — | **absent**: run dir exists, `checkpoints/` empty (SCRATCH purge) |

**The 50k → 100k comparison is not a clean data-scaling axis.** Moving between the families
also changes epochs (5 → 1), sequence length (8192 → 20000), LoRA bias (`none` →
`lora_only`), optimizer (`adamw_torch_fused` → `adamw_8bit`), weight decay (0 → 0.01), and
the SFT data root. `sft100k_mlp_1950` additionally changes LoRA targets **and** has seen
6% of one epoch — it is an undertrained checkpoint, not a "more data" checkpoint. Any
conclusion of the form "more data helps/does not help" must name which axis moved.

## 8. Structural differences vs Jiang et al.

These are design-level, and no amount of RL tuning removes them:

1. **Specialist vs unified.** Jiang's headline row is one model *per task* (SFT + FOARL,
   BoN=8). Ours is one unified model over five tasks. The comparison is structurally
   unfavourable to us. Jiang's own unified result is appendix-only with no released
   checkpoint or data manifest.
2. **SFT volume.** Jiang: 500k examples/task. Ours: 50k/task (paper-era) or 100k/task —
   5–10× less.
3. **Base model.** Jiang pins Qwen2.5-7B **base** (no chat template, `max_pos` 131072).
   Ours is Qwen2.5-7B-**Instruct**. The reproduction protocol explicitly rejects
   substituting one for the other.
4. **LoRA capacity.** Jiang uses `bias=lora_only` over attention+MLP. Our paper-era
   `checkpoint-18400` is attention-only with `bias=none`.
5. **Context.** Jiang trains at 20000; the 50k family trained at 8192.
6. **MIS contamination.** All 300 public MIS eval instances appear in Jiang's released MIS
   RL training data. Their MIS row (94% / 1.04%) is not a clean target and must be
   disclosed as such whenever it is quoted.

Persistence audit note (2026-08-14): `lora_only` describes the intended
trainable scope, not necessarily the deployed saved policy.  PEFT 0.18.0 and
0.19.1 omit those trained biases from CLASP adapter saves.  Historical CLASP
`lora_only` rows therefore remain descriptive development results and cannot
initialize the corrected publication campaign.  The fresh campaign uses the
scope-equivalent Qwen2.5-7B policy `bias=all` with exact saved-tensor auditing;
see `docs/lora_bias_persistence_remediation_20260814.md`.

## 9. Measured baselines — all five checkpoints, 100 instances/task

Feasibility = instance-level best-of-8; gap = conditional mean. All cells n=100.

| Checkpoint | TSP | CVRP | OP | MIS | JSSP |
| --- | --- | --- | --- | --- | --- |
| `sft50k_16025` (4.32 ep) | 99% / 2.531% | 55% / 5.884% | 96% / 1.861% | 70% / 1.666% | 100% / 11.164% |
| `sft50k_18350` (4.95 ep) | 99% / 2.611% | 61% / 4.795% | 94% / 1.454% | 73% / 2.177% | 100% / 11.075% |
| **`sft50k_18400` (4.96 ep)** | **100% / 2.455%** | **63% / 6.139%** | **94% / 1.372%** | **72% / 1.727%** | **100% / 10.838%** |
| `sft100k_attn_31250` (1.00 ep) | 100% / 2.308% | 58% / 5.154% | 98% / 2.643% | 70% / 1.379% | 100% / 12.390% |
| `sft100k_mlp_1950` (0.06 ep) | 85% / 6.896% | 19% / 8.967% | 93% / 11.863% | 56% / 4.449% | 87% / 29.105% |
| **Jiang (specialist)** | 100% / 1.07% | 100% / 4.53% | 100% / 1.85% | 94% / 1.04% | 100% / 8.20% |

### More SFT data did not help

`sft100k_attn_31250` sees **2× the data** of the 50k family and lands statistically level
with it: CVRP 58% [48, 67] vs 63% [53, 72], MIS 70% [60, 78] vs 72% [63, 80]. At n=100 the
95% interval on feasibility is roughly ±9 points, so **no pair of the four trained
checkpoints is separable on CVRP or MIS feasibility**. Do not rank them.

What the 100k run buys is a slightly better *gap* on TSP (2.308% vs 2.455%) and MIS (1.379%
vs 1.727%) and better OP feasibility (98% vs 94%) — while being *worse* on JSSP gap (12.390%
vs 10.838%). Nothing here is a feasibility breakthrough.

`sft100k_mlp_1950` is far worse on every task, exactly as its 0.06 epochs predict. It is
evidence about undertraining, not about attention+MLP LoRA, and should not be cited as
either.

Within the 50k family the epoch trend flattens: CVRP 55 → 61 → 63, MIS 70 → 73 → 72. More
epochs on the same mixture is converging around CVRP ~60%, MIS ~72%, not climbing toward 100%.

### Where the deficit actually is (best checkpoint, `sft50k_18400`)

| Task | Feasibility deficit | Gap deficit |
| --- | ---: | ---: |
| TSP | 0 pp | −1.39 pp |
| OP | −6 pp | **+0.48 pp (we are better)** |
| JSSP | 0 pp | −2.64 pp |
| **MIS** | **−22 pp** | −0.69 pp |
| **CVRP** | **−37 pp** | −1.61 pp |

**Gaps are close everywhere — within 1–3 points, and OP is already better than Jiang.** The
entire meaningful deficit is CVRP and MIS *feasibility*.

### CVRP and MIS fail for different reasons — measured, not inferred

Distribution of `feasible_candidates` (how many of the 8 draws were feasible) per instance,
`sft50k_18400`:

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | all-8 fail | all-8 pass |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| TSP | 0 | 0 | 4 | 9 | 12 | 18 | 22 | 14 | 21 | 0% | 21% |
| CVRP | **37** | 18 | 11 | 9 | 7 | 9 | 5 | 3 | **1** | 37% | 1% |
| OP | 6 | 6 | 8 | 19 | 8 | 9 | 16 | 13 | 15 | 6% | 15% |
| MIS | **28** | 4 | 4 | 4 | 2 | 3 | 2 | 6 | **47** | 28% | 47% |
| JSSP | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 8 | 88 | 0% | 88% |

The two deficit tasks have **opposite shapes**, and this is the most actionable result here:

- **MIS is bimodal** — 28 instances fail all 8 draws, 47 succeed on all 8, only 25 in
  between. There is a genuine *hard subset*. More sampling or better average quality will
  not touch those 28; they have to be targeted directly.
- **CVRP is not bimodal** — it decays monotonically from 37 at zero, and only **one**
  instance out of 100 gets all 8 draws feasible. CVRP is not a hard-subset problem, it is
  *uniformly marginal*: essentially every instance sits near the feasibility threshold.
  That is encouraging, because a modest uplift in per-sample feasibility should convert a
  large number of instances at once — the 37 zero-instances mostly sit just below a
  threshold that 18 neighbours clear at 1/8.

(An earlier draft inferred within-instance correlation from the pooled candidate rate.
That inference does not hold: a pooled rate over heterogeneous instances mechanically
overpredicts BoN success by Jensen's inequality, independent of any correlation. The
histogram above measures the property directly and supersedes it.)

Crucially this is an **SFT-level** deficit, and the existing RL evidence says RL does not
close it: the 40-step CLASP checkpoint trained from the 100k SFT scores CVRP 66% and MIS
77% on `eval[0:100]` — a few points over the SFT starting point, not 30. Expecting a
40-step RL stage to recover 37 points of CVRP feasibility is not supported by anything
measured so far.

## 10. Implication for the CLASP plan

Two results reframe the plan:

- **Scaling SFT data is not the lever.** 100k/task at 1 epoch ≈ 50k/task at 5 epochs, and
  more epochs on 50k has flattened. Another SFT data-scaling run is the *least* promising
  use of the remaining ~9.5k A100-hours. (The 500k/task checkpoint that would have tested
  this further no longer exists.)
- **Gaps are already competitive; only CVRP/MIS feasibility is not.** The objective is
  narrow and well-posed: move CVRP and MIS instances from all-8-infeasible to
  at-least-1-feasible.

  Note this will not be free. The feasibility/gap tension is already visible in our own
  data: `16025 → 18400` gains 8 points of CVRP feasibility (55% → 63%) while the CVRP gap
  *worsens* from 5.884% to 6.139%. Newly-feasible instances tend to be the hard ones, and
  they enter the conditional mean at a poor gap. That trade is precisely what CLASP's
  feasibility-gated, Pareto-anchored advantage exists to manage, so it strengthens rather
  than weakens the case — but any plan that assumes feasibility gains at zero gap cost is
  assuming a free lunch our own checkpoints do not show.

In priority order:

1. **Diagnose the two failure modes further — cheap and CPU-only.** The per-instance
   records are already written. For MIS, characterise the 28 all-8-fail instances (size,
   which constraint fails via `constraint_fraction` and the per-task checkers): that is a
   discrete, targetable set. For CVRP, the question is different — since the whole
   distribution is marginal, find out *which* constraint dominates (capacity vs route
   structure vs depot handling), because a single systematic error would explain a
   uniformly-shifted distribution.
2. **Then run CLASP on CVRP and MIS.** Its feasibility-gated advantage is the mechanism
   for buying feasibility at controlled gap cost, and CVRP's uniformly-marginal shape is
   the favourable case for it — a modest per-sample uplift should convert many instances
   at once. Running these as single-task rather than unified is worth *testing*, on the
   hypothesis that the unified model pays a capacity tax; note we have **no measurement of
   a specialist-vs-unified delta on our own models** (every checkpoint here is unified),
   so that is a hypothesis borrowed from Jiang's setup, not an established mechanism.
   For single-task runs `--clasp_feasibility_warmup_targets` must be narrowed to the
   active task or the validator rejects the run.
3. **Only then consider SFT changes**, targeted at the diagnosed failure mode
   (e.g. CVRP/MIS-weighted or hard-instance-weighted mixtures) rather than raw volume.
4. **Keep the comparison honest**: label it unified-vs-specialist, note the base-model
   difference, and flag the MIS contamination every time the 94% target is quoted.

Budget note: the whole 25-job matrix cost roughly 10 A100-hours, so this diagnostic loop is
cheap relative to the ~9.5k remaining.

## 11. Artifacts

| Path (Jean Zay) | Contents |
| --- | --- |
| `$SCRATCH/correct_clasp/scripts/evaluate_a100.slurm` | A100 dev launcher, protocol pinned |
| `$SCRATCH/correct_clasp/audit_reference_solutions.py` | reference-solution + index audit |
| `$SCRATCH/correct_clasp/prompt_lengths.py` | prompt-budget/truncation check |
| `$SCRATCH/correct_clasp/aggregate_baseline.py` | recomputes the table from raw records |
| `$SCRATCH/correct_clasp/inference/sft_baseline_eval100_20260804/` | all run outputs |
| `$SCRATCH/correct_clasp/audit/reference_solutions_eval200_100.json` | audit receipt |

**These live on SCRATCH, which Jean Zay purges on access age** — the same policy that
destroyed the 500k/task checkpoint this study wanted to test. `$WORK` is over quota, so
the durable copy is on **seacove**:

| Path (seacove) | Contents |
| --- | --- |
| `~/phd_project/correct_clasp/results/sft_baseline_20260804/` | 251 run summaries + per-task CSVs + `baseline_table.json` (5.4 MB) |
| `~/phd_project/correct_clasp/docs/sft_baseline_findings.md` | this document |
| `~/phd_project/correct_clasp/{aggregate_baseline,audit_reference_solutions,prompt_lengths}.py` | analysis scripts |
| `~/phd_project/correct_clasp/scripts/evaluate_a100.slurm` | the launcher |

The raw `run_details__*.jsonl` (full candidate texts) were left on SCRATCH — they are large,
and everything reported here is recomputable from the summaries. If the per-candidate
diagnosis in §10 step 1 is wanted, pull those before they age out.
