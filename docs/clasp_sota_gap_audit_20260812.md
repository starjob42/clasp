# Why CLASP is not yet beating LLMCoSolver — audit and experiment plan

Date: 2026-08-12

## Bottom line

The present comparison does **not** isolate CLASP versus Jiang et al.'s FOARL.  Most of the
observed deficit is already present before RL because the earlier CLASP models use a unified
Qwen2.5-7B-Instruct policy, 50k or 100k examples per task, and often attention-only LoRA.
Jiang's headline system uses a Qwen2.5-7B base **specialist per task**, 500k unique
solver-labelled examples per task, attention+MLP LoRA, one additional FOARL pass over at most
3,200 examples, and verifier-selected best-of-8 inference.

The controlled SFT evidence makes this causal ordering unusually clear:

- fixed one-epoch CVRP specialist training improves from `64% / 6.635%` at 100k examples to
  `93% / 4.768%` at 500k examples;
- the 500k TSP specialist reaches `100% / 0.984%` on the development slice, already beyond
  Jiang's reported `100% / 1.070%` target there;
- repeating the 100k/task unified data to later epochs does not reproduce this effect: unified
  step 137500 is Pareto-worse than step 134550.

So the first-order blocker is **candidate support from SFT**, particularly long-horizon
constraint maintenance on large CVRP and MIS instances.  CLASP then adds a second blocker:
its current default exact-BoN leave-one-out advantage is extremely sparse and tie-prone, and
several advertised controls are bypassed or wired to the wrong metric.  There has not yet been
a clean, corrected, specialist-initialized, equal-budget FOARL-versus-CLASP experiment.

## Decision-grade held-out baseline

The exact evaluation contract and complete raw artifact ledger are in
[`sft_validation_baseline_20260812.md`](sft_validation_baseline_20260812.md).  The development
slice is `eval[200:300]`, exactly 100 distinct rows per task, raw best-of-8, one `generate`
call, seed 271828, fp16, temperature 0.7, top-p 0.9, top-k 50, 3,000 new tokens, and no repair
or reranking.  The reference re-audit found 100% feasible labels and zero reference gap.

Each cell below is instance-level best-of-8 feasibility / conditional best-feasible gap.
Conditional gap is never interpreted without feasibility.

| Checkpoint | TSP | CVRP | OP | MIS | JSSP |
| --- | ---: | ---: | ---: | ---: | ---: |
| unified 100k attention, step 134550 | 100 / 1.636 | 72 / 5.238 | 99 / 1.002 | 81 / 1.731 | 100 / 8.499 |
| unified 100k attention, step 137500 | 100 / 1.364 | 67 / 5.737 | 99 / 0.961 | 79 / 1.462 | 100 / 8.961 |
| specialist TSP 500k, step 31250 | **100 / 0.984** | — | — | — | — |
| specialist CVRP 100k, step 6250 | — | 64 / 6.635 | — | — | — |
| specialist CVRP 500k, step 31250 | — | **93 / 4.768** | — | — | — |
| Jiang SFT+RL+BoN reported target | 100 / 1.070 | 100 / 4.530 | 100 / 1.850 | 94 / 1.040 | 100 / 8.200 |

The clean CVRP 100k-to-500k comparison gives +29 feasibility points (`p=3.7e-9`, exact paired
McNemar) and -1.867 gap points.  On the 64 instances feasible for both, gap improves by 2.801
points.  This is much stronger evidence than a cross-architecture or cross-RL comparison.

The remaining failures are concentrated rather than generic:

- unified 134550 CVRP: 28 all-eight failures; 20/30 large (`n>=76`) instances fail.  Among
  infeasible candidates, 81.0% violate capacity, 43.6% omit customers, and 35.5% duplicate
  customers;
- unified 134550 MIS: 19 all-eight failures, all with independence conflicts; no all-eight
  failure through size 50, then 8/31 at size 51–75 and 11/24 at size 76+;
- CVRP 500k reduces the failures to seven, all at size 76+, all involving capacity plus
  coverage/uniqueness errors.

This is the signature of an autoregressive construction failure on hard, long instances—not
primarily a parser or prompt-truncation failure.

## What Jiang's result actually contains

The primary references are the [LLMCoSolver paper](https://arxiv.org/html/2509.16865) and
[official repository](https://github.com/Summer142857/LLMCoSolver).  The paper explicitly
uses 500k SFT instances per task, at most 3,200 FOARL instances, and verifier-selected BoN=8.
Its own ablation is important:

| Task | SFT greedy | SFT+RL greedy | SFT+RL+BoN |
| --- | ---: | ---: | ---: |
| TSP | 89 / 2.30 | 91 / 2.32 | 100 / 1.07 |
| OP | 54 / 2.32 | 92 / 4.25 | 100 / 1.85 |
| CVRP | 59 / 6.02 | 80 / 8.27 | 100 / 4.53 |
| MIS | 80 / 1.71 | 83 / 1.34 | 94 / 1.04 |
| JSSP | 100 / 11.01 | 100 / 10.94 | 100 / 8.20 |

FOARL mainly improves candidate feasibility; it actually worsens greedy gap on OP and CVRP.
The final quality jump comes largely from drawing eight candidates and selecting the best
feasible one.  A greedy CLASP number is therefore not comparable to Jiang's headline number.

Important reproduction caveats:

- the paper says BoN top-p 0.7, while released `eval.py` defaults to temperature 0.7 and
  top-p 0.9 (with the Transformers top-k default); both variants should be reported;
- Jiang reports the best of three independent LLM experiments rather than only mean/std;
- released CVRP/MIS validators are weaker than the strict CLASP validators (for example,
  duplicate/range checks), so both systems should ultimately be rescored by the same hardened
  verifier;
- released MIS RL data contain the 300 public MIS evaluation instances.  The published MIS
  number is therefore not a clean target for an uncontaminated run;
- these CLASP specialists still use Qwen2.5-7B-Instruct, whereas Jiang's headline specialists
  start from Qwen2.5-7B base.  Call them "Jiang-scale" rather than exact reproductions.

## Highest-confidence CLASP defects

### 1. The default BoN advantage is too sparse for gap optimization

`CLASP_BON_ALIGNED` defaults on.  Its exact-temperature-zero objective assigns each completion
the leave-one-out marginal contribution to the set maximum.  In a successful group with two
or more feasible candidates, only a unique best candidate gets nonzero quality credit; all
non-best feasible candidates and noncritical infeasible candidates get zero.  Best-quality
ties produce zero signal.  The production smoke audit measured a 75% best-quality tie rate
and a 58.3% zero-quality-signal rate among successful groups.

This is especially damaging for discrete MIS/JSSP objectives and duplicate generations.  It
also matches the broader finding that GRPO can sharpen common modes while suppressing rare
correct trajectories ([Rewarding the Unlikely](https://aclanthology.org/2025.emnlp-main.1298/)).
Exact set-objective alignment is not useful if most successful groups carry no gradient.

### 2. SFT support is absent in precisely the groups where RL needs it

In an all-infeasible K=8 group, CLASP can rank violation magnitudes but has no feasible anchor.
For large CVRP it must simultaneously discover capacity-safe partitioning, exact coverage, and
no duplicates; for MIS it must undo many interacting conflicts.  Reweighting cannot choose a
Jiang-quality candidate that the generator never samples.  The 500k CVRP scaling result is the
direct empirical proof.

Dynamic resampling of zero/equal groups and maintaining diversity are supported by
[DAPO](https://dapo-sia.github.io/); response-length and group-standard-deviation biases are
documented by [Dr.GRPO](https://arxiv.org/html/2503.20783).  These should be measured, not
assumed away.

### 3. The raw-feasibility preset was internally incompatible

`--clasp_raw_feasibility_targeting` defaulted to a 500-step feasibility warmup, while the
trainer independently defaulted `CLASP_BON_ALIGNED=1` and rejects BoN mode with any positive
warmup.  The documented preset could therefore abort unless a launcher manually disabled BoN.
This audit fixes the preset by explicitly selecting the strict three-regime branch.

### 4. Feasibility warmup gates on candidate feasibility, not best-of-8 feasibility

The warmup release and checkpoint gate use per-candidate feasibility targets around
98.5–99.5%, although the benchmark goal is whether **any** of eight candidates is feasible.
Historically, checkpoint 2530 already had 99.4% BoN feasibility at only 92.1% candidate
feasibility.  Holding feasibility-only training until 98.5% candidate feasibility suppresses
objective learning long after the deployment constraint is met; during warmup, fully feasible
groups receive zero objective advantage.  This is a direct mechanism for the observed
high-feasibility/high-gap frontier and should be corrected before the next raw-preset run.

### 5. Some advertised mechanisms do not affect the default policy gradient

- The per-component "proper dual" updates and logs component lambdas, but default BoN
  advantages consume raw positive violations plus a separate task-level set controller.  The
  component lambdas are therefore a policy no-op in that path.
- The legacy "Pareto anchor" is diagnostic: its own implementation notes that group centering
  cancels additive anchor terms.  Default BoN mode bypasses it anyway.
- The adaptive per-task completion length uses the maximum task budget for a mixed local
  batch, so the longest task dominates; it is only truly per-task in specialist/homogeneous
  batches.

These are not necessarily bad ablations, but they cannot be claimed as active CLASP
contributions in the default configuration.

### 6. Historical experiments do not constitute a clean algorithm comparison

Earlier CLASP runs demonstrate a real feasibility/gap tradeoff: feasibility-heavy checkpoints
reach roughly 99.4% BoN feasibility with about 9.3% gap, while a true-gap lane reaches about
4.9% gap but falls to 95.6% feasibility.  However, many runs predate the correction that moves
the BoN multiplier at optimizer boundaries; one optimizer step could mix two Lagrangians.
The planned matched specialist corrected-CLASP jobs were cancelled before training.  There is
therefore no valid evidence yet that corrected CLASP loses to matched FOARL.

### 7. Evaluation and launch wiring could select the wrong scientific conclusion

The audit found and fixed three high-confidence issues:

- `best_run` selected only the conditional gap, which rewards a checkpoint for failing hard
  instances; it now selects lexicographically by BoN feasibility, then conditional gap;
- checkpoint symlinks could lose the parent `hyperparams.json` and silently choose the wrong
  prompt template; run resolution now follows the real checkpoint path;
- task representation switches, notably MIS `Set:` versus `Response:`, were not persisted and
  replayed; new SFT metadata records them and evaluation restores them unless explicitly
  overridden.

A paired A100 check then evaluated both unified checkpoints with the alternate Jiang-style
`Response:` instruction, holding all other settings and the 100 examples fixed.  It did not
rescue MIS: step 134550 moved from `81% / 1.731%` (`Set:`) to `78% / 1.732%` (`Response:`),
and step 137500 moved from `79% / 1.462%` to `77% / 1.600%`.  The paired feasibility changes
were not significant (`p=0.375` and `p=0.688`).  Thus the main baseline remains valid and
step 134550 remains the better unified initialization; prompt-prefix mismatch was a real
provenance risk, but not the source of the observed MIS deficit.

Existing result records were not rewritten.  The generic non-A100 launchers still require an
audit before scientific use: they do not pin the Jiang decoding/protocol controls and one GRPO
launcher can truncate long MIS prompts.

## Jean-Zay evaluation state

Completed raw artifacts:

```text
/lustre/fsn1/projects/rech/vas/uag72kz/correct_clasp/inference/
  sft_validation_matrix_eval200_20260812/
```

The reproducible launcher is `scripts/evaluate_a100.slurm`.  It uses one A100 per task under
`qos_gpu_a100-dev`, a two-hour limit, and the pinned 100-row protocol above.

Training checkpoints still pending on Jean-Zay:

- unified 100k/task attention-only continuation to step 156250 (epoch 5), chain ending at job
  600563;
- 500k attention+MLP specialists for OP (873904→873905), MIS (873906→873907), and JSSP
  (873908→873909).

Dependency evaluations are already registered and will start only after successful training:

| Eval job | Target | Dependency |
| --- | --- | --- |
| 879214 | OP specialist 500k, 100 OP rows | after 873905 |
| 879215 | MIS specialist 500k, 100 MIS rows | after 873907 |
| 879216 | JSSP specialist 500k, 100 JSSP rows | after 873909 |
| 879217 `[0-4]` | unified step 156250, 100 rows/task | after 600563 |

All were pending on dependencies at the time of this audit.  The paired MIS prompt-contract
jobs 879561 and 879564 both completed successfully in 12m35s; their raw outputs are under
`inference/sft_validation_prompt_contract_20260812/`.  A separate hard-only 50k SFT was
found and is not worth rerunning: it achieved only 38.8% macro feasibility.

Operationally, A100 allocation is healthy (about 9,013 of 15,000 GPU-hours remained), but WORK
inode use is 84.6%; raw JSONL and checkpoint retention need active housekeeping.  SCRATCH is
purgeable, so final summaries and selected checkpoints should remain on durable WORK/STORE.

## Experiment ladder that can answer the research question

### Stage A — finish the SFT frontier

1. Let the queued OP/MIS/JSSP 500k specialists and unified epoch-5 checkpoint finish and use the
   dependency evaluations above.
2. Choose each task's initialization feasibility-first, then gap.  Current choices are TSP500k,
   CVRP500k, and unified134550 only as the unified control.
3. Do not spend RL on TSP unless the sealed set reverses the development result; it is already
   at `100 / 0.984`.

### Stage B — run the missing matched RL ablation

For CVRP and MIS first, use the same 500k specialist initialization and the same clean RL rows,
K=8, optimizer, token budget, KL, clipping, number of examples, and three seeds:

1. released-style scalar FOARL/standard GRPO;
2. strict three-regime CLASP: violation-only for all-infeasible groups, strict feasible
   dominance for mixed groups, dense objective ranking for all-feasible groups;
3. dense/tie-safe BoN CLASP: smooth/rank frontier credit for all feasible candidates plus
   dynamic resampling of zero-signal groups.

Do not combine representation changes, repair, curriculum, and advantage changes in this first
A/B/C.  It must identify the algorithm effect.

Required training diagnostics per task and size band:

- zero-feasible, mixed, and all-feasible group rates;
- candidate feasibility and `pass@8` feasibility;
- all-equal reward and zero-quality-signal rates;
- reward-component variance and advantage mass;
- completion entropy, unique-solution rate, truncation, and token length;
- conditional gap only alongside feasibility, plus `feasible AND gap <= {1,5,10}%` over all
  instances.

[MO-GRPO](https://arxiv.org/html/2509.22047) supports normalizing objective components before
combining them when their variances differ.  Preference training such as
[BOPO](https://proceedings.mlr.press/v267/liao25a.html) is a strong follow-up if on-policy CLASP
continues to erase rare high-quality candidates.

### Stage C — target the remaining structural failures

After the clean algorithm ablation:

- stratify/oversample CVRP and MIS size 76+ while retaining easier examples to prevent the
  observed hard-only collapse;
- test grammar/action masking for IDs, uniqueness, coverage, and route delimiters;
- test deterministic projection/repair and local search as a **separate** system lane, always
  reporting raw versus repaired results;
- use symmetry augmentation/canonicalization and multiple near-optimal labels per instance
  ([POMO](https://papers.nips.cc/paper_files/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html),
  [Sym-NCO](https://arxiv.org/abs/2205.13209));
- for JSSP, compare free-form schedules with a compact priority/order representation decoded by
  a deterministic schedule generator, as motivated by
  [STARJOB](https://arxiv.org/html/2503.01877).

### Stage D — final evaluation

Keep `eval[0:100]` sealed until the model/decoder is frozen.  Then report:

1. the exact Jiang first-100 protocol with both paper top-p 0.7 and released-code top-p 0.9;
2. a clean independent in-distribution set;
3. OOD sizes and standard benchmark families where labels and solver settings are auditable.

One hundred examples is adequate for development but not a top-feasibility claim.  Use at least
500–1,000 final instances, paired bootstrap/Wilson intervals, and three seeds.  Report mean/std
and separately the best-of-three convention used by Jiang.  Also plot equal-budget curves for
K in `{1,2,4,8,16,32}`; feasibility and gap should be compared at the same token and wall-clock
budget.

## Code verification

The evaluator/provenance fixes and the raw-preset compatibility fix pass the complete local
suite in the configured research environment: 180 tests passed (five third-party/runtime
warnings).  No historical result files were rewritten.  The repository has no usable Git
metadata locally or on Jean-Zay, so this report, the test receipt, file hashes, and remote paths
are the available provenance trail.
