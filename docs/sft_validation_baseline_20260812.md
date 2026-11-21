# SFT validation baseline for CLASP — 2026-08-12

This report records the decision-grade SFT baseline used to choose the next CLASP
experiments.  It compares checkpoints only when they were decoded by the same evaluator,
on the same examples, with the same random seed and generation settings.

## Evaluation contract

- Hardware/queue: one Jean Zay A100 per task, `qos_gpu_a100-dev`, two-hour limit.
- Development slice: exactly 100 rows per task, `eval[200:300]`.
- Decoding: raw model outputs, best-of-8 in one `generate` call, fp16, seed 271828,
  temperature 0.7, top-p 0.9, top-k 50, repetition penalty 1.0, 3000 new tokens, and
  a pinned 29768-token prompt budget.
- Feasibility: fraction of instances with at least one feasible completion among eight.
- Gap: best feasible gap per instance, averaged only over instances where a feasible
  completion exists.  Gap therefore must not be used to reward a model for failing on
  difficult instances.
- Selection rule: feasibility first; conditional gap is the secondary/Pareto criterion.

The development rows are clean: the 2026-08-12 re-audit found 100 distinct rows per task,
100% feasible reference solutions, and exactly zero reference gap for all five tasks.  The
public `eval[0:100]` rows remain quarantined for the final comparison; `eval[100:150]` has
already been used for checkpoint selection and is not an untouched validation set.

## Checkpoints

The historical matrix already covers the old unified 50k/task checkpoints (steps 16025,
18350, and 18400), unified attention-only 100k/task checkpoints (steps 9400, 31250,
108550, and 134550), and the undertrained attention+MLP checkpoint 1950.

The new matrix adds:

- unified attention-only 100k/task checkpoint 137500, at epoch 4.40;
- TSP specialist, 500k examples, one epoch, checkpoint 31250;
- CVRP specialist, 100k examples, one epoch, checkpoint 6250;
- CVRP specialist, 500k examples, one epoch, checkpoint 31250.

The specialist models use completion-only SFT, the no-reason Alpaca template, a 20k
training sequence length, rank/alpha 64 LoRA over attention and MLP projections, and the
same Qwen2.5-7B-Instruct base.  The CVRP 100k-versus-500k comparison is the cleanest data
scaling comparison available: task, architecture, one-epoch schedule, optimizer, prompt,
and context are held fixed.  Specialist-versus-unified is still confounded by LoRA targets
and training mixture and should be interpreted as an end-to-end model-design comparison.

## Historical frontier on `eval[200:300]`

Each cell is `best-of-8 feasibility / conditional gap`.

| checkpoint | TSP | CVRP | OP | MIS | JSSP |
| --- | ---: | ---: | ---: | ---: | ---: |
| unified 50k, step 16025 | 99% / 2.531% | 55% / 5.884% | 96% / 1.861% | 70% / 1.666% | 100% / 11.164% |
| unified 50k, step 18350 | 99% / 2.611% | 61% / 4.795% | 94% / 1.454% | 73% / 2.177% | 100% / 11.075% |
| unified 50k, step 18400 | 100% / 2.455% | 63% / 6.139% | 94% / 1.372% | 72% / 1.727% | 100% / 10.838% |
| unified 100k attn, step 9400 | 97% / 6.165% | 22% / 7.116% | 86% / 4.979% | 55% / 2.843% | 100% / 20.656% |
| unified 100k attn, step 31250 | 100% / 2.308% | 58% / 5.154% | 98% / 2.643% | 70% / 1.379% | 100% / 12.390% |
| unified 100k attn, step 108550 | 100% / 2.008% | 64% / 6.012% | 99% / 1.041% | 78% / 1.767% | 100% / 9.596% |
| **unified 100k attn, step 134550** | **100% / 1.636%** | **72% / 5.238%** | **99% / 1.002%** | **81% / 1.731%** | **100% / 8.499%** |
| unified 100k attn+MLP, step 1950 (0.06 epoch) | 85% / 6.896% | 19% / 8.967% | 93% / 11.863% | 56% / 4.449% | 87% / 29.105% |
| Jiang et al. specialist target | 100% / 1.070% | 100% / 4.530% | 100% / 1.850% | 94% / 1.040% | 100% / 8.200% |

Step 134550 is the existing unified SFT frontier: macro feasibility is 90.4% and macro
conditional gap is 3.621%.  The early attention+MLP checkpoint is evidence of severe
undertraining, not evidence that MLP LoRA is harmful.

## New held-out matrix

All jobs completed with exit code 0.  Each cell is `best-of-8 feasibility / conditional
gap`; candidate feasibility is included separately because it explains how much headroom
remains inside an eight-sample group.

| checkpoint | task | result | candidate feasibility | versus Jiang |
| --- | --- | ---: | ---: | ---: |
| unified 100k attn, step 137500 | TSP | 100% / 1.364% | 79.0% | 0 pp / +0.294 pp gap |
| unified 100k attn, step 137500 | CVRP | 67% / 5.737% | 25.9% | -33 pp / +1.207 pp gap |
| unified 100k attn, step 137500 | OP | 99% / 0.961% | 72.4% | -1 pp / **-0.889 pp gap** |
| unified 100k attn, step 137500 | MIS | 79% / 1.462% | 64.8% | -15 pp / +0.422 pp gap |
| unified 100k attn, step 137500 | JSSP | 100% / 8.961% | 98.6% | 0 pp / +0.761 pp gap |
| specialist 500k, step 31250 | TSP | **100% / 0.984%** | 89.1% | 0 pp / **-0.086 pp gap** |
| specialist 100k, step 6250 | CVRP | 64% / 6.635% | 29.9% | -36 pp / +2.105 pp gap |
| specialist 500k, step 31250 | CVRP | **93% / 4.768%** | 52.8% | -7 pp / +0.238 pp gap |

The unified 137500 macro result is 89.0% feasibility and 3.697% conditional gap, versus
90.4% and 3.621% at unified checkpoint 134550.  In a paired comparison, 137500 loses five
CVRP feasibility points (10 instances rescued only by 134550, five only by 137500), loses
two MIS points, holds OP/TSP/JSSP feasibility, improves TSP/OP/MIS gap, and worsens CVRP and
JSSP gap.  The CVRP and MIS feasibility changes are not individually significant at n=100,
but 134550 Pareto-dominates 137500 on the macro pair and is the better unified CLASP
initialization.

The clean CVRP scaling comparison is decisive.  At fixed one-epoch specialist training,
500k examples improve feasibility from 64% to 93% (+29 pp; exact paired McNemar
`p=3.7e-9`) and gap from 6.635% to 4.768%.  On the 64 instances feasible for both models,
the 500k model reduces gap by 2.801 pp.  Relative to unified 134550, the 500k specialist
adds 21 feasibility points (`p=5.7e-6`) and improves gap by 0.941 pp on their 71 common
feasible instances.  Specialization alone at 100k does not help; Jiang-scale, task-specific
data does.

The 500k TSP specialist also reaches 100% / 0.984%, better than Jiang's reported
100% / 1.070% target on this development slice.  This is a development result, not a
headline benchmark claim; `eval[0:100]` remains sealed.

## Diagnosed feasibility barriers

At unified checkpoint 134550, CVRP reaches 72% instance feasibility but only 27.5%
candidate feasibility.  Among infeasible candidates, 81.0% violate capacity, 43.6% miss
customers, and 35.5% duplicate customers.  Every one of the 28 all-eight-failed instances
contains a capacity violation; all 28 also contain a customer-uniqueness failure somewhere
in the group, and 27 contain a coverage failure.  The all-eight failure rate rises sharply
with problem size: 0/11 at size <=25, 1/30 at 26--50, 7/29 at 51--75, and 20/30 at size
>=76.

MIS reaches 81% instance and 62.5% candidate feasibility.  Independence conflicts occur
in 98.7% of infeasible candidates and in every one of the 19 all-eight-failed instances.
Its all-eight failure rate is 0% through size 50, 8/31 at 51--75, and 11/24 at size >=76.

This localizes the SFT barrier: the remaining feasibility deficit is not generic formatting
or decoding failure.  It is long-horizon constraint maintenance on large CVRP and MIS
instances.  CVRP needs capacity-safe partitioning plus exact one-time customer coverage;
MIS needs conflict-free selections on the hard, larger graphs.

The 500k CVRP specialist eliminates every all-eight failure through size 75.  Its seven
remaining failures are all in the size >=76 band, and all seven contain capacity, coverage,
and customer-uniqueness failures.  Thus the next CVRP stage has a sharply defined target:
the largest instances, not the whole size distribution.

## Comparison caveats

Jiang's headline models are task specialists built from Qwen2.5-7B base, whereas these
unified checkpoints use Qwen2.5-7B-Instruct.  Their public MIS RL training data also
contains the public MIS evaluation instances, so the reported 94% MIS feasibility is not
a clean uncontaminated target.  These facts do not invalidate the development target, but
they prevent a strict architecture-attribution claim.

## Decision rule for the next CLASP experiments

1. Choose the unified initialization lexicographically by per-task feasibility, then gap;
   do not pick a lower-gap checkpoint that merely fails on more hard instances.
2. Use the 500k CVRP specialist as the CVRP CLASP initialization.  It is already at 93%
   feasibility and 4.768% gap, and the seven residual failures are confined to size >=76.
   Use unified 134550 for a unified regression/control lane, not unified 137500.
3. Run task-focused CLASP pilots for CVRP and MIS, emphasizing the existing exact component
   violations.  In all-infeasible groups, CLASP should rank only violation reduction; in
   mixed groups, every feasible completion must dominate every infeasible completion; only
   fully feasible groups should optimize gap without a feasibility trade.
4. Stratify checkpoint-selection reports by size and track the zero-feasible, mixed, and
   all-feasible best-of-8 group rates.  A mean reward or conditional gap alone can conceal
   regressions on the large hard instances.
5. Keep TSP, OP, and JSSP in the unified regression gate, but spend the first RL budget on
   CVRP capacity/coverage and MIS independence.  They account for nearly all of the current
   feasibility deficit.

## Artifacts

- Jean Zay matrix: `$SCRATCH/correct_clasp/inference/sft_validation_matrix_eval200_20260812/`
- Re-audit receipt: `$SCRATCH/correct_clasp/audit/reference_solutions_eval200_100_reaudit_20260812.json`
- Preserved checkpoint snapshot: `.../seed42/eval_snapshots/validation_eval200_20260812_ckpt137500/`
- Launcher: `scripts/evaluate_a100.slurm`
- Failure analyzer: `analyze_sft_failures.py`
- Aggregate metrics: `results/sft_validation_20260812/aggregate_eval200.json`
- Paired tests: `results/sft_validation_20260812/paired_*.json`
