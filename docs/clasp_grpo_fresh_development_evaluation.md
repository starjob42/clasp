# Fresh GRPO/CLASP development evaluation

`scripts/evaluate_clasp_grpo_fresh_dev_a100.slurm` runs one checkpoint on one
A100 under the two-hour development QoS. It uses the frozen v30 evaluator and
the independent CVRP/MIS `data_v3` benchmark. The management wrapper and its
standard-library helper live outside v30 and record their own file hashes.

Required environment variables are `FRESH_SOURCE`, `FRESH_MANAGEMENT_ROOT`
(containing the two new files under `scripts/`), `FRESH_RUNTIME`,
`FRESH_DATA_ROOT`, `FRESH_CHECKPOINT`, `FRESH_OUTPUT`, `FRESH_TASK` (`cvrp` or
`mis`), and `FRESH_METHOD` (`sft`, `grpo_z`, `v3`, or `v4`). RL methods also
require `FRESH_TRAIN`, the certified seed-43, step-100 training directory.
SFT must be the exact task checkpoint-31250 used to initialize all methods.

The wrapper refuses to launch until the completed zero-semantic-overlap
receipt is bound to the benchmark manifest and all six SFT/RL/public-eval
source scans. Dataset and reference hashes are verified, along with every MIS
native solver certificate and reference feasibility. Every method uses the
same 100 instances, raw eight-candidate generation, seed 271828, float16,
temperature 0.7, top-p 0.9, top-k 50, 29,768 prompt tokens, and 3,000 completion
tokens. No model-output repair is applied. The launch receipt and completed
result index are immutable and bind each instance and reference answer.

SFT uses the `jiang_f638_sft` representation profile and RL uses
`jiang_f638_clasp`, matching their certified stage metadata. Their effective
CVRP/MIS prompts are identical; unused TSP flags differ (empty strings versus
`0`). The strict comparator is unchanged. SFT comparisons are descriptive if
those metadata differences yield unequal protocol fingerprints.

This fresh-data protocol supersedes the earlier plan to evaluate the new
seed-43 policies on public `eval[100:200]`. The change was fixed before any
policy outputs on the fresh benchmark. Historical public-slice results remain
historical diagnostics; the fresh benchmark supplies the new comparison table.
Reference output text is parsed and checked against the exact certified MIS
set or verified CVRP routes and the shared verifier's objective arithmetic.

Report CVRP **gap to a verified heuristic reference**, because its reference
routes have no optimality certificate. Report MIS **optimality gap**, because
all 100 incumbent cardinalities equal their native certified upper bounds.
Read conditional best-feasible gap together with pass@8 and candidate
feasibility. This remains a development experiment, not confirmation evidence.
The headline and strict paired gaps use the frozen evaluator's zero-clipped
directional relative gap. Supplementary signed mean gaps and counts beating
the reference are descriptive and have no additional confidence intervals.
Table rates and gaps are percentages; strict comparison JSON uses fractions.
Size strata use the evaluator's instance size, including the CVRP depot.

Use `resume_clasp_grpo_fair_dev.py --training-only` for remaining training.
The separate `queue_clasp_grpo_fresh_dev.py freeze --training-root ... --data ...
--management ... --output ...` command creates the immutable fresh-evaluation
addendum after the benchmark audit passes. It fixes eight evaluations, all
checkpoint paths, decoding, metric definitions, paired comparisons, size bands,
and bootstrap settings before any fresh outputs. It preserves the original
training protocol and binds the summary and strict comparison scripts by hash.

Then run `queue_clasp_grpo_fresh_dev.py run --output ... --user uag72kz --submit
--max-new-jobs 2` periodically. Without `--submit`, it only reports status. The
manager counts expanded array tasks against the ten-slot development QoS,
waits for completed training and certified terminal checkpoints, and records
each submission in an immutable event file. It never cancels unrelated jobs,
deletes partial outputs, or schedules the superseded public-slice evaluations.
