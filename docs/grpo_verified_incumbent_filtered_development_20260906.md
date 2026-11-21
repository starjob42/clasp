This pilot tests ordinary GRPO with graded violations and incumbent-filtered,
verified improvement supervision. It includes the separately verified TRL
checkpointing restoration fix. It is a development experiment, with no
established superiority or SOTA claim before the paired evaluations complete.

The existing per-completion teacher rule remains available. For the new arm,
`GRPO_VERIFIED_TEACHER_SELECTION=best_per_prompt` collects all raw candidates
and eligible verified targets across ranks before TRL shuffles its rollout
buffer. It recomputes objectives with the task verifier, without reference
solutions. Rows without a teacher still contribute their raw feasible solution
to the incumbent.

For each distinct task/prompt/instance, keep one best teacher only when it
strictly improves the best raw feasible solution. If every raw solution fails,
keep the best verified repair. Equal-quality and dominated targets are
excluded; duplicate targets receive one weight. CVRP uses a relative comparison
tolerance of 1e-9, while MIS cardinality comparisons are exact. Deterministic
text/row-ID tie breaking chooses between equal best targets. Original rollout
tokens, rewards, advantages, and old-policy probabilities remain unchanged.

Example: if the best raw CVRP cost is 100, improving a weaker sample from 150
to 120 is insufficient for supervision. A verified teacher costing 95 can be
selected. When every raw candidate is invalid, feasible repairs costing 110
and 130 produce the single 110-cost target. This prevents the auxiliary loss
from promoting inferior repairs merely because their source was invalid.

The loss remains `GRPO + 0.05 * mean_target_NLL`: token means within complete
targets, then the pooled eligible-target mean across ranks in each microbatch,
with the existing gradient accumulation scaling. Ranks with no local target
still participate in the teacher forward when another rank has a target.
No inference-time search, repair, or constrained decoding is added.

The motivating offline audit evaluated existing operators on the first 20
already-inspected development instances per task (eight retained raw candidates
each). It found 103/134 CVRP teachers inferior to their prompt's raw incumbent.
Filtering retained 14 CVRP improvements and six MIS targets (four improvements,
two recoveries of entirely invalid groups), excluding every dominated target.
These are operator diagnostics, not model performance or training-set estimates.

The fresh comparison uses classic GRPO and the new modified arm on CVRP and MIS,
each initialized from the identical task-specific certified SFT checkpoint
31250. Both arms use the same fixed source/runtime, seed/data seed 43, 100 fresh
updates, B9, global 36 completions/update, LoRA, optimizer, learning-rate
schedule and generation limits. The modified arm enables both graded reward
and filtered teacher NLL; the control disables both. This combined comparison
cannot attribute gains separately to reward shaping and teacher selection.

Each training job requests four Jean Zay A100 GPUs with a two-hour development
limit. Exact policy agreement is checked at steps 1, 2, 5 and every ten updates.
The controller rejects failed runs or inconsistent endpoints, compares every
actual trainer configuration field except bound output/run identifiers, and
records allocated GPU-hours. Equal 3,600 rollout samples are a sample-matched
budget; auxiliary computation adds cost and is reported separately.

Evaluation uses the same 100 disjoint development instances/task, eight raw
samples, seed 271828, temperature 0.7, top-p 0.9, top-k 50, FP16 and 3,000
completion tokens. Report candidate feasibility, feasible@8, each arm's
successful-instance gap, paired jointly feasible gap, and all-instance solved
quality. CVRP references are verified heuristic solutions, while MIS references
are certified optima. This reused development split and one training seed do
not supply untouched confirmation or establish an advantage over Jiang's
separately specified FOARL baseline.
