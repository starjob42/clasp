This development pilot implements plain GRPO with graded constraint violations
and verified online improvement supervision. It is a new algorithm experiment;
the completed CLASP results and frozen source snapshots remain historical records.

The two changes are independently opt-in. `GRPO_GRADED_REWARD=1` replaces the
CVRP/MIS objective head. An empty or unparseable answer receives −1. Other
invalid answers receive `−V/(1+V)`, where V is a sum of verifier-derived defects
normalized by fixed instance quantities. Invalid route distance and independent
set cardinality receive no objective credit. Feasible CVRP receives
`1 + reference_distance/(reference_distance + verified_distance)`, while feasible
MIS receives `1 + cardinality/num_nodes`. Better-than-reference CVRP solutions
remain distinguishable. These disjoint reward intervals order individual
candidates; they do not guarantee lexicographic population optimization.

CVRP V is `(missing customers + duplicate customers + invalid nodes + bad route
boundaries)/(n−1) + total capacity overflow/max(total customer demand, capacity)`.
MIS V is `(invalid nodes + duplicates + selected conflicting edges)/n`.
The classic control retains its existing strict-feasible/objective scalar reward.
Both methods use ordinary group-z advantages with sample standard deviation,
the same clipped token-level GRPO objective, and zero dual or KL penalties.

`GRPO_VERIFIED_IMPROVEMENT=1` adds `0.05 × NLL` on independently verified teacher
completions. The NLL averages tokens within a target and eligible targets across
GPUs within each optimization microbatch; it then follows the existing gradient
accumulation convention. Targets have uniform weight. Teachers attach to
individual sampled completions before TRL shuffling, preserving prompt identity.
The auxiliary uses the existing TRL log-probability helper and thus the same
temperature 0.7 as the rollout policy; it is not an untempered SFT softmax.
Ranks with no local target participate in a dummy forward when another rank has
targets, contributing zero loss. The global eligible-target count controls the
denominator, rather than an unweighted mean of rank-local means.

CVRP teachers keep sampled customer order where possible, remove invalid or
duplicate IDs, split over-capacity routes, insert missing customers at the
cheapest feasible position (including a new route under the existing unlimited
fleet contract), then apply two best-improvement passes of 2-opt, relocation,
or exchange. MIS teachers resolve conflicts, greedily fill, then attempt up to
two remove-one/add-two exchanges, with at most 4096 pair checks. Both recheck the
final candidate with the shared task verifier. Feasible sources require a strict
objective improvement; invalid sources require a feasible repair. A repair's
gain is recorded as zero instead of comparing invalid and feasible objectives.
Teacher search uses sampled solutions and instance data, not reference solutions.

Malformed or empty parsed outputs do not get teacher targets in this version;
the graded reward still handles them. Teachers are bounded to 256 CVRP nodes and
512 MIS nodes, with additional input/edge caps. Overlong targets are rejected,
not truncated. EOS is included within the 3000-token target cap. Raw rollout
tokens, rollout rewards, and old-policy probabilities remain the on-policy GRPO
data; external teachers enter only the auxiliary supervised loss. Inference
remains raw, without repair or constrained decoding.

The launch plan is four separate Jean Zay A100 dev jobs: classic `grpo_z` and
combined `grpo_verified` on each of CVRP and MIS. Each uses four A100s and a
two-hour limit, fresh task-specific SFT-LoRA checkpoint-31250, seed/data seed43,
B9, global36 completions per update, and 100 uninterrupted updates. This is
3600 on-policy completions and 400 prompt groups per run. All optimizer, LoRA,
generation, data, and model settings are matched. Training-time teacher search
and extra forward/backward computation are part of the modified method and
must be reported separately: equal rollouts are not equal consumed GPU-hours.

The already-inspected 100-instance-per-task development split may be reused for
the initial raw K8 evaluation. It cannot establish untouched confirmation, and
one training seed cannot quantify training-seed uncertainty. Feasibility/gap
claims require completed evaluations; successful launch or nonzero teacher loss
is not evidence of a performance win. Additional reward-only/teacher-only
ablations and independent seeds remain follow-up experiments.

Implementation: [graded rewards](../src/clasp/rewards/co_tasks.py),
[CVRP teacher](../src/clasp/rewards/cvrp_verified_improvement.py),
[MIS teacher](../src/clasp/rewards/verified_improvement.py),
[trainer](../src/clasp/trainers/primal_dual_grpo.py), and the
[isolated launcher](../scripts/train_grpo_verified_checkpoint_dev_a100.slurm).
