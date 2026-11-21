# CLASP v4: a controlled candidate-feasibility correction

This is an experimental development arm, not an established improvement over
GRPO. It addresses the loss of candidate-feasibility pressure in the v3
best-of-eight residual. See the measured historical results in
[the diagnosis](clasp_grpo_diagnosis_20260905.md).

The completed fresh seed-43 CVRP pilot does not meet the desired improvement:
v4 solves 96/100 versus GRPO's 98/100. Its lower headline gap is partly due to
different successful instances; the jointly feasible gap difference has a
95% interval spanning zero. See the
[verified fresh CVRP comparison](../analysis_outputs/clasp_grpo_fair_dev_20260905/audit_cvrp_complete_triplet_v1/README.md).
MIS evaluation is also complete: v4 and GRPO solve the same 95/100 instances with identical selected objectives and 2.300122% gap. See the [complete original study](clasp_grpo_fresh_development_results_20260905.md). The coefficient remains experimental; this pilot provides no best-of-eight improvement.

For B=9/K=8, every leave-one-out set succeeds whenever at least two candidates
are feasible. The binary set-failure coefficient is then identically zero.
With a unique best candidate and best/runner-up quality difference delta, the
quality coefficient is `[8*delta, -delta, ..., -delta]`, up to permutation.
Every other candidate receives the same negative tail correction regardless
of whether it is feasible. If the best quality ties, this quality coefficient
also vanishes. These are properties of the exact set objective, not numerical
bugs.

V4 keeps every v3 coefficient and adds one term:

```text
A_v4[i] = A_v3[i] + 0.5 * (1 + frozen_lambda) * (feasible[i] - mean(feasible))
```

The extra term is nonzero only for mixed groups. It increases each feasible
candidate's coefficient and decreases each infeasible candidate's coefficient
by centered amounts. It leaves all pairwise quality differences among feasible
candidates unchanged. Uniformly feasible and uniformly infeasible groups are
unchanged, so v3's existing objective credit and failed-set recovery remain
responsible there. The scalar is frozen across the rollout group; the
controller still commits at optimizer boundaries.

This is an explicit candidate-feasibility surrogate added to a set objective.
It is not an unbiased estimator of the original best-of-eight-only objective,
and the coefficient property does not guarantee better learned-model behavior.
The scale 0.5 is fixed before these new results, not selected on them. No new
solver demonstrations, repaired outputs, reference imitation, or extra rollout
budget are added to training.

The fresh pilot uses classic GRPO, v3, and v4 for CVRP and MIS, starting each
arm directly from the same certified task SFT checkpoint-31250. It fixes train
seed 43, 100 steps, B=9, global generation batch 36, four A100 GPUs, 1e-6 learning rate,
the same linear schedule and warmup, token importance ratios, GRPO's response
token-mean reduction, zero KL weight, and identical Jiang-F638 representation.
The retained comparator uses the same strict feasible-objective reward as the
CLASP anchor. It is the classic GRPO objective under this shared reward, not
the repository's separate default three-reward-head configuration.

Both the comparator and the anchor use `PD_GROUP_STD_CORRECTION=1`, matching
the sample standard deviation in installed TRL 0.24. Historical experiments
used population standard deviation; their default remains available as 0.
The new choice is shared by every fresh arm and recorded in each contract.

All jobs use `qos_gpu_a100-dev` and a two-hour maximum. The launcher respects
available submission slots and gives training priority over dependent
evaluation jobs. Run the printed resume command as jobs finish. An existing
unrelated array can reduce available slots; it is never canceled by these
scripts. Reconciliation records immutable submission events and refuses to
reuse partial output directories.

Evaluate saved terminal checkpoints with eight raw samples in one call,
seed 271828, T=0.7, top-p=0.9, top-k=50, FP16, 3000 completion tokens, and 29768 prompt
tokens. Report candidate feasibility, feasibility@8, paired gap on jointly
feasible instances, and unconditional solved quality. The old public slice is
development context. The separate 100-instance-per-task fresh set is generated
before inference and requires zero semantic overlap with the exact SFT and RL
training sources. MIS reference proofs and CVRP heuristic references are
reported separately; CVRP reference gap must not be called a certified
optimality gap. Neither a single training seed nor an instance bootstrap
establishes training-seed robustness.

The audit also corrected two independent legacy bugs: tied rank/elite rewards
no longer favor arbitrary sample positions, and the Dr.GRPO alternative now
uses its configured completion horizon rather than sampled padding width.
Those branches are inactive in this pilot, so they cannot explain its results.

Primary algorithm references: [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300),
[Dr.GRPO analysis](https://arxiv.org/abs/2503.20783), and
[TRL 0.24 GRPO documentation](https://huggingface.co/docs/trl/v0.24.0/en/grpo_trainer).
