The original B=9 development study did **not** demonstrate that CLASP beats
classic GRPO on feasibility and gap. On CVRP, both CLASP variants solve fewer
instances; their small improvements in paired gap are uncertain. On MIS, all
three RL policies have exactly the same best-of-eight feasibility and selected
cardinalities. These are the completed original results; the separate
[B=18 follow-up](clasp_grpo_b18_followup_20260905.md) has results pending and contributes
no numbers to this report.

| Task | Method | Feasible candidates ↑ | Feasible@8 ↑ | Mean feasible gap ↓ | Solved quality ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CVRP | SFT initialization† | 51.625% | 95/100 | 3.363468% | 0.862753 |
| CVRP | Classic GRPO | 64.625% | 98/100 | 3.114904% | 0.895090 |
| CVRP | CLASP v3 | 65.875% | 96/100 | 2.992214% | 0.879940 |
| CVRP | CLASP v4 | 64.875% | 96/100 | 2.969460% | 0.880657 |
| MIS | SFT initialization† | 78.250% | 94/100 | 2.466205% | 0.880601 |
| MIS | Classic GRPO | 78.875% | 95/100 | 2.300122% | 0.893732 |
| MIS | CLASP v3 | 79.375% | 95/100 | 2.300122% | 0.893732 |
| MIS | CLASP v4 | 79.250% | 95/100 | 2.300122% | 0.893732 |

Each row uses 100 instances and 800 raw candidates. Feasible@8 means at least
one strictly feasible candidate. The headline gap averages only that method's
successful instances: `100 × max(0, (distance − reference)/reference)` for CVRP,
and `100 × (optimum − cardinality)/optimum` for MIS. CVRP references are verified
heuristics, while all MIS references are certified optima. Failed instances
have no conditional gap; solved quality includes them as zero and averages
`feasible × exp(−3 × gap)` with gap expressed as a fraction. The
[canonical CSV](../analysis_outputs/clasp_grpo_fair_dev_20260905/original_complete_archive_v1/campaign/comparison_complete_v1/table.csv)
also retains supplementary signed gaps and better-than-reference counts.

† SFT is descriptive initialization context. Its stage and tokenizer
serialization differ from the RL checkpoints, and all six strict SFT-versus-RL
pairing attempts are rejected. The comparison code was not relaxed. All three
RL arms have matching strict protocol fingerprints within each task.

The matched recipe was fixed before these evaluation outputs:

- **Initialization and training:** the same task-specific, certified Qwen2.5-7B
  SFT checkpoint-31,250 in every arm, trained from the clean 500k-instance SFT
  corpus. Each RL arm starts fresh with seed/data seed 43 and ends at 100
  optimizer updates. B=9 and global generation batch 36 give four prompt
  groups and 36 completions per update, or 3,600 training completions. Training
  uses four Jean Zay A100s; evaluations use one A100. Every job requests
  `qos_gpu_a100-dev` with a two-hour limit.
- **Shared optimization:** BF16 training, LoRA rank/alpha 64/64, dropout zero,
  `bias=all`, AdamW with 8-bit state, learning rate 1e-6, linear decay, 3% warmup,
  weight decay 0.01, betas (0.9, 0.999), epsilon 1e-8, gradient norm limit 1,
  token-mean GRPO loss, length alpha zero, one update per generated batch, and
  no KL penalty. Reward standard deviation uses sample correction 1. Classic
  GRPO (`grpo_z`) and the CLASP anchor use the same strict feasible-objective
  reward; this is not a claim of bit-exact equivalence to another native
  entrypoint with different reward heads. Recorded trainer settings, dataset
  and context match within each task apart from run names and output/log paths.
- **Declared algorithm differences:** v3 adds capped, normalized set-quality
  credit and raw failure/recovery credit to the GRPO anchor. V4 adds the
  centered candidate-feasibility term `0.5 × (1 + lambda) × (f − mean(f))`.
  These anchored, normalized methods are surrogates; they do not inherit the
  raw canonical estimator's unbiasedness claim.
- **Evaluation:** the frozen source-v30 evaluator, Jiang-F638/no-reason task
  representation, FP16, K=8 in one generation call, seed 271828, temperature
  0.7, top-p 0.9, top-k 50, repetition penalty 1, prompt limit 29,768 tokens,
  and completion limit 3,000 tokens. Outputs are unmodified and unrepaired.
  The frozen addendum replaces the old public slice with fresh `eval[0:100]`
  for each task; the old public data is not part of this table.

The [training protocol](../analysis_outputs/clasp_grpo_fair_dev_20260905/current/protocol.json),
[fresh evaluation addendum](../analysis_outputs/clasp_grpo_fair_dev_20260905/current/protocol_addendum.json)
and [complete configuration audit](../analysis_outputs/clasp_grpo_fair_dev_20260905/current/telemetry_audit_all_terminal100_v1/matched_configuration.json)
retain the exact contracts, source and checkpoint hashes. All six training
runs reached checkpoint 100 and passed terminal checkpoint/fidelity checks.

The [fresh benchmark](../data/development_fair_cvrp_mis_20260905_v3/README.md)
uses fixed independent generation seeds 2026090501 (CVRP) and 2026090502 (MIS)
and the repository's distributions. A completed scan of 1,005,800 rows from the
actual SFT inputs, RL training data and public evaluations found zero semantic
overlap. References were frozen before policy outputs: CVRP takes the best
strictly verified raw cost from the initial ten-second ORtools reference and
two fixed ten-second LKH searches; MIS has 100 CP-SAT OPTIMAL receipts with
matching objective/bound. LKH materially strengthened 54 CVRP references.
These are development instances, not a repeatedly reusable confirmation set.

The lower CVRP headline gaps partly reflect different sets of solved instances.
A fair paired gap compares only instances both policies solve:

| CVRP comparison | Joint successes | GRPO gap | CLASP gap | CLASP − GRPO gap, 95% paired CI |
| --- | ---: | ---: | ---: | ---: |
| GRPO vs v3 | 95 | 3.014503% | 2.980134% | −0.034369 pp [−0.252682, +0.169272] |
| GRPO vs v4 | 96 | 3.026469% | 2.969460% | −0.057009 pp [−0.226668, +0.100914] |

Both CLASP CVRP arms lose two feasible@8 successes relative to GRPO. Their
paired feasibility intervals are [−6, +2] pp for v3 and [−5, 0] pp for v4;
exact McNemar p-values are 0.625 and 0.5. V4's solved-quality change is
−0.014433, interval [−0.040326, +0.003570]. No observed tradeoff establishes a
win on both requested targets.

On MIS, all three RL policies solve the same 95 instances with identical best
cardinalities. Every RL paired feasibility, gap and solved-quality difference
is zero, with empirical bootstrap intervals [0, 0]. This is an observed tie,
not proof of population equivalence. V3 improves candidate feasibility by
0.500 pp, percentile interval [+0.125, +1.000], but only four instances improve
and none regress; the exact two-sided sign-test p-value is 0.125. V4 improves
the candidate-rate point estimate by 0.375 pp, interval [−0.500, +1.625], with
two improving and two regressing instances. These small changes do not improve
best-of-eight outcomes.

The strongest measured diagnostic is sparse MIS credit. The
[complete 100-update telemetry](../analysis_outputs/clasp_grpo_fair_dev_20260905/current/telemetry_audit_all_terminal100_v1/README.md)
shows:

| Fraction of training groups | CVRP v3 | CVRP v4 | MIS v3 | MIS v4 |
| --- | ---: | ---: | ---: | ---: |
| Nonzero set-quality coefficient | 92.50% | 91.75% | 18.25% | 22.00% |
| Nonzero binary-failure coefficient | 5.50% | 8.00% | 2.50% | 3.50% |
| Zero final combined coefficient | 0.50% | 0.50% | 43.75% | 42.50% |
| Active added dense-feasibility term | — | 82.50% | — | 24.00% |

At B=9/K=8, binary-failure credit can distinguish candidates only when exactly
one of the nine is feasible. A tied best objective can eliminate set-quality
credit. Increasing the failure multiplier cannot create centered failure
credit in all-failing groups. These algebraic limitations and their observed
frequency explain why added CLASP credit is often absent; they do not prove
the cause of a held-out score difference. Zero final coefficient is a
per-group telemetry measurement under the trainer's finite signal tolerance,
not proof of zero parameter gradients in an entire batch. V4 demonstrably activates more candidate-feasibility credit,
yet this pilot does not show a best-of-eight benefit.

The [descriptive MIS string comparison](../analysis_outputs/clasp_grpo_fair_dev_20260905/audit_mis_complete_four_arm_v1/candidate_text_agreement/candidate_text_agreement.json)
finds 767/800 identical aligned completion strings against GRPO for both v3
and v4 (95.875%). Their 33 changed slots span 26 and 24 instances respectively.
The trained adapter hashes differ. This establishes substantial decoding
agreement under one seed; it does not establish identical token probabilities,
equivalent policies or a causal mechanism.

All uncertainty here resamples paired instances, with 20,000 frozen bootstrap
draws. It does not estimate training-seed uncertainty or adjust for multiple
comparisons. Candidates are clustered within their 100 instances. With GRPO
already solving 98 CVRP and 95 MIS instances, this sample cannot establish a
positive feasibility effect at the 5% level using exact two-sided McNemar,
even if another policy solves all 100. A claimed general advantage needs more
than these single-seed development results.

The [independent cross-check](../analysis_outputs/clasp_grpo_fair_dev_20260905/original_complete_crosscheck_v1/crosscheck.json)
reverified all 101 canonical archive files and matched every field of all eight
summary rows and all six strict paired RL statistics/quality intervals exactly
against the separate [CVRP](../analysis_outputs/clasp_grpo_fair_dev_20260905/audit_cvrp_complete_triplet_v1/README.md)
and [MIS](../analysis_outputs/clasp_grpo_fair_dev_20260905/audit_mis_complete_four_arm_v1/README.md)
raw audits. The retained raw artifacts, identities and verification scripts
make the table reproducible. The pending B=18 study remains a separately
reported follow-up on already inspected development data.
