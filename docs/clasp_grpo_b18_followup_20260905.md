The completed CVRP B18 follow-up improves point estimates against matched B18
GRPO, but does **not** beat the retained original B9 GRPO on both feasibility
and gap. All four CVRP follow-up jobs completed successfully within two-hour
A100 development limits. The table and all three strict paired comparisons
passed an independent raw-candidate audit and an exact cross-check.

| Method | B | Feasible@8 | Candidate feasibility | Own-feasible gap |
| --- | ---: | ---: | ---: | ---: |
| Original classic GRPO | 9 | 98/100 | 64.625% | 3.114904% |
| Matched classic GRPO | 18 | 96/100 | 65.625% | 3.233939% |
| CLASP v3 | 18 | 97/100 | 66.125% | 3.104702% |

Against matched B18 GRPO, CLASP's feasibility change is +1 percentage point,
95% paired interval [-2, +5]; its joint-success gap change is -0.131605
percentage points, interval [-0.348535, +0.079131], over 95 joint successes.
Against original B9 GRPO, CLASP loses one feasible instance and its gap on
97 joint successes is **0.015848 percentage points worse**. Its slightly
lower headline gap conditions on a different success set. These results do
not establish superiority. CVRP references remain verified heuristics.

The [complete comparison](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_cvrp_comparison_archive_v1/campaign/comparison_b18_v1/comparison.json),
[independent audit](../analysis_outputs/clasp_grpo_fair_dev_20260905/audit_b18_cvrp_complete_three_arm_v1/audit_report.json),
and [exact cross-check](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_cvrp_complete_crosscheck_v1/crosscheck.json)
retain raw identities, paired intervals and signed reference gaps. All 100
training updates are retained in the [terminal telemetry](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_telemetry_all_terminal100_v1/audit.json).
Binary failure credit is active in 45.5% of B18 v3 groups versus 5.5% at B9;
mean absolute failure credit is 0.004935 versus 0.006970. Broader support does
not imply larger average credit or a better policy. The full recorded trainer
configurations match within the B18 pair apart from run/log/output names.

The original rationale and declaration are retained below.

The first frozen pilot did not meet the requested CVRP target: GRPO solved
98/100 instances and CLASP v3/v4 each solved 96/100. The B=18 follow-up tests a
different, direct remedy for sparse subset credit. It is a separately declared
development experiment; it does not replace or alter that result.

For B rollouts and deployment K=8, a mixed group's binary failure credit can
be nonzero only when its feasible count m satisfies 1 <= m <= B-K. At B=9,
that allows only m=1; at B=18 it allows m=1 through 10. More subsets can also
distinguish tied or non-best feasible candidates. All-failing groups still
have zero centered binary-failure credit. Larger B is not a guarantee of
better optimization or generalization.

The implementation enumerates all 43,758 eight-subsets of 18 candidates.
PyTorch 2.9.1's `torch.combinations` constructs a B^K intermediate mask,
making this setting impractical inside the original implementation. The safe
replacement caches immutable `itertools.combinations` tuples and creates a
fresh tensor on the requested device. It preserves order, dtype and exact
subset contents, and caller mutation cannot poison later results. Full
two-prompt canonical computation took a median 0.551 seconds in a one-thread
Jean Zay CPU check; GPU timing and full-run completion are measured separately.

The declared matched arms are classic GRPO and CLASP v3 with B=18. Both start
fresh from the same CVRP SFT checkpoint-31250, use seed 43, 100 updates, four
A100s, two-hour development QoS, 36 completions per update, sample reward
standard deviation, token-mean loss, learning rate 1e-6 and the original
linear schedule. Both see two prompt groups per update, compared with four
in the original B=9 pilot. The total is still 3,600 training completions; the
reduced number of distinct training prompts is an explicit tradeoff.

CLASP keeps v3's GRPO anchor, capped quality normalization and controller;
it adds no v4 candidate-feasibility bonus. Exact subset enumeration does not
make the normalized anchored surrogate an unbiased set-objective estimator.
The only recipe change from v3 is B and its required complete subset count.
The B=18 override is accepted only for these two named methods and must be
validated explicitly. The original six-arm scheduler clears inherited
overrides and still requires B=9.

Evaluation keeps K=8, seed 271828, the same 100 CVRP development instances,
verified references, raw outputs, prompting and decoding. Those instances
have now been inspected; this follow-up is not untouched confirmation.
Report both original B=9 GRPO and the new matched B=18 GRPO, candidate
feasibility, feasible@8, jointly feasible gaps and solved quality. Retaining
the stronger original baseline prevents a weaker group-size comparator from
being presented as an overall win. No superiority claim is justified before
these results, and a later positive development result still needs independent
training seeds and confirmation data.

Both jobs ran on Jean Zay: classic GRPO 1793279 and CLASP v3
1793280, starting at 19:23:30 and 19:42:32 UTC respectively. Startup contracts
verify B=18 and the declared matched settings; CLASP uses all 43,758 subsets. The CLASP
run completed without an out-of-memory error. Training took 1h06m07s and
1h06m16s; their evaluations, jobs 1796415 and 1797311, took 29m43s and 29m17s.
The separately frozen evaluation controller evaluated only complete,
validated checkpoint-100 adapters.

The [training declaration](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_declaration_v1/ledgers/protocol.json)
and [evaluation declaration](../analysis_outputs/clasp_grpo_fair_dev_20260905/b18_evaluation_remote_declaration_v1/evaluation_protocol.json)
retain the exact commands. The valid main training declaration is unaffected
by a corrected auxiliary preflight receipt; the old file and explicit
restoration addendum remain preserved in the training declaration archive.
The [complete original B9 results](clasp_grpo_fresh_development_results_20260905.md)
remain separately reported.
