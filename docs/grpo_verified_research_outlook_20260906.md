# GRPO for combinatorial optimization: diagnosis and next experiment

The completed comparisons do not establish that CLASP beats classic GRPO on both feasibility and gap. The strongest next direction is to learn verified solution improvements while making invalid decisions harder to generate. Changing reward weights alone does not address either the missing learning signal or the underlying search problem.

## What the evidence establishes

**CLASP's best-set component discards useful candidate comparisons.** At training group size B=9 and inference budget K=8, each subset omits one candidate. A unique best candidate receives quality credit proportional to eight times the best-versus-runner-up margin; every other candidate receives the same negative margin. Two tied best solutions erase this component entirely. In the recorded MIS v3 run, set-quality credit is active in only 18.25% of groups. Binary failure credit is nonzero only when the feasible count satisfies `1 <= F <= B-K`; at B9/K8 this means exactly one feasible candidate. Its CVRP support is only 5.50% of groups. Increasing the dual multiplier cannot create relative binary-failure credit in an all-failing group. The GRPO anchor and violation-recovery components can still learn; these sparsity statements concern the specified components, not the entire optimizer. [Derivation and complete telemetry](clasp_grpo_complete_development_results_20260905.md), [canonical estimator](../src/clasp/set_grpo/canonical.py).

Increasing B to 18 broadens MIS quality support to 46.50% and CVRP failure support to 45.50%, but halves prompt coverage at the fixed completion budget. It did not produce an overall win. B9 CLASP v3/v4 each solve 96/100 CVRP instances versus GRPO's 98/100; their lower headline gaps partly reflect different success sets. MIS selected feasibility and objectives tie exactly within the corresponding comparisons. [Independently rechecked results and paired statistics](../analysis_outputs/clasp_grpo_diagnosis_20260906/evidence_audit.md).

**The first verified-teacher runs had a separate correctness defect.** The actual TRL generation context re-enabled gradient checkpointing without restoring its declared `use_reentrant=False` setting. In the two-forward policy-plus-teacher path, this produced divergent gradients across DDP ranks. A reproduction using the actual generation lifecycle failed before clipping; preserving the declared checkpointing arguments removed that failure. Native four-A100 MIS job 1825156 subsequently completed five updates with exact gradient and policy agreement at every update. The scoped restoration and early exact guards are implemented. This validates the training path, not task performance, and does not retroactively explain every historical CLASP result. [Causal reproduction receipt](../analysis_outputs/grpo_verified_ddp_diagnosis_20260906_v1/checkpointing_causal_receipt.json), [implementation validation](../analysis_outputs/grpo_verified_ddp_diagnosis_20260906_v1/checkpointing_and_selection_validation.json).

**Uniform teacher supervision promotes many inferior targets.** On a fixed inspected slice of 20 instances/task, 103/134 CVRP teachers are worse than the same instance's best raw feasible candidate, despite each target improving or repairing its own source. Yet the best teacher improves the raw incumbent on 14 instances. MIS teachers improve four raw incumbents and repair two entirely infeasible groups. These are offline operator outcomes, not policy gains. The implemented `best_per_prompt` selector retains one target only when it strictly beats every raw feasible candidate, or the best verified repair when all candidates fail. In this slice it retains 14 CVRP and six MIS targets, excluding all dominated targets. [Audit, witnesses, and compute accounting](../analysis_outputs/clasp_grpo_diagnosis_20260906/teacher_audit/report.md), [selection replay](../analysis_outputs/clasp_grpo_diagnosis_20260906/teacher_audit/selection_replay.json).

## Immediate experiment

The fresh development comparison pairs classic GRPO with graded violations plus selected verified teachers, separately on CVRP and MIS. Each arm starts from the same task SFT-LoRA checkpoint, uses the same seed/data order, B9, 36 completions/update, and 100 updates. The modified arm adds training-only search and auxiliary supervision; raw rollout tokens, old-policy probabilities, and evaluation decoding remain intact. Exact distributed checks and final checkpoint certification are mandatory. Each job uses four Jean Zay A100s with a two-hour dev limit. Completed evaluation is required before reporting a performance difference.

This pilot has now completed. CVRP feasible@8 stays at 96/100, with conditional
gap 3.0689% → 3.0479%; MIS stays at 95/100, with gap 2.3678% → 2.2147%. Candidate
feasibility changes by +0.5 and −0.5 percentage points respectively. Paired gap
intervals include zero, and MIS improves only two selected objectives. The
recipe therefore does not establish a win on both feasibility and gap. Training
and evaluation cost 19.3656 A100-hours in total; modified training uses 10.40%
more allocated time on CVRP and 5.55% more on MIS. The stronger algorithm below
remains a research proposal. [Completed results and methods](clasp_grpo_findings_20260906.md).

The next attribution ablation should separate graded reward, selected-teacher supervision, and their combination. A feasible reward interval orders individual candidates but cannot guarantee population feasibility: expected scalar reward can still trade feasibility probability for conditional quality. Report target eligibility, dominance, unique targets, group-advantage support, clipping, completion length, and auxiliary gradient magnitude alongside task outcomes. Count teacher search and supervised tokens in compute accounting.

## A stronger algorithmic direction

Learned improvement and local search are established approaches; using them is not a novelty claim by itself. NeuOpt, for example, learns flexible k-opt moves and reports benefits from guided exploration of both feasible and infeasible routing states. Its results motivate testing the treatment of intermediate constraints, rather than assuming that feasibility masking is universally preferable. [Ma, Cao and Chee, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/9bae70d354793a95fa18751888cea07d-Abstract-Conference.html). The proposed contribution would need evidence for the specific GRPO credit, teacher-selection and training-efficiency choices.

1. **Train an improvement policy.** Condition on the instance and a verified feasible incumbent, and learn a bounded sequence of improving edits. For CVRP, start with relocation, exchange, and 2-opt proposals; for MIS, removals followed by compatible additions. Reward the independently verified change in objective and retain the best feasible incumbent. Teacher trajectories come exclusively from training instances. Short edits expose which decisions changed the objective and allow recovery from earlier choices; full-solution terminal rewards spread the same credit across many unrelated tokens. This is a research hypothesis requiring comparison against both constructive GRPO and the same classical operators without an LLM.

2. **Use valid-action masks for construction and returned-output validity.** MIS construction can exclude selected vertices and their neighbors. Unlimited-fleet CVRP can restrict choices to unvisited customers fitting residual capacity, permit depot returns, and allow termination only after all customers are served. A grammar-aware token trie must handle multi-token node IDs. Apply the identical legal-action mask to sampling and policy/old-policy log probabilities. Correct masks can preserve feasibility for valid instances and completed outputs; they do not guarantee optimality or successful completion within a token cap. For iterative search, compare complete edits applied and verified atomically with a separately controlled regime that permits temporary violations. In both regimes, retain the best verified feasible incumbent and return a feasible solution. Predeclare the exploration/repair budget and account for its cost; NeuOpt's evidence makes this a useful ablation rather than a settled choice.

3. **Distill verified search into the policy.** Selected improved targets supply solutions the raw policy did not find, addressing all-invalid or tied-reward groups. Preserve distinct high-quality solutions across training examples and monitor diversity. Increasing the teacher weight or local-search budget without measuring target quality and compute is not an improvement strategy by itself.

An incumbent-conditioned or constrained-decoding system is a different evaluation track from raw end-to-end text generation. Give its GRPO/FOARL controls the same incumbent, action masks, search operators, and inference budget. Also retain the raw-generation comparison so gains from training and gains from the decoder remain distinguishable.

An implementable first version would keep the existing task SFT weights, then
perform a small, declared edit-SFT stage using verified local-search trajectories
from training instances. Give every learned control in this track the same
edit-SFT initialization. At each RL state, sample a group of bounded edit
sequences from the same incumbent, apply and verify each proposal, and compute
its improvement relative to that incumbent: `(C_before - C_after) / C_before`
for positive-cost CVRP, or `(|I_after| - |I_before|) / n` for MIS. Penalize
invalid completed proposals separately and preserve the best feasible incumbent.
Apply group-relative advantages to proposal log probabilities; use selected
verified improvements for auxiliary supervision when the rollout group offers
little or no reward contrast. Predeclare the edit count, proposal-token limit,
acceptance rule, and teacher budget before comparing results.

A feasible initial incumbent plus a retained-incumbent fallback makes final
feasibility largely a property of the search system. All controls must receive
that same fallback; a move to 100% feasibility would then be a system result,
not evidence that the LLM learned to generate feasible solutions unaided.
Measure quality gained over the identical initial incumbent at equal proposal,
token, and wall-time budgets, including a classical-search-only control. This
tests whether GRPO improves the choice of edits enough to justify its cost.

## A fair Jiang/FOARL benchmark

Jiang et al.'s FOARL is GRPO with task-specific feasibility and objective rewards after Qwen2.5-7B LoRA SFT, followed by best-of-eight inference. [NeurIPS 2025 paper](https://papers.neurips.cc/paper_files/paper/2025/file/f0c6c93e1dbff84daf6082fef7e1a094-Paper-Conference.pdf). Pin the [official repository](https://github.com/Summer142857/LLMCoSolver) at the audited commit `f6388991d243c0aa91d9ee7e2fa18ed1c9e72477`; the repository already supplies its reward oracle and executed objective. A beta-zero, strict-gated classic-GRPO control is not an implementation of FOARL. [Paper/code reconciliation](jiang_llmcosolver_f638_training_audit_20260821.md).

| Control | Proposed confirmation contract |
|---|---|
| Methods | Classic GRPO, pinned matched FOARL, and the selected proposed method |
| Initialization | Identical certified task SFT-LoRA and frozen base/tokenizer; declare local SFT corrections |
| Training | Same instances, seed-specific order, rollout/token budget and checkpoint rule; matched hyperparameter-search allowance |
| Horizon | Use 100 updates for screening; the existing matched FOARL contract specifies 3,200 updates for the longer comparison |
| Replication | At least three independently declared training seeds; report all seeds and uncertainty |
| Test data | Untouched, instance-disjoint benchmark; predeclare size/density strata and perform power analysis before opening results |
| Inference | Same raw K1 and K8 decoder settings, token caps, precision and verifier; separate constrained/search-assisted track |
| Compute | Report rollout tokens, target tokens, teacher CPU time, allocated GPU-hours, failed-job cost and inference time; also compare quality at equal compute budgets |

Start confirmation with CVRP/MIS; extend to TSP, OP and JSSP before making a broad cross-task claim. Report candidate feasibility, feasible@8, conditional gap, gap on jointly solved instances, and unconditional success-versus-gap curves. Preserve signed better-than-reference CVRP values and label its heuristic-reference gaps accurately; MIS has certified optima. Cluster uncertainty by instance and include training-seed variation. Preregister the superiority test and multiplicity correction. A claim of beating GRPO on **both** feasibility and gap requires evidence for both outcomes, rather than a lower conditional gap with fewer solved instances. Current diagnostics and the selected-teacher implementation support a concrete experiment; they do not establish superiority over FOARL or the state of the art.
