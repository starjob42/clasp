The next CVRP/MIS development experiment compares classic GRPO, a matched
FOARL-f638 port, and GRPO with feasibility-triggered incumbent supervision.
The preceding completed pilot did not establish superiority in both feasibility
and gap. Its results and source snapshots remain immutable.

The new teacher pool contains the canonicalized raw feasible candidates and
the existing independently verified repair/local-search candidates. For each
prompt, select one best feasible target if any raw candidate is infeasible, or
if the target strictly improves the best raw candidate. No selected target may
be worse than a raw feasible incumbent. All-feasible groups without improvement
receive no teacher. In a group without any feasible raw candidate or usable
teacher, one globally designated row constructs a bounded reference-free
fallback from the instance: greedy completion/exchanges for MIS, or capacity-safe
insertion and the existing local search for CVRP. The shared verifier checks the
fallback again; targets exceeding 3000 tokens including EOS are rejected.

This addresses a concrete lost learning signal: a mixed MIS group containing an
optimal feasible raw answer could not receive any strict-improvement teacher,
even when most sibling samples violated constraints. The quality floor applies
to supervision targets; population feasibility and gap must still be measured.

Teacher NLL averages tokens within a target and then selected targets across the
complete optimizer update. With A accumulation microbatches, W ranks, and M
selected targets, each rank contributes `0.05 * A * W * sum(target_NLL) / M`
before the existing outer division by A and DDP averaging by W. Empty ranks
participate in dummy forwards when another rank has a teacher. Empty windows
have zero teacher loss. This mode requires one generation buffer per complete
optimizer update and one policy iteration; partial or misaligned windows fail
explicitly. NLL uses the existing temperature-0.7 policy log-probability helper.

The experiment retains the previous graded constraint reward, teacher coefficient
0.05, and bounded local search. Raw on-policy completion tokens, advantages and
old log probabilities remain unchanged by teacher construction. Inference uses
the same raw best-of-8 decoder for every arm, without repair or solver assistance.
Fast-tokenizer padding/truncation state is restored after teacher encoding so
auxiliary training cannot change serialized tokenizer settings.

All three arms start fresh from the same certified task-specific SFT LoRA
checkpoint-31250. Each uses seed43, 100 updates, four A100s, B9, global36
completions per update, per-device batch3, accumulation3, learning rate1e-6,
AdamW8bit, weight decay0.01, warmup0.03, and a linear schedule. Prompts are
`jiang_f638`, no-reasoning, MIS response-output0 in every arm; decoder caps are
29768 prompt and3000 completion tokens. Five-update runs are correctness gates,
not performance results. Performance allocations use the two-hour A100 dev QoS. Corrected five-update
diagnostics request 25 minutes within the same dev QoS.

FOARL preserves the two released reward heads, group sample-standard-deviation
normalization, sequence-mean GRPO loss, beta0.05, and a frozen copy of the same
SFT policy as KL reference. Common prompts, B9, horizon and optimizer settings
make this an explicitly matched development port, not an official reproduction.
Historical native-profile source pins and production launchers are unchanged.

The common budget is 3600 sampled completions and400 prompt groups per run.
Actual generated tokens, teacher search, supervised forwards and FOARL reference
forwards consume different work; measured allocated GPU-hours must accompany
the performance table. A sample-matched result is not an equal-compute claim.
The existing100-instance data_v3 split is already inspected development data.
Report feasibility@8, candidate feasibility, own-success gap, paired gap on the
joint feasible subset, and paired uncertainty. Published Jiang results use a
different benchmark and must appear in a separately labeled table. A later
independent seed and untouched evaluation are necessary for confirmation.

The normalization-only ablation retains `best_per_prompt` selection and enables
`optimizer_window` normalization. The main modified arm instead uses
`feasibility_incumbent`. Historical defaults remain `per_completion` and
`microbatch`, so prior launchers retain their declared teacher-selection behavior.

The first native checks completed all five distributed gradient/policy checks
for both modified-task arms. The CVRP FOARL check stopped at its first
post-step exact-KL audit: the closure inventory uses safetensors dtype labels
(e.g. F32), whereas the full-policy inventory uses torch labels
(e.g. torch.float32). Comparing these strings rejected identical tensors.
The corrected audit maps the labels explicitly while preserving exact dtype,
shape, numel, value hashes, and the original inventory encodings. Twelve
actual-PEFT regressions cover both live and serialized receipts at three dtypes
and reject real dtype/hash changes. The original jobs remain excluded from
performance. Four fresh five-update diagnostics cover modified GRPO and FOARL
on both tasks, using one new immutable source before the six-arm experiment.
