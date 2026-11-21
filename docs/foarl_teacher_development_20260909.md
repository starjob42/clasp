The previous raw and assisted comparisons did not establish superiority over
FOARL. Those reports and the consumed 300-instance test draw remain historical
results. This successor study is development work, with no performance claim.

The next candidate adds a verified teacher likelihood term to the matched FOARL
port. The control uses the same new trainer with teacher weight zero. Both use
the released f638 optimality and feasibility reward heads, group sample standard
deviation plus 1e-4, token clipping at0.2, completion-token mean then sequence
mean, and beta0.05 against an independently loaded, frozen original SFT model.
The candidate adds weight0.05 times the mean token NLL of one selected verified
target per eligible prompt, averaged over the complete optimizer window.
Targets must improve the group's best feasible sample or teach its best feasible
incumbent when some samples fail; all-failed groups can receive one bounded
constructive fallback. The teacher never accesses reference objective labels.
The candidate uses the existing two-pass bounded CVRP/MIS operators.

Both methods start afresh from the same task SFT adapter, seed43, and have a
fixed400-update learning-rate horizon from their first update. Stops at100,
200,300,400 split the run into two-hour A100 development jobs; the original
100-update checkpoints are not extended. Each job uses4A100, rollout B9,
global36 samples/update, per-device batch3, accumulation3, LR1e-6, AdamW8bit,
weight decay0.01, warmup0.03, linear decay, and identical original common
no-reasoning prompts, temperature0.7, top-p0.9, top-k50, prompt cap29768 and
completion cap3000. At most1600 prompt groups are drawn, remaining within the
first epoch of each retained training dataset. The development evaluator will
use the existing decontaminated100-instance task splits with raw best-of8.

The first native stage consists only of correctness diagnostics: MIS control
uninterrupted10, MIS teacher uninterrupted10, MIS teacher5 then resume to10,
and CVRP teacher5. All retain the400-step scheduler horizon and check gradient
and policy consensus at every update. They include actual task-tokenizer tests
with synthetic instances. Training continuation must preserve optimizer,
scheduler, all four rank RNG states, original SFT KL identity, method, source,
data and run contract. The native uninterrupted/resumed diagnostic must pass
before admitting the100-update development stage.

Development comparisons will inspect the same predeclared100/200/400 milestones
for both arms. Step400 is the primary development endpoint; steps100/200 describe
the learning curve. Extra teacher tokens/search and allocated A100 time must be
reported separately: equal rollout budgets do not imply equal training compute.
If a candidate is selected, freeze its recipe and checkpoint selection rule
before drawing a new final test set. A new classic-GRPO control and independent
training seeds are required before claiming broader superiority. Published
Jiang results remain a separate cross-study table unless their exact test
instances and execution protocol can be matched.
