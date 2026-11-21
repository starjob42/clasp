# LLMCoSolver f638 compatibility baseline

This package freezes the reproducible part of the official LLMCoSolver/FOARL
implementation at commit
`f6388991d243c0aa91d9ee7e2fa18ed1c9e72477`. It does **not** authorize a run or
support a state-of-the-art claim. The source contract remains `HOLD` until
model, data, budget, and reference-policy choices are resolved.

## What is faithfully pinned

The official RL code uses exactly two reward heads, in this order:
`optimality` and `feasibility`. TRL sums them with implicit unit weights. It
then normalizes the summed rewards within each eight-generation prompt group
using the unbiased sample standard deviation and `1e-4` epsilon.

The pinned TRL 0.15.2 / Unsloth 2025.3 objective is one-iteration GRPO with
`beta=0.05`. It averages token losses within each completion, then averages
over completions. In current TRL terminology this is `loss_type=grpo` with one
iteration; it is not `dr_grpo`.

The six upstream reward pairs (TSP, CVRP, OP, MVC, MIS, and JSSP) are retained
without silently repairing their quirks. A newline-normalized copy of the MIT
licensed upstream reward source is kept as a test oracle, and representative
outputs are compared exactly against the CLASP port. The publication
intersection is TSP, CVRP, OP, MIS, and JSSP; CLASP has no MVC benchmark lane.
PFSP has upstream environment/evaluation code but no RL reward mapping.

Publication metrics must always come from CLASP's strict evaluator. The
upstream training rewards are not valid replacements for it.

## The reference-policy mismatch

There are two distinct, currently disabled variants:

1. **Official-code operational semantics (context/reproduction only).** The
   official defaults use `bias=lora_only`. Unsloth obtains KL reference logits
   from the same live PEFT model inside `disable_adapter()`. On Qwen2 this turns
   off LoRA matrices but leaves the trainable q/k/v `base_layer.bias` values in
   the forward pass. The reference therefore evolves during optimization: it
   is neither a fixed foundation model nor a fixed SFT policy. PEFT 0.17.0 also
   saves none of those live biases under `lora_only`, so a resumed checkpoint
   cannot reproduce the uninterrupted policy/reference state.

2. **Scientifically corrected frozen step-zero SFT reference.** This uses the
   capacity-equivalent `bias=all` persistence correction and a separately
   loaded, frozen step-zero SFT adapter as the exact no-merge reference. CLASP's
   post-step verifier must prove its immutability and resume fidelity. This is a
   corrected, semantics-preserving reward/loss port, not a byte-identical FOARL
   reproduction, because its KL reference is intentionally different.

The pinned-stack CPU probe demonstrates the first behavior on exact versions:
mutating one trainable q-projection bias changes logits while adapters are
disabled, while the saved adapter contains zero bias tensors. Its self-hashed
receipt is
`experiments/sota_20260814/llmcosolver_f638_stack_probe.json`.

## Provenance that is still missing

Source defaults pin Qwen2.5-7B, LoRA rank/alpha 64, SFT and RL optimizer
defaults, seeds, and evaluation generation defaults. They do not pin the
released SFT/RL/evaluation rows, per-task row counts or hashes, exact reference
objectives, per-task initialization checkpoints, or an RL `max_steps` budget.
The default `checkpoint-31250` string is evidence of one example path, not a
per-task budget contract. One RL epoch is undefined without the missing
`train_rl.json` size.

Until those artifacts and a matched compute/token budget are frozen, neither
variant may be described as a matched baseline. The existing CLASP
`METHOD=grpo` lane is also not FOARL: it uses three strict heads and `dr_grpo`.

## Reproducing the semantic seal

Run the stack probe in an environment containing exactly torch 2.7.1,
Transformers 4.50.3, PEFT 0.17.0, and TRL 0.15.2. Then invoke
`scripts/freeze_llmcosolver_f638_contract.py` with the pinned upstream archive,
extracted source root, and package source roots. The freezer verifies every
upstream and dependency source hash, the vendored oracle, the probe self-hash,
and all local compatibility files before creating the output with exclusive
create semantics. It never overwrites an existing contract.

The resulting semantic contract is
`experiments/sota_20260814/llmcosolver_f638_semantic_contract.json`. Both
variants remain non-executable in that contract; choosing or launching one is
a separate publication decision.
