The verified-improvement GRPO pilot exposed a generation/training lifecycle bug
in the pinned TRL 0.24 runtime. The entrypoint requests gradient checkpointing
with `use_reentrant=False`. TRL's `unwrap_model_for_generation` disables
checkpointing, then restores it with `gradient_checkpointing_enable()` without
the original arguments. Transformers therefore reinstalls reentrant
checkpointing. The configuration saved in `hyperparams.json` describes the
requested mode, not the mode actually used after generation.

The current auxiliary loss performs policy and teacher forwards before a shared
backward. Reproducing the real TRL generation context makes that path leave
different gradients on different ranks. The matched GRPO single-forward path
stays synchronized. Earlier CPU and A100 probes omitted this generation
lifecycle and consequently passed, including many-bucket and mixed-precision
Accelerate probes. A bare multi-forward test was insufficient regression
coverage.

Evidence:

- Original 100-update modified CVRP/MIS jobs 1816053/1816059 failed the final
  exact policy check; both classic controls completed.
- A100 diagnostic 1824871, using the real MIS SFT-LoRA checkpoint and five-update
  plan, failed before optimizer update one. All 476 trainable gradient tensors
  had different hashes across four ranks. The matched control 1824870 passed
  gradient and policy checks at all five updates and completed.
- A tiny Qwen/LoRA reproducer using the actual TRL context, Accelerate, gradient
  accumulation and 25 DDP buckets reproduced numerical gradient divergence
  before clipping: maximum absolute difference 0.111328125 at update one.
  Reapplying the configured checkpointing arguments removed divergence in the
  same reproducer. This is an infrastructure result, not a feasibility/gap result.

The local fix reapplies the declared checkpointing arguments immediately after
each generation/scoring call. It runs in `finally`, preserving the training
configuration on exceptional exits too. Both classic and modified GRPO use the
same correction; no external TRL installation is modified. The checkpoint and
distributed-consistency checks remain strict.

`scripts/probe_verified_grpo_distributed.py` now exercises `none`, `trl` and
`restored` generation lifecycles, with byte-exact checks of gradients, policy
weights and optimizer state. `trl` reproduces the dependency behavior;
`restored` applies the repository correction. Native four-A100 MIS diagnostic
1825156 completed five updates in 544 seconds, with exact gradient and policy
agreement at every update. The fresh 100-update comparison applies the same
correction to both arms and requires exact policy agreement at steps 1, 2, 5,
every tenth update, and the saved endpoint. The diagnostic validates the
correction; task performance is evaluated separately.

[Independent native diagnostic review](../analysis_outputs/grpo_verified_ddpfix_20260906_v2/full_model_diagnostic_v3/native_checkpoint_fix_independent_review.json)
and [current comparison status](../analysis_outputs/grpo_verified_ddpfix_20260906_v2/STATUS.md).

Independent of this bug, an offline operator audit found 103/134 CVRP teacher
targets worse than the same prompt's best raw candidate. Filtering teacher
targets against that incumbent is a separate algorithm change. It must be
declared and evaluated separately from evidence that checkpointing is correct.
