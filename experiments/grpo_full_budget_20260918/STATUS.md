Live monitoring moved to [RLOO recovery status](../rloo_recovery_20260927/STATUS.md).

The previous collector stopped after the OP evaluation inventory failure. The replacement reads fresh scheduler and completion evidence and keeps monitoring after controller stops.

See the [repair and recovery record](../rloo_recovery_20260927/README.md).


## Continuation v2 (2026-11-23): FOARL MIS + OP

The 2026-11-13/14 prune destroyed the FOARL MIS/OP adapters, chunk receipts and
the campaign_v1 initializer receipts, so neither arm can be scored at its
declared horizon from what survived. `launch_declaration_v2.json`
(sha256 `ffeb3cc087fa90769d1607153fb785eed26858173530027b9c586fd939c1849a`)
declares a scoped continuation — protocol
`budget-matched-grpo-development-20260918-v2-continuation`, scope
`{tasks: [mis, op], methods: [foarl]}`, 44 jobs, 344 reserved A100-hours —
restarting both arms from the same shared step-31250 SFT initializers.

Recorded deviations, all explicit in the declaration:

- `initialization_attestation: checkpoint_fidelity_adapter_bytes` — the OP
  initializer's `trainer_state.json` (and its vocab/merges/special-token files)
  were pruned; the trainer never reads the trainer state, and the starting
  policy is attested by the surviving `checkpoint_fidelity.json` receipt. The
  MIS initializer is intact.
- Per-method diagnostics: the first training chunk waits for its own 5->10
  diagnostic instead of all four arms' diagnostics (a single-arm continuation
  has nothing to compare against).
- Evaluation bindings for mis/op point at the frozen `data_eval` DEV100 cohort
  (the slice every published row was measured on); the v1-bound benchmark_v1
  files were pruned on 2026-11-12.
- `grpo_quality_signal_20260916/launch_declaration_v3.json` (sha-verified
  binding, pruned from /lustre) was restored byte-identically from the
  repository copy.
- Staging note: a first source_v2 bundle was staged and failed its native CPU
  preflight (the trainer-side contract validator still pinned the v1 protocol);
  the unreferenced staging was deleted and re-staged after the fix.

Final scoring uses the standalone matched-protocol evaluator (DEV100 data_eval,
raw8, seed 271828, jiang_f638), identical to every published row.
