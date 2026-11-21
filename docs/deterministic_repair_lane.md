# Deterministic repair-and-improve lane

`clasp-repair` evaluates a separate hybrid system: a frozen raw generator followed by
CLASP's deterministic, task-specific repair and bounded local improvement. Results from
this lane must never be reported in the raw-generation column.

The command accepts only completed, fingerprinted `clasp-eval` output. It verifies the
source run config, experiment and protocol fingerprints, dataset row locators, embedded
raw-instance and transformed-row hashes, semantic instance identity, normalized benchmark
row identity, reference-output identity, strict-verifier identity, candidate counts, and
any candidate-specific TSP label-TTA contexts before doing work. Legacy or ambiguous
records fail closed.

```bash
clasp-repair \
  --details /path/to/run_details__...jsonl \
  --source_config /path/to/run_config__...json \
  --output /path/to/repaired_details.jsonl \
  --system_name clasp_checkpoint_name_repair_v1
```

The output retains every source record and raw completion verbatim as JSON values, then
stores independently re-verified raw and repaired metrics, the repaired completion,
actions, status, and wall-clock repair time. Both the JSONL and summary are written
atomically. Their fingerprint covers the source config and details files, repair settings,
repair implementation, and strict verifier implementation.

The native evaluation remains the semantic evidence root. Each derived summary therefore
embeds a self-hashed portable snapshot of the exact completed source config and, for a
beta-positive run, integrity-checked copies of the root/selected step-zero and
step-one/post-step exact-KL receipts bound by that config. Publication validation does not
depend on an absolute path or same-directory basename coincidence after an exact artifact
relocation. When the original source config is still available at its declared path (or an
exact co-located copy exists), the aggregator reopens and rehashes it as an additional
operational check. The repair record also stores the verbatim raw record and its hash,
source experiment/protocol fingerprints, source-row locator, raw representation
`instance_sha256`, and separate `semantic_instance_sha256`, `benchmark_row_sha256`, and
`reference_output_sha256`. Preserve the native config, summary, CSV, and details whenever
possible, but their path is not the sole authority for an otherwise closed derived bundle.

Version 1 uses cheapest insertion plus bounded 2-opt for TSP; capacity-respecting route
completion plus bounded relocation/exchange/2-opt for CVRP; budget repair, open-path 2-opt,
and prize insertion for OP; conflict removal plus greedy maximalization for MIS; and serial
schedule generation plus bounded adjacent-swap descent for JSSP. Repair functions receive
the instance and raw candidate, but no reference solution. Ground truth is available only
to the shared verifier for reporting the same objective-gap metric as raw evaluation.

## Reporting and limits

- Label this result **deterministic repair-assisted hybrid**, not raw CLASP.
- Keep it separate from both the raw K=8 lane and any adaptive-repair lane. Adaptive
  repair has a different stopping/compute contract and must report actual candidate count,
  generated tokens, GPU time, and wall time; its values never replace fixed-K=8 values.
- Report repair runtime in addition to model generation runtime and use identical candidate
  budgets for matched comparisons.
- Do not describe this implementation as FALCON-equivalent. It does not reproduce FALCON's
  grammar, adaptive controller, BOPO training, or proprietary/unreleased implementation.
- The heuristics guarantee strict feasibility for supported, well-formed benchmark
  instances or abort the artifact; they do not guarantee globally optimal objectives.
- Changing any repair pass limit defines a different fingerprinted system configuration.
