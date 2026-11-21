# Independent IID confirmation benchmark

Audit date: 2026-08-14. This document records what can and cannot be
reconstructed from the LLMCoSolver paper and released repository. It is the
operational companion to `experiments/sota_20260814/PROTOCOL.md`.

## Bottom line

The historical 300 evaluation rows cannot serve as an unseen confirmation set.
The repository now has a deterministic two-phase path for a new 1,000-instance
set per task, semantic overlap checking against prompt-only and structured
historical rows, strict reference validation, and a pre-inference content-addressed
manifest. It deliberately does **not** claim that the new generator is
byte-identical to the undocumented historical invocation.

The production reference-solving stage remains blocked. The released code does
not pin or ship several solver dependencies, and some code paths accept a
feasible incumbent without an optimality certificate. The checked-in solver
contract therefore has status `BLOCKED_UNRESOLVED_DO_NOT_USE`; `assemble`
refuses it. Consequently, there is no frozen confirmation benchmark and no
defensible SOTA claim.

The unlabeled production draft was generated before any model inference on
2026-08-14 and archived read-only at:

```text
/lustre/fswork/projects/rech/vas/uag72kz/clasp/benchmarks/
  iid_confirmation_v1_unlabeled
```

It contains exactly 1,000 rows for each of the five tasks. Its schema-v1
generation record stores the historical receipt identifier
`2e5d7b75aed3a0da7b5bec807ddcb3b128d96a71d115ce9e447ac5585f81c56c`;
the independently recorded archive copy is
`experiments/sota_20260814/iid_generation_receipt.json`. That copy is not a
valid receipt under the current remove-the-hash-field self-hash rule: its
observed unsigned digest is
`9ee000a5fd1c29d45430ef5a4921a1347e39a481bc5bf01cf1f3ca439dd18d4e`.
No current assembly path consumes schema v1; preserve the record unchanged as
historical lineage rather than repairing it in place. It remains
`publishable=false` and is now a **stale historical draft**, not a draft to
upgrade. It predates generation-receipt schema v2, explicit ordered row/reference
identities, and the fix that makes the stored OP budget equal the displayed
one-decimal budget. Keep this read-only archive as evidence; do not attach
production labels to it. Generate a new version after the reference contract is
approved and freeze that version into a schema-v3 manifest.

The campaign's source snapshot v1 (training smoke) and v2 (evaluation/benchmark
smoke, recorded by `source_snapshot_v2.json`) are also immutable. The current
identity and checkpoint-selection gates postdate v2, so neither snapshot is a
confirmation environment. A new source snapshot with a new manifest hash is
required; never rewrite v1/v2 and retain their jobs as smoke evidence only.

## Primary-source audit

The audited source is the official MIT-licensed LLMCoSolver repository at
commit `f6388991d243c0aa91d9ee7e2fa18ed1c9e72477` and the paper source for
[arXiv:2509.16865](https://arxiv.org/abs/2509.16865). The immutable local
snapshot receipt is:

```text
/lustre/fswork/projects/rech/vas/uag72kz/clasp/jiang_reproduction_assets/
  llmcosolver_f6388991_20260716_v1/snapshot_receipt.json
```

The generator contract records the SHA-256 of the MIT license and each relevant
environment file. Non-smoke generation requires `--official-source-root` and
recomputes every one of those hashes.

The evidence supports the following distributions:

| Task | Released generator interpretation | Released reference path |
|---|---|---|
| TSP | node coordinates from uniform, GM(2,5), or GM(3,10), scaled and truncated to integer; released code weights size `n in [10,100]` proportionally to `n` | `elkai.DistanceMatrix(...).solve_tsp(runs=10)` |
| CVRP | uniform customer count 10--100 plus depot; same three coordinate families; integer demands 1--10; `int(Triangular(3,6,25) * mean demand)` capacity | auto-downloaded LKH-3.0.8, `RUNS=10`, `SEED=0` |
| OP | uniform node count 10--100; same coordinate families; prizes 1--10 with depot overwritten to zero; sample route budget uniformly in `[0.5 L_NN, 0.7 L_NN]`, then round once to the one-decimal value used by both prompt and verifier | Compass/EA4OP command `op-solver opt --op-exact 0` |
| MIS | uniform node count 10--100; equal ER/BA choice; ER `p~U[0.1,0.4]`; BA `m~U{1,2,3,4}` | Gurobi binary model |
| JSSP | jobs and machines independently uniform 5--20; processing times uniform integers 1--99; random machine permutation per job | OR-Tools CP-SAT, one worker, 300-second limit |

There are material ambiguities and inconsistencies:

- The paper says TSP sizes are randomly sampled over 10--100, but
  `TSPEnv.py` uses weights proportional to the size. The first released test
  row (`n=81`, uniform coordinates beginning `[374,950]`) exactly follows the
  seed-42 weighted code path. The contract follows this observed released-code
  behavior and says so explicitly.
- The paper describes CVRP capacity as the average demand; released code uses a
  triangular multiplier. The contract follows code.
- OP depends on an unpinned `pyCombinatorial` nearest-neighbor implementation,
  absent from the released `requirements.txt`. The new contract freezes the
  chosen start node, return edge, and tie-breaking rather than pretending the
  missing version is known.
- The first released CVRP and OP rows reproduce the seed-42 size/coordinate
  stream. The first MIS row reproduces the paper configuration with seed 6.
  The released JSSP first instance corresponds to a child seed of 93633, but
  the command/master-seed provenance needed to derive it was not released.

## Why the reference stage is blocked

`experiments/sota_20260814/reference_solver_contract.template.json` names the
missing decisions instead of silently filling them in. The detailed
[five-task solver audit](reference_solver_audit_20260814.md) now pins the
available official versions, source/package hashes, licenses, deterministic
parameters, Jean Zay artifacts, and native certification rules.

- TSP is reproducible as `elkai==2.0.1` with effective LKH seed 1, but remains
  a heuristic reference without an optimum certificate.
- CVRP's official LKH-3.0.8 archive and Jean Zay binary are pinned. The released
  `SEED=0` is clock-based, and its rounded `EUC_2D` search objective differs
  from the raw Euclidean benchmark score.
- The released OP path remains unusable: its writer changes the instance and
  the corrected upstream heuristic smoke aborted. The in-repository
  `gurobi_op_exact_v2` backend instead consumes the structured instance under
  the raw metric. It exports a row only after native `OPTIMAL`, exact integral
  incumbent/bound equality, zero gap, and strict route replay. The audit note
  for the first 93-node development probe records feasible prize 438 and bound
  504 but no retained native log or exact termination code. It did not close
  its gap, so the production lane
  remains blocked pending a preregistered scalability audit and 1,000/1,000
  certified rows; there is no heuristic fallback.
- Gurobi 13.0.1 and its Jean Zay license are operational for MIS. A production
  row is certifiable only with native `OPTIMAL`, a retained incumbent/bound,
  exact integral equality, zero gap, and strict feasibility verification.
- OR-Tools 9.15.6755 is installed in a hash-pinned offline environment. JSSP
  may accept only native `OPTIMAL` with integer objective equal to best bound;
  the released code's `FEASIBLE` case is a best-known incumbent, not an optimum.

The repository now includes `scripts/run_reference_evidence.py`, a fail-closed
two-phase evidence runner. Every row binds native status, incumbent, bound,
gap, exact parameters and seed, package/binary identity, raw native-log
identity, active strict-verifier result, semantic and benchmark-row identities,
and a record self-hash. A self-hashed summary binds the details file and all
native logs. `export-references` independently reopens and revalidates the
immutable draft, contract, logs, records, and strict metrics before producing
assembly inputs.

This implementation does **not** approve a solver or create production
evidence. No bounded 1,000-instance run was performed. The checked-in template
therefore remains `BLOCKED_UNRESOLVED_DO_NOT_USE`, contains explicit unresolved
fields and empty `accepted_statuses`, and cannot pass the runner. OP's exact
adapter is code-complete, but the all-five-task production contract remains
hard-blocked until its scalability and all-OPTIMAL evidence gates close.

For TSP/CVRP, report `gap-to-reference`. For OP/MIS, accept only Gurobi
`OPTIMAL` with incumbent equal to bound and zero gap. For JSSP, either accept
only `OPTIMAL`, or retain status, incumbent, best bound, and call a time-limited
`FEASIBLE` result a best-known reference. Never relabel it as ground-truth
optimality.

## Two-phase workflow

After the solver gate is resolved and a fresh source snapshot exists, create an
isolated CPU environment with the generator pins in
`iid_generator_contract.json` (`numpy==2.3.2`, `networkx==3.5`). Then generate
the unlabeled instances into a **new** versioned destination. The destination
must not already exist; the v1 archive is not a valid target.

```bash
PYTHONPATH=src python scripts/generate_iid_benchmark.py generate \
  --generator-contract experiments/sota_20260814/iid_generator_contract.json \
  --official-source-root /lustre/fswork/projects/rech/vas/uag72kz/clasp/jiang_reproduction_assets/llmcosolver_f6388991_20260716_v1/official_pristine \
  --forbidden-source-registry /path/to/frozen_forbidden_source_registry_v1.json \
  --count 1000 \
  --output-dir /path/to/new/iid_confirmation_v2_unlabeled
```

Each task is written to `<task>/eval/test.json` with `instruction`, `input`, and
`instance_json`, but no `output`. Every row has a deterministic
`benchmark_row_id`, compatibility `instance_sha256`, and explicit
`semantic_instance_sha256`. Production generation first validates the complete
registry described under "Leakage audit and freeze" below and binds its compact
identity into the schema-v2 self-hashed generation receipt. Assembly propagates
that identity, and the final freeze requires exact equality with the embedded
registry snapshot. The receipt also retains the ordered semantic-instance and
normalized benchmark-row identities. The raw
serialized-instance hash used later by evaluation remains a separate field and
is never substituted for semantic identity. Consequently, the draft is not a
loadable evaluation set and cannot expose model metrics prematurely.

After review, create a finalized, self-hashed schema-v2 solver contract with
status `approved_for_reference_evidence`, the active verifier identity, the
active `scripts/run_reference_evidence.py` SHA-256 in `runner_code_sha256`, and
exact artifact paths/hashes. This execution contract is immutable: do not
change its status after solving starts. Run each approved task into a new
atomic evidence directory, repeating the command for all five tasks:

```bash
PYTHONPATH=src python scripts/run_reference_evidence.py solve-task \
  --draft-root /path/to/new/iid_confirmation_v2_unlabeled \
  --solver-contract /path/to/final_reference_solver_contract.json \
  --task mis \
  --output-dir /path/to/evidence/mis
```

Then independently reopen exactly five complete 1,000-row evidence trees and
create the one all-five approval receipt. The receipt, not the immutable solver
contract, has status `approved_for_confirmation`:

```bash
PYTHONPATH=src python scripts/run_reference_evidence.py approve-evidence \
  --draft-root /path/to/new/iid_confirmation_v2_unlabeled \
  --solver-contract /path/to/final_reference_solver_contract.json \
  --evidence tsp,/path/to/evidence/tsp \
  --evidence cvrp,/path/to/evidence/cvrp \
  --evidence op,/path/to/evidence/op \
  --evidence mis,/path/to/evidence/mis \
  --evidence jssp,/path/to/evidence/jssp \
  --output /path/to/evidence/reference_evidence_approval.json
```

Only after that receipt exists, independently reopen and export the completed
evidence. This step still does not assemble, freeze, or run inference:

```bash
PYTHONPATH=src python scripts/run_reference_evidence.py export-references \
  --draft-root /path/to/new/iid_confirmation_v2_unlabeled \
  --solver-contract /path/to/final_reference_solver_contract.json \
  --evidence tsp,/path/to/evidence/tsp \
  --evidence cvrp,/path/to/evidence/cvrp \
  --evidence op,/path/to/evidence/op \
  --evidence mis,/path/to/evidence/mis \
  --evidence jssp,/path/to/evidence/jssp \
  --approval-receipt /path/to/evidence/reference_evidence_approval.json \
  --output-dir /path/to/new/iid_confirmation_v2_references
```

Each compact export record contains:

```json
{
  "benchmark_row_id": "tsp-000000",
  "instance_sha256": "<64 hex characters>",
  "semantic_instance_sha256": "<same semantic hash, retained explicitly>",
  "benchmark_row_sha256": "<normalized unlabeled row hash>",
  "output": "Route: [...], Objective: ...",
  "solver_status": "BEST_KNOWN_HEURISTIC_VERIFIED",
  "solver_runtime_seconds": 1.23,
  "solver_contract_task_sha256": "<hash of that task's contract object>",
  "reference_evidence": {
    "native_status": "HEURISTIC_COMPLETE",
    "native_incumbent": 123.4,
    "native_best_bound": null,
    "native_gap": null,
    "raw_native_log": {"sha256": "<hash>", "size_bytes": 123},
    "reference_evidence_record_sha256": "<hash>",
    "reference_evidence_summary_sha256": "<hash>"
  },
  "reference_export_record_sha256": "<self-hash>"
}
```

TSP/CVRP are always labeled `best_known_heuristic`, never optimal, and carry
no native bound or gap. Their implemented adapters search and rescore the raw
Euclidean objective used by CLASP; rounded TSPLIB objectives are rejected.
Both adapters are deterministic and expose no synthetic seed: their contract
`seed_parameter` is `null`, and their exact parameter objects contain only the
fields consumed by the adapter. OP and MIS require Gurobi `OPTIMAL` (numeric
status 2), `SolCount>0`, integral
incumbent equal to bound, and zero gap. OP additionally replays the open route
against the raw-Euclidean budget. JSSP requires CP-SAT `OPTIMAL` (numeric
status 4) with integral objective equal to best bound. Every merely feasible
or resource-limited result is rejected.

After the immutable `approved_for_reference_evidence` contract, all-five
`approved_for_confirmation` receipt, and receipt-bound reference export have
all passed revalidation, assemble the CLASP-compatible set:

```bash
PYTHONPATH=src python scripts/generate_iid_benchmark.py assemble \
  --draft-root /path/to/new/iid_confirmation_v2_unlabeled \
  --references-root /path/to/new/iid_confirmation_v2_references \
  --generator-contract experiments/sota_20260814/iid_generator_contract.json \
  --solver-contract /path/to/final_reference_solver_contract.json \
  --output-dir /path/to/new/iid_confirmation_v2
```

Assembly is atomic and refuses overwrite. Production assembly additionally
requires `reference_export_summary.json`, validates every export self-hash and
native certificate, and preserves the complete evidence in each row's
`reference_provenance` plus ordered evidence identities in the schema-v2
receipt. The production freeze rejects dropped or altered evidence. Because no
frozen all-OPTIMAL OP production evidence exists yet, the all-five-task
production assembly gate intentionally cannot pass.

The canonical downstream gate is
`clasp.data.benchmark.validate_frozen_benchmark_manifest(manifest, *,
expected_tasks=None, require_generation_receipt=True,
require_reference_evidence=True)`. Confirmation evaluation and publication
aggregation use the strict defaults. The freeze tool uses the same validator;
its relaxed flags are reserved for explicitly non-publishable smoke/legacy
manifests.

## Leakage audit and freeze

Before generating or opening any IID confirmation instances, freeze the complete
forbidden-source registry. Point each source at the authoritative dataset file
or directory, not a convenience subset. A directory is recursively enumerated;
the registry records every JSON file's relative path, binary SHA-256, byte size,
and row count, plus a self-hash for the complete task/label matrix:

```bash
PYTHONPATH=src python scripts/freeze_forbidden_source_registry.py \
  --source tsp,sft-train,/path/data/tsp/train \
  --source tsp,rl-train,/path/data_rl/tsp/train \
  --source tsp,historical-eval,/path/data_rl/tsp/eval \
  --source cvrp,sft-train,/path/data/cvrp/train \
  --source cvrp,rl-train,/path/data_rl/cvrp/train \
  --source cvrp,historical-eval,/path/data_rl/cvrp/eval \
  --source op,sft-train,/path/data/op/train \
  --source op,rl-train,/path/data_rl/op/train \
  --source op,historical-eval,/path/data_rl/op/eval \
  --source mis,sft-train,/path/data/mis/train \
  --source mis,rl-train,/path/data_rl/mis/train \
  --source mis,historical-eval,/path/data_rl/mis/eval \
  --source jssp,sft-train,/path/data/jssp/train \
  --source jssp,rl-train,/path/data_rl/jssp/train \
  --source jssp,historical-eval,/path/data_rl/jssp/eval \
  --output /path/to/frozen_forbidden_source_registry_v1.json
```

The command defaults to all five tasks and therefore requires exactly 15 unique
task/label specifications. It refuses an omitted, extra, duplicate, empty, or
unparseable tree and refuses to overwrite an existing receipt. Publish both
`registry_sha256` and `source_tree_manifest_sha256` before IID generation. The
draft shape is documented in
`experiments/sota_20260814/forbidden_source_registry.template.json`; it is not a
valid frozen receipt.

`scripts/freeze_benchmark.py` streams top-level JSON arrays, so all 500,000-row
SFT files can be audited without loading them simultaneously. It constructs a
task-semantic identity through the strict evaluator's parsers. This is crucial:
CVRP, OP, MIS, and JSSP SFT rows often contain only the rendered prompt, while
RL rows retain the structured instance. Prompt-only and structured copies now
collide. OP route budgets are hashed at the one-decimal precision exposed by
the released prompt, closing another representation loophole.

Supply every SFT train, RL train, and historical 300-row evaluation file as a
`--forbidden TASK,LABEL,PATH` argument. A production invocation has this form:

```bash
PYTHONPATH=src python scripts/freeze_benchmark.py \
  --test tsp,iid-confirmation-new,/path/to/new/assembled/tsp/eval/test.json \
  --test cvrp,iid-confirmation-new,/path/to/new/assembled/cvrp/eval/test.json \
  --test op,iid-confirmation-new,/path/to/new/assembled/op/eval/test.json \
  --test mis,iid-confirmation-new,/path/to/new/assembled/mis/eval/test.json \
  --test jssp,iid-confirmation-new,/path/to/new/assembled/jssp/eval/test.json \
  --forbidden tsp,sft-train,/path/data/tsp/train/train_tsp.json \
  --forbidden tsp,rl-train,/path/data_rl/tsp/train/train_rl.json \
  --forbidden tsp,historical-eval,/path/data_rl/tsp/eval/test.json \
  --forbidden-source-registry /path/to/frozen_forbidden_source_registry_v1.json \
  --generator-contract experiments/sota_20260814/iid_generator_contract.json \
  --solver-contract /path/to/final_reference_solver_contract.json \
  --generation-receipt /path/to/new/assembled/generation_receipt.json \
  --minimum-instances-per-task 1000 \
  --output /path/to/new/preinference_iid_manifest_v3.json
```

Repeat the three forbidden arguments for every task with its actual filenames.
Production requires the exact labels `sft-train`, `rl-train`, and
`historical-eval` for every frozen task. Every supplied tree must exactly match
the pre-frozen registry: a non-empty decoy subset, omitted shard, added shard,
renamed relative path, changed byte, substituted dataset, or duplicate label is
rejected. The full self-hashed registry and compact identity are embedded in
the benchmark manifest. The shared schema-v3 validator revalidates both and
cross-binds them to every forbidden-source file manifest; campaign aggregation
independently binds the registry and tree hashes. It also requires
`overlap_audit.tested_forbidden_sources` to equal the exact source manifest
count. The freeze refuses a missing or old-schema generation receipt,
solver-contract or verifier mismatch, missing/per-row solver provenance,
duplicate IID row, any unparseable forbidden row, content collision, invalid
reference, ordered semantic/row/reference mismatch, existing output, or fewer
than 1,000 rows per task. The schema-v3 manifest retains both semantic identity
and the exact row and reference sequences needed by the confirmation
aggregator.

Publish the final manifest SHA-256 before running `clasp-eval`. The manifest and
assembled test directory must then be read-only for the confirmation run. A GRPO
confirmation must use the exact selected checkpoint and pass
`--evaluation_purpose confirmation --benchmark_manifest <manifest>
--checkpoint_selection_receipt <receipt> --no-prefer_subset_files`, with no row
cap, offset, or size filter. Until all placeholders above refer to reviewed,
create-once assets, do not launch confirmation.

## CPU smoke evidence

A concrete five-task, one-row smoke passed locally on 2026-08-14:

1. `generate --count 1 --smoke` created deterministic unlabeled rows for all
   five tasks.
2. `smoke-references` created feasibility-only references explicitly marked
   `smoke_only`.
3. `assemble --smoke` passed the shared strict task evaluators for all five.
4. `freeze_benchmark.py --nonpublication-mode --allow-smoke-receipt
   --minimum-instances-per-task 1`
   bound the assembled ordered semantic, row, and reference hashes to a
   self-hashed schema-v3 manifest.

The same workflow is covered by `tests/test_iid_benchmark_generation.py`.
Smoke artifacts have `publishable=false`; both the receipt escape hatch and a
missing registry are accepted only with explicit `--nonpublication-mode`.
Those manifests carry `benchmark_mode=nonpublication` and are rejected by
confirmation evaluation and publication aggregation.
