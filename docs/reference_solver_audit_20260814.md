# Five-task reference-solver audit

Audit date: 2026-08-14. Scope: the independent IID confirmation benchmark only.
This audit did not generate, attach, freeze, or evaluate any production labels.
The machine-readable record is
`experiments/sota_20260814/reference_solver_contract.template.json`.

## Decision

The reference stage is **not approved**. The contract remains
`BLOCKED_UNRESOLVED_DO_NOT_USE`, and all five `accepted_statuses` lists remain
empty. Therefore no current artifact supports an IID confirmation or SOTA
claim. Three exact lanes are technically available (OP, MIS, and JSSP), though
OP still awaits production certification, and the TSP heuristic is
reproducible. The five-task publication contract is not yet complete:

| Task | Audited state | Maximum defensible claim | Remaining hard issue |
|---|---|---|---|
| TSP | in-repository all-start NN + raw-distance 2-opt runner | gap to a deterministic heuristic reference | freeze parameters and audit all 1,000 rows |
| CVRP | in-repository raw-distance Clarke-Wright + 2-opt runner | gap to a deterministic heuristic reference | freeze parameters and audit all 1,000 rows |
| OP | raw-Euclidean exact Gurobi MIP runner; real token-license optimum smoke passed | certified optimum only for native `OPTIMAL` plus exact integer bound checks and strict replay | scalability audit, wheel path, then require all 1,000 rows `OPTIMAL` |
| MIS | Gurobi 13.0.1 evidence runner and token license operational | certified optimum when native status and integer bound checks pass | resolve wheel path and complete bounded run |
| JSSP | OR-Tools 9.15.6755 evidence runner in pinned offline environment | certified optimum only for CP-SAT `OPTIMAL` | complete bounded run; reject every non-OPTIMAL row |

An `OPTIMAL` claim is never inferred from feasibility, a small observed gap, a
solver name, or a time limit. TSP and CVRP must use `gap-to-reference`;
OP/MIS/JSSP may use `optimality gap` only for rows that pass
the native certificate rules below.

## Source identity, license, and runtime receipts

### TSP: elkai

- Official repository: [fikisipi/elkai](https://github.com/fikisipi/elkai).
  Release tag `v2.0.1` resolves to commit
  `a988d3187abbbab25e67c6911867c29f8c100150`.
- PyPI package: [elkai 2.0.1](https://pypi.org/project/elkai/2.0.1/).
  The CPython 3.10 Linux wheel SHA-256 is
  `b71b5b03286288c2cdb27bb3d18fcf1710e8371125affcfdd91495c2bc4d501b`;
  the source distribution SHA-256 is
  `1d8204b85fc960d2619c280e6143b14c1b62e09c532b388fff51e40e18404ba4`.
- The official license restricts elkai, and its bundled LKH, to academic and
  non-commercial use/distribution. It is not an OSI license.
- The audited Python 3.10.8 environment is
  `/lustre/fswork/projects/rech/vas/uag72kz/clasp/reference_solvers/elkai_2.0.1_audit/venv`.

`DistanceMatrix.solve_tsp(runs=10)` does not write a `SEED` parameter. The
bundled LKH-3.0.8 initializes `Seed=1`; only explicit `SEED=0` changes it to a
clock-derived value. Three repeated audit calls returned the same tour. The
frozen candidate parameters are therefore `RUNS=10`, effective `SEED=1`.
This establishes repeatability, not optimality: elkai describes this as an
approximate LKH solution and supplies no optimum certificate.

Production evidence does not depend on elkai. The reviewed
`raw_euclidean_tsp_heuristic_v1` adapter evaluates every deterministic nearest-
neighbor start, applies bounded raw-distance 2-opt, canonicalizes the depot,
and then replays the shared strict verifier. Its source file, parameters,
native log, output, and verifier result are hash-bound. It uses no random-number
generator, so the production contract has `seed_parameter=null` and no `seed`
field. It remains a heuristic.

### CVRP: LKH-3.0.8

- Official page: [LKH-3](https://webhotel4.ruc.dk/~keld/research/LKH-3/).
  The official `LKH-3.0.8.tgz` archive SHA-256 is
  `b5a82dfad55a380b2784faba85f24606e47bfcbf3d8443adf97b58b053ca3222`.
- The official README identifies version 3.0.8 (October 2022), permits
  distribution for research use, and says the author reserves all rights.
- The Jean Zay x86-64 binary is
  `/lustre/fswork/projects/rech/vas/uag72kz/clasp/reference_solvers/lkh_3.0.8_audit/LKH-3.0.8/LKH`,
  SHA-256
  `87016cd10c3739d80fd4aa7e2517b2bc3ddeeb673a23de0791bab6ecd9dbafc4`.
  The released LLMCoSolver wrapper SHA-256 is
  `fc6bb24f46edaae99a1bb9ed04c5afa4c7f30deae0cb22954d3d4a2beb32da24`.

The released wrapper writes `RUNS=10`, `TRACE_LEVEL=1`, `SPECIAL`, and
`SEED=0`. LKH interprets zero as clock seeding, so the historical invocation
was not reproducible. A hypothetical LKH reproduction would need a nonzero
fixed seed, but LKH is not the production reference runner.

There is also an objective mismatch. The wrapper writes integer coordinates
with `EDGE_WEIGHT_TYPE=EUC_2D`, so LKH searches using TSPLIB-rounded distances.
The shared CLASP verifier scores the returned routes with raw Euclidean
distance. This does not make a strictly verified route infeasible, but it means
LKH is optimizing a surrogate objective. The protocol must either adopt an
objective-aligned encoding/solver or explicitly disclose rounded-cost search
followed by raw-distance rescoring. Either choice remains a heuristic reference
and cannot certify an optimum.

The historical helper additionally leaves capacity validation as a TODO, and
expects a downloaded executable named `LKH.exe` although the official Linux
build produces `LKH`; it cannot serve as the production runner.

That production runner is now implemented independently of LKH as
`raw_euclidean_cvrp_heuristic_v1`: deterministic Clarke-Wright savings followed
by route-local raw-distance 2-opt and strict capacity/coverage replay. The
rounded LKH path is retained only as historical audit context. The new runner
uses no random-number generator, so its production contract has
`seed_parameter=null` and no `seed` field. It still supplies a heuristic
reference, so its parameters and full 1,000-row quality/runtime distribution
must be frozen before approval.

### OP: op-solver / EA4OP

- Official repository: [gkobeaga/op-solver](https://github.com/gkobeaga/op-solver).
  It has no release tags. The audited commit is
  `992a2443dd845c7a6f4952e1f28c51a4297e887b`; its GitHub-generated source
  archive SHA-256 is
  `2515ed7ff81758a28c40cf59cd8a07e7b818e323b0ad06e212758c4ae6537f8b`.
- The upstream license is Apache-2.0. Its heuristic EA4OP mode does not need an
  LP solver; exact relax-and-cut mode requires separately licensed IBM CPLEX.
  No CPLEX module was available in the audited Jean Zay stack.
- The heuristic binary is
  `/lustre/fswork/projects/rech/vas/uag72kz/clasp/reference_solvers/op_solver_992a2443_audit/op-solver-992a2443dd845c7a6f4952e1f28c51a4297e887b/build/src/.libs/op-solver`,
  SHA-256
  `a120ff1f4ff0885fc34f88f3e5a1729ba3699b4be9fbd9705a712c480b60b4af`.

The released path is not merely unpinned. `OPEnv.py` supplies both the depot
and the complete coordinate/prize arrays. `compass_solve.py` declares
`DIMENSION=len(loc)` but writes `[depot] + loc` and `[0] + prize`. The parsed
OPLIB instance therefore duplicates the depot, shifts prizes, and omits the
last intended node. Those wrapper files have SHA-256
`b9b24145c393aad9d7a63716bc6f96ddf3cb6a74ebe57e6cb6a10a86636d77e2`
and
`b631bc61f735a6554cb197064eb4ac9456a40353e06a9cf4cd880fa7087a58b5`,
respectively. The writer also truncates the float budget and requests rounded
`EUC_2D`, while the verifier uses the original float budget and raw distance.
Historical outputs from this path are not references for the displayed prompt
instances.

The audited upstream heuristic build did not pass a corrected ten-node smoke:
without `--stats` it aborted at the upstream `linkern.c` assertion
`best == sol->val`; with `--stats`, a heuristic-only build dereferenced a null
exact-solver environment. The default command is clock-seeded when `--seed` is
omitted; an explicit `--seed 0` is deterministic. This legacy path is not used.

The repository now contains a separate `gurobi_op_exact_v2` backend. It consumes the
structured coordinates, integer prizes, and frozen one-decimal budget directly;
models a depot-rooted open path ending at a zero-cost dummy sink; uses raw
float64 Euclidean arc lengths; excludes disconnected cycles with a
single-commodity flow; and strengthens the same integer feasible set with
resource-ordering indicators and valid pairwise budget-incompatibility
inequalities.
A deterministic feasible path is supplied only as a MIP start; it never
changes the acceptance rule. A four-node licensed smoke was certified optimal. On the
first 93-node IID row, the contemporaneous audit note records an approximately
60-second development probe with a strictly feasible prize-438 incumbent and
upper bound 504. The native log/evidence record was not retained, so its exact
termination code, runtime, work, and node count are not recoverable; it cannot
distinguish `WORK_LIMIT` from `NODE_LIMIT` and is inadmissible evidence. The
feasible incumbent and finite bound indicate computational non-closure, not
model infeasibility or an implementation exception. The exact adapter accepts only
native `GRB.OPTIMAL` (numeric 2), `SolCount>0`, an integral incumbent exactly
equal to the integral bound, reported gap zero, a retained nonempty native log,
and strict replay of the route, prize, depot, uniqueness, and raw-Euclidean
budget. `TIME_LIMIT`, `WORK_LIMIT`, `NODE_LIMIT`, and every other termination
abort the atomic evidence directory. Approval remains blocked until the
audited wheel path and resource policy are frozen and every one of the 1,000
rows passes this rule; no incumbent fallback is permitted.

### MIS: Gurobi

- Jean Zay provides `gurobi/13.0.1`; `gurobi_cl --version` reports Optimizer
  13.0.1. The site token service successfully solved an audit MIP.
- PyPI package: [gurobipy 13.0.1](https://pypi.org/project/gurobipy/13.0.1/).
  The CPython 3.10 Linux wheel SHA-256 is
  `7fc1b5ff46530aee391ab0f616620a1399abea0778b25e374b37f7ed222d5dfb`.
- The audited Python 3.10.8 environment is
  `/lustre/fswork/projects/rech/vas/uag72kz/clasp/reference_solvers/gurobi_13.0.1_audit/venv`.
  Gurobi is proprietary; execution uses Jean Zay's authorized academic/site
  license and does not make the solver redistributable.

A minimal maximum-independent-set smoke returned native `Status=2`, one
solution, integer objective 3, bound 3, and gap 0 with `Threads=1`, `Seed=0`,
`Method=1`, `TimeLimit=300`, `MIPGap=0`, `MIPGapAbs=0`,
`FeasibilityTol=1e-9`, `IntFeasTol=1e-9`, `NumericFocus=3`, retained output,
and no console mirroring. Per the
[official status documentation](https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html),
`OPTIMAL` is subject to solver tolerances. Production acceptance must therefore
require native `GRB.OPTIMAL`, `SolCount>0`, exact equality of the integral
incumbent and integral bound, zero reported gap, and strict independent-set
verification. The released wrapper does none of the status/bound checks and
must not be reused unchanged.

### JSSP: OR-Tools CP-SAT

- PyPI package: [OR-Tools 9.15.6755](https://pypi.org/project/ortools/9.15.6755/),
  Apache-2.0. Upstream tag `v9.15` resolves to commit
  `551ad10d94835c99e5e1e684500d3db398c0e345`.
- The CPython 3.10 Linux OR-Tools wheel SHA-256 is
  `03424136aa48555e7f4d1bc73edeb99f80ec35a2e5f17700e2072640344980fc`.
- The fully offline Python 3.10.8 environment and wheel directory are under
  `/lustre/fswork/projects/rech/vas/uag72kz/clasp/reference_solvers/ortools_9.15.6755_audit/`.

The frozen candidate parameters are one search worker, random seed 1, a
300-second limit, zero absolute/relative gap limits, and retained search-progress
logging (`log_search_progress=true`). A smoke schedule
returned `OPTIMAL` with objective equal to best bound. The
[official CP-SAT documentation](https://developers.google.com/optimization/cp/cp_solver)
distinguishes `OPTIMAL` from `FEASIBLE`: the latter is not known to be optimal.
The released code accepts both, which is unsuitable for an optimum reference.
Production acceptance must require native `OPTIMAL`, exact equality of the
integer makespan and best bound, strict schedule verification, and retained
response statistics. Whether all 1,000 frozen instances certify within the
fixed cap remains an empirical blocker; no long solve was run in this audit.

The offline wheel receipt is:

| Package | Version | SHA-256 |
|---|---:|---|
| absl-py | 2.5.0 | `0f17b89f2a4eaaedc4f28c622998aa690564b3012a396a4ffad0821007fe03ba` |
| immutabledict | 4.3.1 | `c9facdc0ff30fdb8e35bd16532026cac472a549e182c94fa201b51b25e4bf7bf` |
| numpy | 2.2.6 | `fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915` |
| ortools | 9.15.6755 | `03424136aa48555e7f4d1bc73edeb99f80ec35a2e5f17700e2072640344980fc` |
| pandas | 2.3.2 | `cc03acc273c5515ab69f898df99d9d4f12c4d70dbfc24c3acc6203751d0804cf` |
| protobuf | 6.33.6 | `e9db7e292e0ab79dd108d7f1a94fe31601ce1ee3f7b79e0692043423020b0593` |
| python-dateutil | 2.9.0.post0 | `a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427` |
| pytz | 2026.3.post1 | `dd95840dd199baea12d9cc096a1d452caa6596a1c1e4b5f3dbd1541855d5e815` |
| six | 1.17.0 | `4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274` |
| typing-extensions | 4.16.0 | `481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8` |
| tzdata | 2026.3 | `dc096730c87af6cab1b171c9d532be840741ff5d459015e7f6947bd7d7e54931` |

## Required evidence before approval

The repository now contains the fail-closed runner/schema in
`scripts/run_reference_evidence.py`. It hash-binds:

1. benchmark row and semantic instance hashes;
2. source/package and executable hashes, native version, license class, and
   complete parameters including resource limits and a seed only for adapters
   that actually consume one;
3. native termination status, incumbent, best bound where available, gap,
   runtime, and solver log/response statistics;
4. independently recomputed feasibility and objective under benchmark
   semantics; and
5. the exact task contract hash and the active evidence-runner source hash.

For Gurobi and OR-Tools wheel-backed adapters, the finalized identity also binds
the distribution name, absolute imported module origin, and a SHA-256 manifest
of the installed distribution. A wheel filename/hash alone is insufficient.

Assembly and freeze now preserve each native evidence record hash, bind each
task summary and export record, and carry those identities through row
provenance, the generation receipt, and the schema-v3 manifest. The shared
`clasp.data.benchmark.validate_frozen_benchmark_manifest` validator is called by
freeze, confirmation evaluation, and publication aggregation. Dropping native
status/bound/log evidence or resealing only an outer manifest is rejected.

The implementation and its adversarial fake-backend tests do not approve a
solver or constitute production evidence. Only after exact artifact
paths/hashes, installed-distribution identities, the active runner hash, and
the active verifier identity are frozen, reviewers approve each adapter, and
every task policy is closed may reviewers populate `accepted_statuses`, set
status `approved_for_reference_evidence`, and self-hash a finalized contract.
That contract then remains immutable throughout solving. Until then, the runner
and `assemble` correctly refuse the checked-in template. OP's serializer and
metric defects are now isolated from the exact adapter, but OP remains a
production-evidence block until a preregistered all-OPTIMAL run succeeds.

After all five 1,000-row evidence trees exist, `approve-evidence` must reopen
and validate them together and create the separate create-once receipt with
status `approved_for_confirmation`. `export-references` must receive that
receipt through `--approval-receipt`; neither evidence approval nor export may
mutate the solver contract.

Approval does not authorize retrofitting the immutable v1 IID draft or source
snapshots v1/v2. Those artifacts predate the current evidence and identity
schemas and remain smoke/stale historical receipts. After approval, create a
new source snapshot and a new IID version. Each evidence row and downstream
receipt must bind `semantic_instance_sha256`, `benchmark_row_sha256`, exact
`reference_output_sha256`, the per-task solver-contract hash, and the complete
native certificate/log evidence. Only the newly assembled receipt and
pre-inference schema-v3 manifest may enter confirmation.
