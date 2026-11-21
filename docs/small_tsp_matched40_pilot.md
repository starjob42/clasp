# Small-TSP matched 40-step pilot

This is a development-only mechanism screen. It is not confirmatory,
publication-eligible, or SOTA evidence. The static contract is
`experiments/additional_evidence/small_tsp_canonical_v1/matched40_v1.contract.json`.
It binds the 90-example training split, completed 80-step Qwen2.5-0.5B SFT
adapter, implementation files, runtime, decoder, and four 40-step arms:

- canonical CLASP with rollout `B=9`, deployment `K=8`, and failure@8 controller;
- the same canonical objective with the controller frozen at lambda zero;
- a strong scalarized component-dual baseline; and
- standard three-head GRPO.

All arms use policy seeds 101, 202, and 303, mapped to physical GPUs 0, 1,
and 2. Crucially, `GRPO_DATA_SEED` remains fixed at `20260815` for every arm
and policy seed, matching the already completed canonical runs. The launcher
sets `PYTHONHASHSEED` and the trainer `--seed` from the policy seed.

## Preflight

Run the read-only preflight before every batch:

```bash
../my_idea_grpo/.venv/bin/python scripts/run_small_tsp_matched40_pilot.py \
  preflight --arm all --seed 101
```

Repeat with `--seed 202` and `--seed 303` to check all create-once destinations.
The preflight verifies artifact hashes and runtime versions, resolves every CLI,
checks the matched fields across arms, and compares the planned canonical CLI
against the completed seed-101 `hyperparams.json`. That completed-run comparison
must report an empty `differences` object after ignoring only `output_dir` and
`seed`.

## Launch templates

`launch` accepts one arm and one seed. It infers the contracted GPU, requires an
idle RTX 2080 Ti, creates a run directory and immutable receipt exactly once,
then starts training. It refuses resume, overwrite, or adoption of any existing
destination.

```bash
# GPU 0
../my_idea_grpo/.venv/bin/python scripts/run_small_tsp_matched40_pilot.py \
  launch --arm dual_scalar --seed 101 --acknowledge-development-only

# GPU 1
../my_idea_grpo/.venv/bin/python scripts/run_small_tsp_matched40_pilot.py \
  launch --arm dual_scalar --seed 202 --acknowledge-development-only

# GPU 2
../my_idea_grpo/.venv/bin/python scripts/run_small_tsp_matched40_pilot.py \
  launch --arm dual_scalar --seed 303 --acknowledge-development-only
```

Replace `dual_scalar` with `canonical_no_controller` or `standard_grpo` for the
other matched arms. The already completed canonical runs should not be relaunched.
Each receipt records the exact environment, resolved CLI, artifact hashes,
runtime, config diff, and launcher hash.

## Exact deployment-K analysis

Supply the raw B=9 holdout-evaluation JSONL for each policy/seed. Labels are
arbitrary but must be unique. `--run` is optional and adds 40-step trajectory
checks. The output directory is create-once.

```bash
../my_idea_grpo/.venv/bin/python scripts/analyze_small_tsp_matched40_pilot.py \
  --details sft=PATH/TO/SFT/run_details.jsonl \
  --details canonical_s101=PATH/TO/CANONICAL_101/run_details.jsonl \
  --details canonical_off_s101=PATH/TO/CANONICAL_OFF_101/run_details.jsonl \
  --details dual_s101=PATH/TO/DUAL_101/run_details.jsonl \
  --details grpo_s101=PATH/TO/GRPO_101/run_details.jsonl \
  --run canonical_s101=experiments/additional_evidence/small_tsp_canonical_v1/canonical_b9k8_seed101_steps40_v1 \
  --run dual_s101=experiments/additional_evidence/small_tsp_canonical_v1/matched40_v1_runs/dual_scalar_b9_seed101_steps40_v1 \
  --reference sft \
  --output-dir results/additional_evidence/small_tsp_matched40_v1_complete
```

The primary estimator enumerates and uniformly averages all nine leave-one-out
size-eight subsets for each of the 31 instances. This applies to pass/failure,
all success thresholds, and best-feasible quality/coverage. Best-of-nine is
written only as a secondary diagnostic. Paired differences are point estimates;
this pilot does not support confidence, significance, or publication claims.

## Known method-level nonmatches

The standard TRL arm uses group reward scaling, while constrained arms force no
reward scaling; the current standard entrypoint cannot match that setting. It
also uses three separately normalized reward heads rather than the constrained
reward manager. The scalarized dual arm controls component violations, whereas
canonical CLASP controls failure@8. Finally, the controller-off arm fixes lambda
at zero; it is not a validation-selected fixed-lambda baseline. These differences
are declared in the contract and emitted by preflight.
