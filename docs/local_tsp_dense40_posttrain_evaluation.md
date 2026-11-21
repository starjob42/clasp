# Contracted TSP dense40 post-training evaluation

`scripts/run_local_tsp_dense40_evaluation.py` is the only supported evaluator for the
create-once local trajectory
`outputs/tsp_k8_development_20260814/contracted_clasp_dense40_seed42_v1`.
It is isolated from the training launcher and never writes into, resumes, or selects a
checkpoint inside that run.

## Evidence classification

Every lane contract, comparison, and final receipt says **DEVELOPMENT-ONLY** and
`publication_eligible=false`. This is necessary because the trajectory starts from
Qwen2.5-7B-Instruct, uses local PEFT 0.19.1 on one RTX A6000, covers one TSP training seed,
and evaluates a historical development file rather than the frozen independent benchmark.
These results can decide whether the corrected algorithm merits publication-scale training;
they cannot establish a SOTA claim.

## Fail-closed prerequisites

The launcher refuses before staging data or reserving an output path unless all of the
following hold:

- the immutable `local_development_run_contract.json` is non-writable, self-hashed, and is
  the exact dense40/seed42/CLASP-dense contract;
- the contracted launcher, complete copied SFT specialist, and 3,200-row RL train file still
  match their fixed SHA-256 values;
- `stage_status.json` says `completed=true` and `global_step=max_steps=40`;
- `checkpoints/checkpoint-40` has adapter weights, trainer state, optimizer, scheduler,
  single-GPU RNG state, and the TSP primal-dual/CLASP-BoN controller state;
- the root model, root trainer state, log history, terminal checkpoint, and hyperparameters
  all agree; and
- the evaluator/repair/comparator code and the exact local dependency versions are hashed.

With no wait option, an unfinished run is refused immediately. `--wait-timeout-seconds N`
polls read-only for at most `N` seconds; malformed or hash-drifted state always fails
immediately.

## Evaluation data and matched protocol

The only allowed data source is:

```text
/home/lamsade/habgaryan/phd_project/hybrid_gnn_llm/data_rl/tsp/eval/test.json
SHA-256 f169771f0ccfe0e74cf247067edcea8f89a357ab4823352e3ff55201cf8b9597
```

At execution time its exact bytes are copied, create-once, to
`data/development_historical_tsp_eval_20260814/tsp/eval/test.json`. The staging root must
contain no other file. The source has 300 rows; this lane fixes the deterministic first
interval `[0,200)`, whose canonical slice hash is
`c9beea11e14f7b31897c515e3fb561fa93e470e258ea62a69478f3e06603437c`.

SFT and checkpoint-40 use identical explicit settings:

| Field | Value |
|---|---:|
| tokenizer | copied audited SFT tokenizer |
| prompt | Alpaca, forced no-reason |
| maximum prompt/completion | 29,768 / 3,000 tokens |
| candidates | K=8, generation batch 8 |
| decode seed | 271828, deterministic per-example derivation |
| sampling | temperature 0.7, top-p 0.9, top-k 50 |
| rows | TSP eval `[0,200)` |

Both native evaluator artifacts must be complete, have 200 records, bind only the audited
eval file, and expose the same protocol fingerprint. Resume and mutable best-run updates are
disabled.

## Pipeline and receipts

After the two raw evaluations, the launcher applies the versioned deterministic repair lane
to each artifact. It then invokes `scripts/compare_development.py` four times:

1. raw SFT versus raw dense40;
2. repaired SFT versus repaired dense40;
3. SFT raw versus SFT repair; and
4. dense40 raw versus dense40 repair.

The output root is fixed and create-once:

```text
experiments/sota_20260814/local_development/
  tsp_dense40_posttrain_k8_seed271828_v1/
```

`evaluation_lane_contract.json` self-hashes the trusted inputs, code, environment, GPU, and
planned commands before the first evaluation starts. On success, `evaluation_receipt.json`
rehashes the training checkpoint, SFT/data/code inputs and every lane artifact, records all
eight commands and logs, binds the shared protocol fingerprint, and self-hashes with
`receipt_sha256`. If any command fails or a trusted input changes during the multi-hour run,
the partial directory is retained and cannot be reused.

## Sealed result

The 200-instance K=8 run completed on 2026-08-14.  Its create-once receipt hash is
`a354f6cf4b9d185d72451babede41992bf96cb7580df24462911893d33715e9d`.
Both raw policies achieved pass@8 = 1.0.  The SFT saved policy had candidate
feasibility 0.89625 and conditional best gap 0.00882610; dense40 had 0.89875 and
0.00917916.  Thus dense40 changed candidate feasibility by +0.0025 (paired 95%
CI [-0.0075, 0.0125]) but worsened gap by +0.00035306 (95% CI
[-0.00006381, 0.00079443]); neither contrast was significant.

Deterministic repair raised candidate feasibility to 1.0 for both systems and
reduced their gaps to 0.00610852 (SFT) and 0.00605487 (dense40).  The repaired
dense-minus-SFT gap was -0.00005365 (95% CI [-0.00032047, 0.00020572]), also a
tie.  Repair itself was decisive: it reduced SFT gap by 0.00271758 and dense40
gap by 0.00312429, with all 127 and 131 non-tied paired instances improving,
respectively.

Post-run checkpoint inspection found that PEFT had trained 84 Q/K/V bias tensors
but persisted none of them under `bias=lora_only`.  The internal terminal
validation therefore describes the live in-memory policy, while this sealed
reload evaluation describes the saved adapter with base biases restored.  The
artifact remains useful saved-policy development evidence but is not a faithful
terminal checkpoint and cannot authorize algorithmic advancement or publication.

## Verify or execute

Read-only verification of the now-completed lane:

```bash
../my_idea_grpo/.venv/bin/python \
  scripts/run_local_tsp_dense40_evaluation.py
```

Execute after reviewing the completed preflight output:

```bash
../my_idea_grpo/.venv/bin/python \
  scripts/run_local_tsp_dense40_evaluation.py \
  --execute --acknowledge-development-only
```

No GPU evaluation is launched merely by adding or testing this orchestrator.
