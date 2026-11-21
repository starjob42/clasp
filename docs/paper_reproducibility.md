# Revised paper reproducibility

The revised paper is built from the machine-readable manifest at
[`final_clasp/revised/paper_results_source_of_truth.json`](../final_clasp/revised/paper_results_source_of_truth.json).
That manifest is the only authoritative input for new comparison tables and
figures. The archived ZIP under `final_clasp/` is retained as historical
material and is not treated as a reproducible source.

## Clean code path

The code that produced the current CLASP training and evaluation runs is the
package under [`src/clasp/`](../src/clasp/), invoked through the entry points
declared in [`pyproject.toml`](../pyproject.toml). The reward and advantage
implementation that defines the reproduced algorithm is
[`src/clasp/rewards/clasp_math.py`](../src/clasp/rewards/clasp_math.py).
Slurm launchers and evaluation wrappers are under [`scripts/`](../scripts/).

The paper must identify the exact source snapshot and launch configuration for
each reported run. A checkpoint or output directory alone is not sufficient:
the source tree, configuration, data slice, decoding parameters, evaluator
version, and result receipt must all be retained together.

## Validation

From the repository root:

```bash
python scripts/validate_paper_source_of_truth.py
```

The validator deliberately rejects a final MIS result while the registered
stage-two search is pending. It also checks that complete tuned results contain
all 96 instances and that the canonical comparison contains all five tasks.

## Result classes

The manifest keeps two result classes separate:

* **Canonical matched baseline:** `eval[0:100]`, fixed decoding, and the
  historical five-task comparison.
* **Tuned validation:** disjoint `eval[100:196]`, used for decoder selection.

The tuned slice is development data and must not be described as an unbiased
test set. FOARL entries are local F638 reimplementations unless an official
checkpoint and receipt are explicitly documented. Partial jobs are never
eligible for a paper table.

## Git workflow

Changes should be committed in coherent steps:

1. reproducibility metadata and validators;
2. finalized run receipts and generated tables;
3. regenerated figures;
4. manuscript and checklist updates.

Large checkpoints, datasets, logs, and generated inference outputs stay
ignored by Git. They must instead be referenced by immutable paths, hashes, and
receipts in the manifest.
