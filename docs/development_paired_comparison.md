# Development-only paired comparisons

`scripts/compare_development.py` compares exactly two completed current-schema CLASP
evaluation artifacts. It is deliberately unable to issue confirmation or SOTA evidence.
Use it for matched development checks such as SFT versus CLASP, dense versus exact credit,
or raw generation versus deterministic repair.

The comparator fails closed unless both sides have:

- stable, content-hashed JSONL and companion snapshots;
- valid record and companion self-hashes;
- `evaluation_purpose=development` provenance and no confirmation binding;
- identical protocol fingerprint, strict verifier, candidate budget, and paired key set;
- exact pairing on `(task, semantic_instance_sha256, sampling_seed)`;
- identical source-row locator, instance, transformed row, and reference identities; and
- internally consistent pass@K, candidate feasibility, and conditional-gap fields.

For a native evaluator JSONL, its companion is the completed `run_config__*.json`. For a
repair or strict-rescore JSONL, its companion is the generated `*__summary.json`. Standard
names are inferred, but explicit companion paths are preferable in experiment scripts.

## Native versus native

```bash
PYTHONPATH=src python scripts/compare_development.py \
  --left /path/to/baseline/run_details__baseline.jsonl \
  --left-companion /path/to/baseline/run_config__baseline.json \
  --left-label sft \
  --right /path/to/challenger/run_details__challenger.jsonl \
  --right-companion /path/to/challenger/run_config__challenger.json \
  --right-label clasp_dense_step40 \
  --output /path/to/create-once/sft_vs_clasp_dense_development.json
```

## Native raw versus its repair lane

```bash
PYTHONPATH=src python scripts/compare_development.py \
  --left /path/to/raw/run_details__raw.jsonl \
  --left-companion /path/to/raw/run_config__raw.json \
  --left-label raw_n8 \
  --right /path/to/repair/clasp_repair.jsonl \
  --right-companion /path/to/repair/clasp_repair__summary.json \
  --right-label deterministic_repair_n8 \
  --output /path/to/create-once/raw_vs_repair_development.json
```

When a native and repair artifact claim the same source experiment fingerprint, the repair
must embed every native record verbatim. Comparing a repair from another model remains
possible, but the receipt marks verbatim linkage as not applicable.

## Reported quantities

All deltas are `right - left`.

- Pass@K: paired instance rate, exact two-sided McNemar test, and 20,000-replicate paired
  percentile bootstrap interval.
- Candidate feasibility: average paired instance-level candidate rate, an exact sign test,
  and paired instance bootstrap interval. Candidates are not treated as independent samples.
- Conditional gap: each side's descriptive feasible-set mean plus the paired gap difference
  on jointly feasible instances, exact sign test, and paired bootstrap interval. Negative gap
  delta favors the right system.

The bootstrap contract is fixed at 20,000 replicates with base seed `20260814`. Per-metric
PCG64 seeds are deterministically derived from the task/scope and metric name. The JSON
receipt binds exact input bytes, comparator bytes, contract, results, and limitations with
`receipt_sha256`; output is create-once and will not overwrite an existing receipt.

Per-task results should be reported. The pooled block is explicitly descriptive, applies no
multiplicity correction, and does not represent training-seed uncertainty.
