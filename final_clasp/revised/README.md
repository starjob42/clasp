# Revised paper workspace

This directory contains the editable manuscript, figures, checklist, and the
machine-readable result manifest for the revised CLASP paper.

## Source of truth

[`paper_results_source_of_truth.json`](paper_results_source_of_truth.json) is
the authoritative registry for reported runs. It records the distinction
between the canonical matched baseline and the decoder-tuned development
validation slice. Pending or partial jobs must not be promoted to final paper
results.

## Code provenance

The executable implementation is not duplicated in this directory. It is
tracked in [`src/clasp/`](../../src/clasp/), with launchers in
[`scripts/`](../../scripts/) and tests in [`tests/`](../../tests/). The manifest
records the Git source snapshot that was frozen for the current paper work.
Keeping one implementation avoids the paper silently drifting from the code
that produced the receipts.

## Legacy material

The original ZIP and its figures are retained for auditability. Their raw
tables and plotting inputs were not included in the archive, so a legacy plot
is not considered reproducible merely because its PDF is present. New tables
and figures must be generated from validated receipts and clearly labeled
when they use development validation data.
