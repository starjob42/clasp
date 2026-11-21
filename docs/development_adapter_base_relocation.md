# Development adapter base relocation

`scripts/relocate_development_adapter_base.py` solves one narrow provenance
problem: a development-only converted adapter may name a writable local base,
even when a byte-identical immutable copy of that base exists. The tool creates
a new adapter directory and changes only the semantic
`base_model_name_or_path` field to that immutable copy.

The output is accepted only after all of the following hold:

- the source conversion receipt and its live source adapter are valid;
- the upstream recorded base, live original base, and frozen base have exactly
  the same path-independent byte manifest;
- every non-config adapter payload file is byte-identical;
- independent source and relocated model loads have the same full model state;
- three fixed probes have bitwise-identical logits, full log-probability
  distributions, and argmax values; and
- the output and its self-hashed receipt are made read-only.

Example (the output directory must not already exist):

```bash
PYTHONPATH=src python scripts/relocate_development_adapter_base.py \
  --source-adapter-path /absolute/path/to/converted-adapter \
  --frozen-base-model-path /absolute/path/to/read-only-base \
  --output-dir /absolute/path/to/new-relocated-adapter \
  --device cuda:0 \
  --dtype float32
```

Verification can be repeated without model loading:

```bash
PYTHONPATH=src python scripts/relocate_development_adapter_base.py \
  --verify-only \
  --source-adapter-path /absolute/path/to/converted-adapter \
  --frozen-base-model-path /absolute/path/to/read-only-base \
  --output-dir /absolute/path/to/new-relocated-adapter
```

The receipt is intentionally and permanently marked `development_only=true`,
`publication_eligible=false`, and `sota_eligible=false`. It proves identity of
the deployable saved policy only. It does not recover Q/K/V biases omitted from
the historical SFT serialization and cannot support publication or SOTA
claims.
