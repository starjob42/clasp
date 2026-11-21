# LLMCoSolver f638 public-data addendum (schema v2)

This addendum supersedes only the data-availability and split-independence
claims in the immutable v1 LLMCoSolver contract. The pinned reward/loss port,
the evolving-bias finding, and both disabled reference-policy variants are
unchanged. This remains a non-executable compatibility artifact, not an
FOARL result or a matched baseline.

## Bytes and row semantics now frozen

The official README links public SFT and RL Google Drive folders. Ten RL JSON
files were acquired directly from the exact publication-task item IDs using a
create-once workflow. Their 266,602,952 raw bytes are locally SHA256-bound,
mode `0444`, under mode-`0555` directories. Every JSON row was parsed. Counts
are:

| Task | RL train | RL eval | SFT train | SFT eval |
| --- | ---: | ---: | ---: | ---: |
| CVRP | 2,000 | 300 | 500,000 | 300 |
| JSSP | 3,200 | 300 | 500,000 | 300 |
| MIS | 3,200 | 300 | 500,000 | 300 |
| OP | 5,000 | 300 | 500,000 | 300 |
| TSP | 3,200 | 300 | 500,000 | 300 |

The neighboring pinned official-repository checkout already contained all
five SFT train/eval JSON pairs (12,944,914,247 JSON bytes) and four of
five `instances.pkl` files. The five train files alone are 12,936,158,629
bytes; 12,946,455,971 total SFT bytes are locally present. Their filenames and
lengths match direct Drive HEAD metadata, and all
2,501,500 JSON rows were stable-hashed and parsed in place. No Drive checksum
or local acquisition receipt exists, so these are **probable released bytes,
not byte-proven Drive copies**. The 249,598-byte TSP `instances.pkl` is absent
locally. Pickles were raw-hashed only, never deserialized; the pinned SFT code
also sets `instances=None`.

The Drive data has no separately observed license statement. The repository's
MIT license covers the source snapshot, but this contract does not infer that
it authorizes redistribution of externally hosted data.

## The author evaluation is not globally held out

Schema v2 uses one identity for all four stage/split combinations: SHA256 over
the task name plus the rendered `input` after Unicode NFC normalization and
whitespace canonicalization. It separately counts full-row duplicates and
whether shared instances carry identical or different `output` references.
This fixes v1's TSP-only schema asymmetry, where eval rows have an extra
`instance` field.

Exact unique-instance overlap is:

| Task | RL train ↔ RL eval | SFT train ↔ SFT eval | SFT train ↔ RL eval | author train union ↔ RL eval |
| --- | ---: | ---: | ---: | ---: |
| CVRP | 0 | 300 | 299 | **299 / 300** |
| JSSP | 0 | 0 | 0 | 0 / 300 |
| MIS | **300** | 0 | 0 | **300 / 300** |
| OP | 0 | **299** | **299** | **299 / 300** |
| TSP | 0 | 0 | 0 | 0 / 300 |

MIS's 300 overlaps are exact duplicate RL rows with identical references.
CVRP and OP are different: the shared problem instances usually carry
different reference outputs. For SFT-train versus RL-eval, 285 of 299 CVRP
instances and 258 of 299 OP instances have disjoint reference sets. Thus
deduplicating only full rows would miss the contamination.

RL and SFT eval contain the same rendered instances for all 300 examples on
JSSP, MIS, OP, and TSP, and for 299 on CVRP. TSP's two eval JSON files are
byte-identical. These linked author splits cannot be called globally held-out
or leakage-free. They are suitable only for author-code context reproduction.
A claim-bearing comparison needs a preregistered fresh, disjoint IID split;
that comparison is deliberately no longer exact-paper-data reproduction.

## What remains unmatched

Public row counts make the default one-epoch horizon interpretable but do not
prove the paper's actual hardware or checkpoint. On a single process, pinned
TRL repeats each prompt eight times, uses batch 8 and gradient accumulation 8,
and therefore implies 250 CVRP, 400 JSSP, 400 MIS, 625 OP, and 400 TSP RL
optimizer steps. It generates eight completions per train row. World size,
actual completion lengths, token throughput, and an executed run receipt are
not published, so matched compute/token budgets remain unresolved.

Each SFT file has 500,000 rows. Under the single-process source defaults
(batch 4, accumulation 4, one epoch), this implies 31,250 optimizer steps and
explains the example `checkpoint-31250` path. It does not identify or hash any
released per-task SFT checkpoint, pin a base-model revision, or prove the
paper's actual distributed configuration.

Consequently both reference variants remain `HOLD`: (A) official operational
`lora_only` with an evolving bias-contaminated KL reference, for context only;
and (B) the corrected `bias=all` persistence plus frozen step-zero SFT
reference, which is publication-scientific but not a byte-identical FOARL
reproduction. Current CLASP `METHOD=grpo` remains three-head `dr_grpo`,
non-FOARL, and unmatched.

The authoritative data receipt is
`experiments/sota_20260814/llmcosolver_f638_public_data_manifest_v2.json`.
It preserves and explicitly supersedes the first create-once data receipt;
the first receipt's raw hashes, schemas, row identities, and reference-field
evidence remain valid, while all v1 overlap claims are deprecated.
