# Matched-budget CLASP vs FOARL (one seed, inspected development cohort)

This is the current, reproducible state of the CLASP-versus-FOARL comparison on the
five CO tasks. Every number below is recomputed from the immutable artifacts in
[`evidence/`](evidence) by

```bash
python experiments/grpo_full_budget_20260918/verify_matched_table.py --markdown
```

which fails closed if a file is missing, the recorded protocol differs, or a value
drifts from the expectation recorded at publication time.

## Table

| Task | CLASP step | CLASP BoN feas / gap | FOARL step | FOARL BoN feas / gap | gap(FOARL-CLASP) pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| CVRP | 500 | 95.0% / 3.881% | 500 | 95.0% / 3.820% | -0.062 |
| JSSP | 800 | 100.0% / 6.440% | 650 | 100.0% / 6.642% | +0.202 |
| TSP | 800 | 100.0% / 0.629% | 650 | 100.0% / 0.730% | +0.102 |
| MIS | 800 | 94.0% / 2.036% | 400 | 90.0% / 2.456% | +0.421 |
| OP | 1250 | 84.0% / 1.259% | — | — | n/a |

`gap` is the mean best-of-8 conditional gap over the instances that have at least
one feasible candidate; feasibility is instance-level best-of-8. A positive
`gap(FOARL-CLASP)` means CLASP has the smaller gap. Gap is never interpreted
without the feasibility beside it.

### Step-matched bracket for JSSP and TSP

CLASP keeps checkpoints only at 100-step boundaries while the FOARL arm stopped
at update 650, so both CLASP neighbours of that update were also measured:

| Task | CLASP 600 gap | CLASP 700 gap | FOARL 650 gap | Both CLASP endpoints better |
| --- | ---: | ---: | ---: | :---: |
| JSSP | 6.396% | 6.354% | 6.642% | yes |
| TSP | 0.637% | 0.633% | 0.730% | yes |

Both tasks are fully feasible at every candidate count on all three CLASP
checkpoints and on FOARL at 650, so the bracket is a gap-only comparison. Since
the FOARL gap lies above *both* CLASP endpoints, the ordering at update 650 does
not depend on assuming that CLASP improves monotonically between 600 and 700.
CLASP's own gap is not monotone here (JSSP and TSP are both slightly better at
700 than at the declared horizon 800), which is why the bracket is reported as
endpoints rather than interpolated.

## Shared evaluation protocol

Both sides were evaluated with the same harness on the same 100-instance
development slice:

| Field | Value |
| --- | --- |
| cohort | `eval[0:100]` of the frozen DEV100 files under `clasp_foarl_v3_reproduction_20261018_v1/data_eval` |
| candidates | 8 raw, one `generate` call, no repair or reranking |
| decoding | fp16, T=0.7, top-p=0.9, top-k=50, repetition penalty 1.0, 3,000 new tokens |
| seed | 271828, deterministic per-example SHA-256 derivation |
| reference | certified optimum (MIS), declared reference otherwise |

CLASP rows use the `jiang_f638` no-reason prompt; FOARL rows use the same
`jiang_f638` no-reason prompt. The earlier `alpaca` FOARL JSSP/TSP evaluations
were superseded by `*_jiang_run_summary.json` in this directory.

## Step accounting

The study uses nine responses per prompt and four prompts per update (36
completions per update). The released FOARL recipe uses eight responses per
prompt, so equal *prompt exposure* corresponds to the declared horizons
`cvrp 500 / jssp 800 / mis 800 / tsp 800 / op 1250`. Equal tokens or GPU time is
not claimed (see [`../../research/grpo_budget_fairness_20260918/FOARL_BUDGET_AUDIT.md`](../../research/grpo_budget_fairness_20260918/FOARL_BUDGET_AUDIT.md)).

## Availability of the FOARL side

Only **CVRP** is an exactly matched pair (500/500). The FOARL campaign
(`grpo_full_budget_20260918`) stopped before its declared horizons and its
submission window (60 days from 2026-09-18) closed before the remaining chunks
could be admitted. Its artifact tree was also trimmed on 2026-11-13/14:

| Task | Declared horizon | FOARL reached | Surviving policy | Table row |
| --- | ---: | ---: | --- | --- |
| CVRP | 500 | 500 | run-root adapter, `global_step=500` | step 500 |
| JSSP | 800 | 650 | `checkpoints/checkpoint-650` (adapter only, no optimizer state) | step 650 |
| TSP | 800 | 650 | `checkpoints/checkpoint-650` (full resume state) | step 650 |
| MIS | 800 | 650 | none (checkpoint dirs empty) | step 400 report only |
| OP | 1250 | 550 | none (checkpoint dirs empty) | unavailable |

Consequences:

- The JSSP/TSP CLASP rows are at the declared horizon while FOARL is 150 updates
  short. The step-matched bracket above removes the need for a monotonicity
  assumption: FOARL at 650 is worse than CLASP at both 600 and 700.
- The MIS FOARL row is the campaign's signed step-400 report
  (`mis_step400_campaign_report.json`): best-of-8 feasibility 90.0%, mean signed
  reference gap 2.456% over its 90 feasible instances. It is **not** a
  declared-horizon number.
- OP FOARL has no surviving evaluation at any step, so no row is reported rather
  than substituting an unmatched number.
- The FOARL MIS/OP adapters and their chunk-completion receipts were pruned. A
  declared-horizon FOARL MIS/OP row requires a new, separately versioned
  continuation run from the shared step-31,250 SFT initializers; the original
  campaign cannot be resumed. It is also noted that the OP initializer's
  `trainer_state.json` was pruned with the same cleanup, so its initialization
  receipt must be re-attested from `checkpoint_fidelity.json` (adapter bytes)
  rather than from the trainer state.

## Paper reference (different cohort — indicative only)

The FOARL paper's reported SFT+RL+BoN targets are
TSP 100/1.070, CVRP 100/4.530, OP 100/1.850, MIS 94/1.040, JSSP 100/8.200
(best-of-8 feasibility % / gap %). They were measured on a different, unseen
cohort and are not a matched comparison to this table.

## Provenance

Each evidence file is a verbatim copy of a Jean Zay artifact. `sha256` is the
file hash; `artifact` is the `summary_sha256` (run summaries) or `receipt_sha256`
(signed campaign reports) embedded in the artifact.

| Evidence file | sha256 | artifact |
| --- | --- | --- |
| `clasp/cvrp_v11_step500_run_summary.json` | `947604ac8e7dc0de9c72d25c0b1564d374da1ad064261f37236261fed237b90e` | `079c2fc4a646729377a0d5ce5b3cc2f0220490d7386a1c547cf8606eac6676b1` |
| `clasp/jssp_v8_step800_run_summary.json` | `8569422f74d34f9b0a977ff38a876f276893755e96aaa92b0dc66f4f3b77e806` | `5d6c711268edc6bb13af9db4648f00935c36374cc6a8ef4b36d9de48aead6e3d` |
| `clasp/tsp_v8_step800_run_summary.json` | `06dd7487c2588ed07b47d4bae4f9a19418ec6dbdab1895e756646a4eb76bdac8` | `e29523208b70cb8c39915316e0368e474c94d6c098da48555b6cecfc71599daf` |
| `clasp/jssp_v8_step600_run_summary.json` | `eb7999b9186ad678612ba813cd1f664b9d4d03259dce64622c6c5c5f8de9f3eb` | `82aaf443b9aefe1c5a213387a3604d46c9309ab5fef23090b94f6411f1bf4fd1` |
| `clasp/jssp_v8_step700_run_summary.json` | `ff2fb3bff98d74e6390dab9ac0298d0e6c50ce6fd4113125c5facee48a6d458b` | `f0a6d6149af52952bff25ce2f85cabf4781b1e2a2eec7511037b67cc7e319be6` |
| `clasp/tsp_v8_step600_run_summary.json` | `6330f74812a7082af958aef5f52bb5b46a6216663dabc7fbfe121b38d94efca2` | `92b28e84a49467d328d15388268627c060ff81d9b5b15be21cbf774faf9bce28` |
| `clasp/tsp_v8_step700_run_summary.json` | `57ea9aee41101224d27db67159207491aef8a7026dcfccfc0db5518a4d9d3f7e` | `f41badddcabfa9ccde6e993a61c88c18cd936db587d7c6b339d8692f8fa0a2fe` |
| `clasp/mis_v15_step800_run_summary.json` | `1d17c0dbc26eb3b32735fff3af0f10fe279590b91868de4b0c6253574f248be5` | `67b532ad393fb7a91d96c01baecd538be9cf5edbcc6ffa32450ad5f7f6b02183` |
| `clasp/op_v14_step1250_run_summary.json` | `0d9ef804acb3a976b0745b3b214c7cb46e7820a2d76825f6aec385f6c1378678` | `fb8672658d50b4088438826e0a287a1edd207e97aa8d32e9b65e9d0f0293425c` |
| `foarl/cvrp_step500_run_summary.json` | `3b8e978aaee6e5a979667e70d77d64c744631afc5f91193b3ca6c4e32c73ab3e` | `6e99755e58420232e4e30b3a48d41ce3ffb0509386f7f642ae4ad4b05b7f6b93` |
| `foarl/jssp_step650_jiang_run_summary.json` | `5885de4fd962948f9edba59c8516fc91b46bb97509692a076d9a8983d8e0f782` | `104c35a1caab4c6633d8dad384b233ef0725b94aae963541e24fdfefc120f19c` |
| `foarl/tsp_step650_jiang_run_summary.json` | `a052cfa694f55de265aafdd4a9b071e0ec45717a561f88f549e5a95140b1c606` | `7001cc575c0e5e665494c880a3b3b9f59eb1db36e3a84ecbb998901b3b95e2da` |
| `foarl/mis_step100_campaign_report.json` | `251da1d03bf0dd724353b11844634b8c3962fc0980bef8c500b66fe75706c44b` | `5e46bc8b0a5ac5168e715bd1fcb5ba6db00b0a277f7ae359d816c5e184b921ec` |
| `foarl/mis_step400_campaign_report.json` | `b3fc96b205cbe95a1215038a1ab921e247466c4b25f92cc7b0ba45d60fae0e9d` | `ef79646a7144e0d2fb3c73486a9757a551f0070f7df3fef983e3c059ef10eed8` |

## Reproduction

- Trainer, campaign controller and evaluators: this directory plus
  [`src/clasp/baselines/budget_matched_dev.py`](../../src/clasp/baselines/budget_matched_dev.py).
- Immutable campaign source: [`source_v1/`](source_v1) with `SOURCE_MANIFEST.sha256`
  (also archived as `source_v1.tar.gz`).
- Launch declaration and budget arithmetic: [`launch_declaration_v1.json`](launch_declaration_v1.json).
- This table only: `PYTHONPATH= python experiments/grpo_full_budget_20260918/verify_matched_table.py --markdown`
  (stdlib only, no GPU, no cluster access).

A one-seed result on an inspected development cohort cannot establish a general
win. Confirmation needs a fresh cohort and multiple training seeds.
