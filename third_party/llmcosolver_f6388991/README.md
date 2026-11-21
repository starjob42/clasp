# LLMCoSolver reward oracle snapshot

`rewards.py` and `utils.py` are newline-normalized copies of the official
LLMCoSolver repository at commit
`f6388991d243c0aa91d9ee7e2fa18ed1c9e72477`. The upstream CRLF byte hashes are
recorded in the frozen semantic contract; the normalized files are checked for
content equality after newline normalization.

The files are retained only as a differential-test oracle. Production CLASP
code imports `clasp.baselines.llmcosolver_f638_rewards`, while publication
metrics come from the strict CLASP evaluator. The included `LICENSE` is the
upstream MIT license and its notice applies to these copied files.
