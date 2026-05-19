# How OpenCoder relates to prior work

OpenCoder is a clean-room implementation written from the paper proposal and the 5-phase / 13-step framework diagram. No source code from any other project was copied or vendored.

## Reference papers

- *Beyond "What to Retrieve": Uncertainty in Retrieval-Augmented Code Generation* (this work)
- *What to Retrieve for Effective Retrieval-Augmented Code Generation?* — referenced for motivation and empirical comparison. We re-derive the API-description-driven retrieval idea independently in `phase1_repo_knowledge/describe.py`.

## What's original to OpenCoder

- Per-item **knowledge uncertainty** scoring (Phase I, Step 3) — derived from the description-generation logprobs.
- **Query Uncertainty Decomposition** (Phase II, Steps 4–6) — step-level uncertainty + retrieval-intent prediction.
- **Uncertainty-aware retrieval scoring** (Phase III, Step 8) — composite of similarity, source weight, and knowledge uncertainty with quantile filtering.
- **Three-signal uncertainty trace** (Phase IV) — token entropy + self-consistency + semantic variance, aggregated.
- **RQ-aligned ablation harness** (`scripts/ablation_rq1.py`, `scripts/ablation_rq2.py`).

## Datasets

Public datasets reused as-is:
- ExecRepoBench (https://github.com/exec-repo-bench)
- CoderEval
- RepoExec

Dataset files are not redistributed in this repository; loaders expect them at user-provided paths.
