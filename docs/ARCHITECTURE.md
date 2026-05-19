# OpenCoder — Architecture Notes

This document explains how each step in the 13-step framework is realized in code, and which knobs map to which research-question ablation.

## Phase I — Repository Knowledge & Uncertainty Profiling

| Step | Module | What it does |
|---|---|---|
| 1. Extract Knowledge | `phase1_repo_knowledge/extract.py` | Walks the repo, parses each `.py` file with `ast`, emits one `RepoFunction` per function/method/class. |
| 2. Generate Descriptions | `phase1_repo_knowledge/describe.py` | LLM produces a 3–5 sentence NL description of each item. Logprobs are kept on the record. |
| 3. Profile Uncertainty | `phase1_repo_knowledge/profile.py` | Per-item knowledge uncertainty = `token_entropy_score(describe_logprobs)`. |

Output: `RepoFunction` list with `description` + `uncertainty` populated. Feeds the API & Similar-Code retrievers in Phase III.

## Phase II — Query Uncertainty Decomposition

| Step | Module | What it does |
|---|---|---|
| 4. Generate Implementation Steps | `phase2_query/decompose.py` | LLM emits 3–6 ordered steps as JSON. |
| 5. Estimate Step-Level Uncertainty | `phase2_query/step_uncertainty.py` | Per-step uncertainty from rationale logprobs. |
| 6. Predict Retrieval Intent | `phase2_query/retrieval_intent.py` | For each step, predicts source weights and per-source search queries. |

## Phase III — Uncertainty-Aware Multi-Source Retrieval

Three retrievers (`api`, `context`, `similar_code`) share a small `VectorIndex` (`phase3_retrieval/_base.py`). Step 8 combines:

```
final = retrieval_score · source_weight · (1 − α · knowledge_uncertainty)
```

then keeps the top `keep_quantile` per source. Step 9 fuses by source for prompt injection.

`uncertainty_aware=False` in `PipelineConfig` sets `α=0` and `keep_quantile=1.0`, giving a pure retrieval baseline — the control for RQ2.

## Phase IV — Uncertainty-Guided Code Generation

`phase4_generation/generate.py` samples N completions at temperature 0.7, computes:
- token-logprob entropy (per-sample, averaged)
- self-consistency (cluster size of normalized samples)
- semantic variance (1 − mean pairwise cosine of embedded samples)

aggregates them into the `UncertaintyTrace`, and returns the consensus sample.

## Phase V — Verification & Uncertainty Mitigation

| Step | Module | What it does |
|---|---|---|
| 11. Static Checks | `phase5_verify/static_checks.py` | `ast.parse` + undefined-name advisory. |
| 12. Test & Validate | `phase5_verify/test_validate.py` | Runs example tests in a subprocess with a wall-clock timeout. |
| 13. Repair / Refine | `phase5_verify/repair.py` | Repair-pass LLM call with diagnostics. Up to `max_repair_rounds` (default 2). |

The feedback arrow in the diagram is implemented in `pipeline.Pipeline.run()` as the loop around static + test + repair.

## RQ → Knob mapping

| Research Question | Knob | Where |
|---|---|---|
| RQ1: per-source effect on uncertainty | `enable_sources` | `PipelineConfig.enable_sources` — toggled in `scripts/ablation_rq1.py` |
| RQ2: uncertainty-aware vs. baseline | `uncertainty_aware` | `PipelineConfig.uncertainty_aware` — toggled in `scripts/ablation_rq2.py` |

## Backends

- `OPENCODER_LLM_BACKEND=openai` → `OPENAI_API_KEY`, uses `gpt-4o-mini` by default.
- `OPENCODER_LLM_BACKEND=lovable` → `LOVABLE_API_KEY`, OpenAI-compatible.

Both expose token logprobs; the uncertainty pipeline is backend-agnostic.

## Embeddings

`opencoder/embeddings/encoder.py` tries `transformers` + UniXcoder; if unavailable, falls back to a deterministic hashing encoder so the pipeline still runs (useful for CI / smoke tests).
