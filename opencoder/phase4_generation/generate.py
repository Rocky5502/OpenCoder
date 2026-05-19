"""Phase IV, Step 10: Uncertainty-Guided Code Generation.

Receives: user query, fused evidence, and the uncertainty trace from
Phase II. Samples N candidate completions; computes the three
uncertainty signals over those samples; selects the best by
self-consistency cluster.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from ..llm.client import LLMClient
from ..uncertainty import (
    UncertaintyTrace,
    aggregate_uncertainty,
    normalize_code,
    self_consistency_score,
    semantic_variance_score,
    token_entropy_score,
)


_SYSTEM = (
    "You are a senior Python engineer. Use the provided evidence to implement the user's "
    "task. Return ONLY the Python code block — no commentary, no markdown outside one code "
    "fence. The code must be self-contained and import what it needs."
)


@dataclass
class GenerationResult:
    code: str
    samples: List[str]
    sample_logprobs: List[List[float]]
    trace: UncertaintyTrace
    components: dict = field(default_factory=dict)


_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def _pick_consensus(samples: List[str]) -> str:
    from collections import Counter
    norm = [normalize_code(s) for s in samples]
    counts = Counter(norm)
    winner_norm, _ = counts.most_common(1)[0]
    for s in samples:
        if normalize_code(s) == winner_norm:
            return s
    return samples[0]


def generate_target_code(
    query: str,
    fused_evidence: str,
    query_uncertainty_summary: Dict[str, float],
    llm: LLMClient,
    encoder,
    n_samples: int = 5,
    uncertainty_weights: Dict[str, float] | None = None,
) -> GenerationResult:
    prompt = (
        f"# Task\n{query}\n\n"
        f"# Query Uncertainty Trace\n{query_uncertainty_summary}\n\n"
        f"# Retrieved Evidence\n{fused_evidence}\n\n"
        f"# Instruction\nImplement the task. Treat high-uncertainty evidence with caution; "
        f"prefer evidence with lower knowledge_uncertainty when there is conflict."
    )
    resps = llm.complete(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
        n=n_samples,
        temperature=0.7,
        return_logprobs=True,
    )
    codes = [_extract_code(r.text) for r in resps]
    logprobs = [r.logprobs or [] for r in resps]

    # Per-sample token entropies, averaged.
    te = sum(token_entropy_score(lp) for lp in logprobs) / max(1, len(logprobs))
    sc = self_consistency_score(codes)
    sv = semantic_variance_score(codes, encoder)
    trace = aggregate_uncertainty(te, sc, sv, weights=uncertainty_weights)

    chosen = _pick_consensus(codes)
    return GenerationResult(
        code=chosen,
        samples=codes,
        sample_logprobs=logprobs,
        trace=trace,
        components={"token_entropy": te, "self_consistency": sc, "semantic_variance": sv},
    )
