"""Phase III, Step 8: Score & Filter Candidates by Uncertainty.

We combine three signals per candidate:
  - retrieval_score (cosine similarity): higher = better
  - knowledge_uncertainty (Phase I, Step 3): higher = riskier
  - source_weight (Phase II, Step 6 intent): higher = more relevant source

Composite score:
    final = retrieval_score * source_weight * (1 - alpha * knowledge_uncertainty)

Then we keep only candidates whose `final` is in the top
`uncertainty_filter_quantile` of their source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from ._base import Hit


@dataclass
class Candidate:
    hit: Hit
    final_score: float
    components: dict = field(default_factory=dict)


def score_and_filter(
    hits_by_source: Dict[str, List[Hit]],
    source_weights: Dict[str, float],
    *,
    knowledge_uncertainty_alpha: float = 0.5,
    keep_quantile: float = 0.7,
) -> List[Candidate]:
    out: List[Candidate] = []
    for source, hits in hits_by_source.items():
        sw = float(source_weights.get(source, 0.0))
        if not hits or sw <= 0:
            continue
        scored: List[Candidate] = []
        for h in hits:
            ku = float(h.metadata.get("knowledge_uncertainty") or 0.0)
            final = h.score * sw * max(0.0, 1.0 - knowledge_uncertainty_alpha * ku)
            scored.append(
                Candidate(
                    hit=h,
                    final_score=final,
                    components={
                        "retrieval_score": h.score,
                        "source_weight": sw,
                        "knowledge_uncertainty": ku,
                    },
                )
            )
        # Filter to top quantile per-source.
        if len(scored) > 1:
            thr = float(np.quantile([c.final_score for c in scored], 1.0 - keep_quantile))
            scored = [c for c in scored if c.final_score >= thr]
        out.extend(scored)
    out.sort(key=lambda c: -c.final_score)
    return out


def merge_step_candidates(per_step: Sequence[List[Candidate]], fused_top_k: int) -> List[Candidate]:
    """Merge candidates across all implementation steps, deduping by item identity."""
    seen = {}
    for cands in per_step:
        for c in cands:
            key = id(c.hit.item)
            if key not in seen or c.final_score > seen[key].final_score:
                seen[key] = c
    merged = sorted(seen.values(), key=lambda c: -c.final_score)
    return merged[:fused_top_k]
