"""RQ1 ablation: per-source uncertainty correlation.

For each example, run the pipeline 4 times:
  - all sources
  - api only
  - context only
  - similar_code only

Record uncertainty trace + correctness. Then compute Spearman
correlation between (presence of each source) and (aggregate
uncertainty / pass@1). The deliverable is a per-source contribution
table for the paper.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from itertools import combinations
from typing import List

import numpy as np
from scipy.stats import spearmanr

from opencoder.data.loaders import load_dataset
from opencoder.pipeline import Pipeline, PipelineConfig


SOURCES = ("api", "context", "similar_code")


def _run_with(pipe, ex, retrievers, enabled):
    pipe.cfg.enable_sources = tuple(enabled)
    return pipe.run(ex, retrievers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="execrepobench")
    ap.add_argument("--dataset-path", default="data/execrepobench_data.jsonl")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--describe-limit", type=int, default=100)
    ap.add_argument("--out", default="results/rq1.json")
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    cfg.llm_backend = os.environ.get("OPENCODER_LLM_BACKEND", cfg.llm_backend)
    pipe = Pipeline(cfg)

    examples = list(load_dataset(args.dataset, args.dataset_path, limit=args.limit))
    rows: List[dict] = []
    for ex in examples:
        repo_root = ex.repo_root or args.repo_root
        _, retrievers = pipe.index_repo(repo_root, describe_limit=args.describe_limit)
        # Conditions: all, single-source x3, leave-one-out x3, none.
        conditions = [
            ("all", SOURCES),
            *[(f"only_{s}", (s,)) for s in SOURCES],
            *[(f"no_{s}", tuple(x for x in SOURCES if x != s)) for s in SOURCES],
        ]
        for name, enabled in conditions:
            try:
                r = _run_with(pipe, ex, retrievers, enabled)
                rows.append({
                    "example_id": ex.id,
                    "condition": name,
                    "enabled": list(enabled),
                    "u": r.uncertainty_trace,
                    "passed": r.test_report.get("passed"),
                })
                print(f"  {ex.id:<24} {name:<18} u={r.uncertainty_trace['aggregate']:.3f} "
                      f"pass={r.test_report.get('passed')}")
            except Exception as e:
                rows.append({"example_id": ex.id, "condition": name, "error": str(e)})

    # Correlations.
    summary = {}
    for s in SOURCES:
        present = []
        agg_u = []
        passed = []
        for r in rows:
            if "u" not in r:
                continue
            present.append(1 if s in r["enabled"] else 0)
            agg_u.append(r["u"]["aggregate"])
            passed.append(1 if r["passed"] else 0)
        if len(set(present)) > 1:
            rho_u, p_u = spearmanr(present, agg_u)
            rho_p, p_p = spearmanr(present, passed)
            summary[s] = {
                "spearman_uncertainty": {"rho": float(rho_u), "p": float(p_u)},
                "spearman_passed":      {"rho": float(rho_p), "p": float(p_p)},
                "mean_u_when_present":  float(np.mean([u for u, x in zip(agg_u, present) if x])),
                "mean_u_when_absent":   float(np.mean([u for u, x in zip(agg_u, present) if not x])),
            }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2, default=str)
    print(f"\nRQ1 summary:\n{json.dumps(summary, indent=2)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
