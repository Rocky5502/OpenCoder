"""RQ2 ablation: with vs. without uncertainty-aware scoring.

Runs the full pipeline twice per example: uncertainty_aware=True and
False. Compares pass@1, mean uncertainty, and calibration (ECE) of the
uncertainty signal against correctness.
"""
from __future__ import annotations

import argparse
import json
import os

from opencoder.data.loaders import load_dataset
from opencoder.evaluation.metrics import pass_at_k, uncertainty_calibration_ece
from opencoder.pipeline import Pipeline, PipelineConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="execrepobench")
    ap.add_argument("--dataset-path", default="data/execrepobench_data.jsonl")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--describe-limit", type=int, default=100)
    ap.add_argument("--out", default="results/rq2.json")
    args = ap.parse_args()

    base = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    base.llm_backend = os.environ.get("OPENCODER_LLM_BACKEND", base.llm_backend)

    out = {"with": [], "without": []}
    for flag in (True, False):
        cfg = PipelineConfig(**{**base.__dict__, "uncertainty_aware": flag})
        pipe = Pipeline(cfg)
        examples = list(load_dataset(args.dataset, args.dataset_path, limit=args.limit))
        for ex in examples:
            _, retrievers = pipe.index_repo(ex.repo_root or args.repo_root,
                                            describe_limit=args.describe_limit)
            r = pipe.run(ex, retrievers)
            out["with" if flag else "without"].append({
                "id": ex.id,
                "passed": r.test_report.get("passed"),
                "u": r.uncertainty_trace,
            })

    summary = {}
    for k, rows in out.items():
        passed = [bool(r["passed"]) for r in rows]
        uncs = [r["u"]["aggregate"] for r in rows]
        summary[k] = {
            "pass@1": pass_at_k(passed, k=1),
            "ece":    uncertainty_calibration_ece(uncs, passed),
            "n":      len(rows),
        }
    out["summary"] = summary

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
