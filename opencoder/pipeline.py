"""End-to-end OpenCoder pipeline.

Wires Phases I-V together. The pipeline is structured so RQ1 ablations
(toggle one retrieval source on/off) and RQ2 evaluations (with/without
uncertainty-aware scoring) reuse the exact same execution path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .data.loaders import Example
from .embeddings import Encoder
from .llm import LLMClient
from .phase1_repo_knowledge import (
    describe_functions,
    extract_repo_knowledge,
    profile_knowledge_uncertainty,
)
from .phase2_query import (
    decompose_into_steps,
    estimate_step_uncertainty,
    predict_retrieval_intent,
)
from .phase3_retrieval import (
    APIRetriever,
    ContextRetriever,
    SimilarCodeRetriever,
    fuse_evidence,
)
from .phase3_retrieval.score_filter import merge_step_candidates, score_and_filter
from .phase4_generation import generate_target_code
from .phase5_verify import repair_code, run_tests, static_check


@dataclass
class PipelineConfig:
    llm_backend: str = "openai"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "microsoft/unixcoder-base"
    embedding_device: str = "cpu"
    api_top_k: int = 10
    context_top_k: int = 10
    similar_code_top_k: int = 10
    fused_top_k: int = 12
    keep_quantile: float = 0.7
    knowledge_uncertainty_alpha: float = 0.5
    n_samples_for_uncertainty: int = 5
    max_repair_rounds: int = 2
    enable_sources: tuple = ("api", "context", "similar_code")  # for RQ1 ablation
    uncertainty_aware: bool = True                              # for RQ2 ablation

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        flat = {}
        flat["llm_backend"] = data["llm"]["backend"]
        flat["llm_model"] = data["llm"]["model"]
        flat["embedding_model"] = data["embedding"]["model"]
        flat["embedding_device"] = data["embedding"]["device"]
        flat["api_top_k"] = data["retrieval"]["api_top_k"]
        flat["context_top_k"] = data["retrieval"]["context_top_k"]
        flat["similar_code_top_k"] = data["retrieval"]["similar_code_top_k"]
        flat["fused_top_k"] = data["retrieval"]["fused_top_k"]
        flat["keep_quantile"] = data["retrieval"]["uncertainty_filter_quantile"]
        flat["n_samples_for_uncertainty"] = data["llm"]["n_samples_for_uncertainty"]
        flat["max_repair_rounds"] = data["verification"]["max_repair_rounds"]
        return cls(**flat)


@dataclass
class PipelineRun:
    example_id: str
    code: str
    uncertainty_trace: Dict[str, float]
    per_step: List[Dict[str, Any]] = field(default_factory=list)
    static_report: Dict[str, Any] = field(default_factory=dict)
    test_report: Dict[str, Any] = field(default_factory=dict)
    repair_rounds: int = 0


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.llm = LLMClient(backend=cfg.llm_backend, model=cfg.llm_model)
        self.encoder = Encoder(model_name=cfg.embedding_model, device=cfg.embedding_device)

    # ---- Phase I ----
    def index_repo(self, repo_root: str, describe_limit: Optional[int] = None):
        items = extract_repo_knowledge(repo_root)
        if describe_limit:
            items = items[:describe_limit]
        items = describe_functions(items, self.llm)
        items = profile_knowledge_uncertainty(items)

        api = APIRetriever(self.encoder)
        api.build(items)
        sim = SimilarCodeRetriever(self.encoder)
        sim.build(items)

        # Context: read raw files.
        ctx = ContextRetriever(self.encoder)
        files = []
        for it in {x.file_path for x in items}:
            p = os.path.join(repo_root, it)
            try:
                files.append((it, open(p, encoding="utf-8", errors="ignore").read()))
            except Exception:
                pass
        ctx.build_from_files(files)
        return items, {"api": api, "context": ctx, "similar_code": sim}

    # ---- End-to-end ----
    def run(self, example: Example, retrievers: Dict[str, Any]) -> PipelineRun:
        cfg = self.cfg
        # Phase II
        steps = decompose_into_steps(example.query, self.llm)
        steps = estimate_step_uncertainty(steps, self.llm)
        intents = predict_retrieval_intent(steps, self.llm)

        # Phase III
        per_step_candidates = []
        per_step_debug = []
        for step, intent in zip(steps, intents):
            hits_by_source = {}
            for src in cfg.enable_sources:
                q = intent.queries.get(src, step.description)
                top_k = getattr(cfg, f"{src}_top_k", 10) if src != "similar_code" else cfg.similar_code_top_k
                hits_by_source[src] = retrievers[src].search(q, top_k=top_k)
            sw = {s: (intent.source_weights.get(s, 0.0) if s in cfg.enable_sources else 0.0)
                  for s in ("api", "context", "similar_code")}
            alpha = cfg.knowledge_uncertainty_alpha if cfg.uncertainty_aware else 0.0
            keep_q = cfg.keep_quantile if cfg.uncertainty_aware else 1.0
            cands = score_and_filter(
                hits_by_source, sw,
                knowledge_uncertainty_alpha=alpha,
                keep_quantile=keep_q,
            )
            per_step_candidates.append(cands)
            per_step_debug.append({
                "step": step.description,
                "step_uncertainty": step.uncertainty,
                "intent": intent.source_weights,
                "n_candidates": len(cands),
            })

        merged = merge_step_candidates(per_step_candidates, cfg.fused_top_k)
        evidence = fuse_evidence(merged)

        # Phase IV
        query_unc = {
            "n_steps": len(steps),
            "mean_step_uncertainty": (sum(s.uncertainty for s in steps) / max(1, len(steps))),
        }
        gen = generate_target_code(
            example.query, evidence, query_unc, self.llm, self.encoder,
            n_samples=cfg.n_samples_for_uncertainty,
        )

        # Phase V
        code = gen.code
        static_rep = static_check(code)
        test_rep = run_tests(code, example.test_code)
        rounds = 0
        while not test_rep.passed and rounds < cfg.max_repair_rounds:
            diag = f"static: {static_rep.__dict__}\ntests:\n{test_rep.stderr}\n{test_rep.stdout}"
            code = repair_code(code, diag, self.llm)
            static_rep = static_check(code)
            test_rep = run_tests(code, example.test_code)
            rounds += 1

        return PipelineRun(
            example_id=example.id,
            code=code,
            uncertainty_trace=gen.trace.to_dict(),
            per_step=per_step_debug,
            static_report=static_rep.__dict__,
            test_report=test_rep.__dict__,
            repair_rounds=rounds,
        )
