"""Dataset loaders for ExecRepoBench, CoderEval, RepoExec.

Each loader normalizes records into the Example dataclass below. Only
the fields needed by the pipeline are required; the rest are kept in
`raw` for downstream analysis.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class Example:
    id: str
    query: str                # natural-language task description
    repo_root: Optional[str]  # path to a checked-out repo, if any
    reference_code: Optional[str] = None
    test_code: Optional[str] = None
    raw: dict = field(default_factory=dict)


def _read_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_execrepobench(path: str, limit: Optional[int] = None) -> Iterator[Example]:
    for i, rec in enumerate(_read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        yield Example(
            id=str(rec.get("task_id") or rec.get("id") or i),
            query=str(rec.get("prompt") or rec.get("instruction") or rec.get("query") or ""),
            repo_root=rec.get("repo_root"),
            reference_code=rec.get("reference") or rec.get("canonical_solution"),
            test_code=rec.get("test") or rec.get("test_code"),
            raw=rec,
        )


def load_codereval(path: str, limit: Optional[int] = None) -> Iterator[Example]:
    for i, rec in enumerate(_read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        yield Example(
            id=str(rec.get("_id") or rec.get("id") or i),
            query=str(rec.get("docstring") or rec.get("nl") or ""),
            repo_root=rec.get("project_path"),
            reference_code=rec.get("code"),
            test_code=rec.get("tests"),
            raw=rec,
        )


def load_repoexec(path: str, limit: Optional[int] = None) -> Iterator[Example]:
    for i, rec in enumerate(_read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        yield Example(
            id=str(rec.get("task_id") or i),
            query=str(rec.get("instruction") or rec.get("prompt") or ""),
            repo_root=rec.get("repo_path"),
            reference_code=rec.get("solution"),
            test_code=rec.get("test"),
            raw=rec,
        )


_REGISTRY = {
    "execrepobench": load_execrepobench,
    "codereval": load_codereval,
    "repoexec": load_repoexec,
}


def load_dataset(name: str, path: str, limit: Optional[int] = None) -> Iterator[Example]:
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Known: {list(_REGISTRY)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return _REGISTRY[name](path, limit=limit)
