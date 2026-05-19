"""Phase I, Step 1: Extract Knowledge.

Walks a repository, parses each .py file with Python's ast module, and
emits one record per function/method/class. Pure stdlib — no external
parser. For other languages, extend with a tree-sitter backend.
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional


@dataclass
class RepoFunction:
    file_path: str
    qualname: str
    signature: str
    docstring: Optional[str]
    body: str
    start_line: int
    end_line: int
    kind: str = "function"          # function | method | class
    description: Optional[str] = None
    uncertainty: Optional[float] = None
    metadata: dict = field(default_factory=dict)


def _iter_py_files(root: str) -> Iterator[str]:
    skip = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _segment(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def extract_repo_knowledge(root: str, max_files: Optional[int] = None) -> List[RepoFunction]:
    items: List[RepoFunction] = []
    for i, path in enumerate(_iter_py_files(root)):
        if max_files is not None and i >= max_files:
            break
        try:
            src = open(path, "r", encoding="utf-8", errors="ignore").read()
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = f"{node.name}({', '.join(a.arg for a in node.args.args)})"
                items.append(
                    RepoFunction(
                        file_path=os.path.relpath(path, root),
                        qualname=node.name,
                        signature=sig,
                        docstring=ast.get_docstring(node),
                        body=_segment(src, node),
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        kind="function",
                    )
                )
            elif isinstance(node, ast.ClassDef):
                items.append(
                    RepoFunction(
                        file_path=os.path.relpath(path, root),
                        qualname=node.name,
                        signature=f"class {node.name}",
                        docstring=ast.get_docstring(node),
                        body=_segment(src, node),
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        kind="class",
                    )
                )
    return items
