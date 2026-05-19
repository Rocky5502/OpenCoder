"""Phase V, Step 13: Repair / Refine.

Sends the failed code + diagnostics back to the LLM for a focused
repair pass. The feedback also flows to Phase III to trigger another
retrieval round if requested (see pipeline.run() loop).
"""
from __future__ import annotations

import re

from ..llm.client import LLMClient

_SYSTEM = (
    "You are repairing Python code that failed verification. Read the diagnostics, "
    "produce a corrected version. Return ONLY a single Python code block."
)

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def repair_code(code: str, diagnostics: str, llm: LLMClient) -> str:
    prompt = (
        f"# Failed Code\n```python\n{code}\n```\n\n"
        f"# Diagnostics\n{diagnostics}\n\n"
        f"# Instruction\nReturn a corrected version that addresses the diagnostics."
    )
    resp = llm.complete_one(prompt, system=_SYSTEM, max_tokens=1200, temperature=0.2, return_logprobs=False)
    m = _FENCE.search(resp.text)
    return (m.group(1) if m else resp.text).strip()
