"""Pluggable LLM client.

Supports two backends:
  - openai   : api.openai.com  (paper-aligned)
  - lovable  : ai.gateway.lovable.dev  (dev-time, free credits)

Both implement an OpenAI-compatible /chat/completions interface, so the
request shape is identical. The client returns text + token logprobs
when available (used by the uncertainty modules).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


@dataclass
class LLMResponse:
    text: str
    logprobs: Optional[List[float]] = None          # per-token logprob of the chosen token
    tokens: Optional[List[str]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_logprob(self) -> Optional[float]:
        if not self.logprobs:
            return None
        return sum(self.logprobs) / len(self.logprobs)

    @property
    def token_entropy_proxy(self) -> Optional[float]:
        """Entropy proxy from per-token logprobs of the chosen token.
        Higher = less confident. Bounded approximation: -mean(logprob)."""
        mlp = self.mean_logprob
        return None if mlp is None else -mlp


class LLMClient:
    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 120,
    ):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if backend == "openai":
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.api_key = os.environ.get("OPENAI_API_KEY")
        elif backend == "lovable":
            self.endpoint = "https://ai.gateway.lovable.dev/v1/chat/completions"
            self.api_key = os.environ.get("LOVABLE_API_KEY")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if not self.api_key:
            raise RuntimeError(
                f"Missing API key for backend={backend}. "
                f"Set {'OPENAI_API_KEY' if backend == 'openai' else 'LOVABLE_API_KEY'}."
            )

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        return_logprobs: bool = True,
    ) -> List[LLMResponse]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "n": n,
        }
        if return_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 1

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(self.endpoint, json=body, headers=headers, timeout=self.timeout)
        if r.status_code == 429:
            raise RuntimeError("Rate limited (429). Back off and retry.")
        if r.status_code == 402:
            raise RuntimeError("Payment required (402). Add credits to your workspace.")
        r.raise_for_status()
        data = r.json()

        out: List[LLMResponse] = []
        for choice in data.get("choices", []):
            text = choice["message"]["content"] or ""
            lp = choice.get("logprobs") or {}
            content_lp = lp.get("content") if isinstance(lp, dict) else None
            logprobs, tokens = None, None
            if content_lp:
                logprobs = [t.get("logprob") for t in content_lp if t.get("logprob") is not None]
                tokens = [t.get("token") for t in content_lp]
            out.append(LLMResponse(text=text, logprobs=logprobs, tokens=tokens, raw=choice))
        return out

    def complete_one(self, prompt: str, system: Optional[str] = None, **kw) -> LLMResponse:
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.complete(msgs, **kw)[0]
