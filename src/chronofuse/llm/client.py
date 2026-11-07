"""Thin OpenAI-compatible chat client with a persistent on-disk cache.

Works with OpenAI, OpenRouter, Together, vLLM and Ollama (``/v1``) endpoints.
Every (model, messages, params) call is cached as JSON so experiments are
reproducible and re-runs cost nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from ..config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


class LLMClient:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024, cache_dir: str | Path = PROJECT_ROOT / ".cache" / "llm"):
        from openai import OpenAI

        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.client = OpenAI(base_url=base_url or os.getenv("LLM_BASE_URL"),
                             api_key=api_key or os.getenv("LLM_API_KEY") or "EMPTY")
        self.temperature, self.max_tokens = temperature, max_tokens
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, messages: list[dict], **kw) -> Path:
        blob = json.dumps({"m": self.model, "msg": messages, **kw}, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(blob.encode()).hexdigest() + ".json")

    def chat(self, messages: list[dict], json_mode: bool = False, retries: int = 3, **kw) -> str:
        params = {"temperature": kw.pop("temperature", self.temperature),
                  "max_tokens": kw.pop("max_tokens", self.max_tokens)}
        path = self._key(messages, json_mode=json_mode, **params)
        if path.exists():
            return json.loads(path.read_text())["content"]
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                           **params, **extra)
                content = resp.choices[0].message.content or ""
                path.write_text(json.dumps({"content": content}))
                return content
            except Exception:  # noqa: BLE001 - network/rate-limit errors
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def chat_json(self, messages: list[dict], **kw) -> dict:
        text = self.chat(messages, json_mode=True, **kw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)) if m else {}
