"""LLM client that logs every call through the Ledger.

Targets any OpenAI-compatible endpoint (local vLLM here). Every call carries a
`purpose` tag so the ledger can attribute generator compute to teach / critic /
probe / grade / diagnose / emit.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from .ledger import Ledger


@dataclass
class Endpoint:
    base_url: str
    model: str
    api_key: str = "EMPTY"


class LLM:
    def __init__(self, endpoint: Endpoint, ledger: Ledger,
                 temperature: float = 0.7, max_retries: int = 4):
        self.ep = endpoint
        self.ledger = ledger
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key,
                             timeout=180.0)

    def chat(self, messages: list[dict], purpose: str,
             temperature: float | None = None, max_tokens: int = 2048,
             seed: int | None = None, think: bool | None = None) -> str:
        """One chat completion, logged. Returns the assistant text (Qwen3 <think>
        block stripped). `think` toggles Qwen3 thinking mode per call."""
        temp = self.temperature if temperature is None else temperature
        extra = {}
        if think is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": think}
        last_err = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.ep.model, messages=messages,
                    temperature=temp, max_tokens=max_tokens, seed=seed,
                    extra_body=extra or None,
                )
                dt = time.time() - t0
                u = resp.usage
                self.ledger.record_call(
                    purpose,
                    prompt_tokens=getattr(u, "prompt_tokens", 0),
                    completion_tokens=getattr(u, "completion_tokens", 0),
                    wall_s=dt)
                return _strip_think(resp.choices[0].message.content or "")
            except Exception as e:  # noqa: BLE001 -- retry any transient server error
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def complete(self, prompt: str, purpose: str, temperature: float | None = None,
                 max_tokens: int = 256, seed: int | None = None,
                 stop: list[str] | None = None) -> str:
        """One text completion (for BASE models with no chat template), logged."""
        temp = self.temperature if temperature is None else temperature
        last_err = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = self.client.completions.create(
                    model=self.ep.model, prompt=prompt, temperature=temp,
                    max_tokens=max_tokens, seed=seed, stop=stop)
                dt = time.time() - t0
                u = resp.usage
                self.ledger.record_call(
                    purpose,
                    prompt_tokens=getattr(u, "prompt_tokens", 0),
                    completion_tokens=getattr(u, "completion_tokens", 0),
                    wall_s=dt)
                return resp.choices[0].text or ""
            except Exception as e:  # noqa: BLE001 -- retry any transient server error
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM completion failed after {self.max_retries} retries: {last_err}")

    def chat_json(self, messages: list[dict], purpose: str, **kw) -> dict | list:
        """Chat + best-effort JSON parse from the reply (handles ```json fences)."""
        txt = self.chat(messages, purpose, **kw)
        return extract_json(txt)


def _strip_think(text: str) -> str:
    """Drop a Qwen3 <think>...</think> block, keep the answer."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def extract_json(text: str):
    """Pull the first JSON object/array out of a model reply."""
    # strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    text = text.strip()
    # direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # find the outermost {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                continue
    raise ValueError(f"No JSON found in reply: {text[:200]!r}")
