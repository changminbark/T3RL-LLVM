"""Hosted-inference backend over an OpenAI-compatible chat API (Together/Fireworks/OpenRouter/…).

Config supplies base_url, model id, and the env var holding the API key. Sampling K completions
is done with `n` if the provider supports it, else K sequential requests.
"""

from __future__ import annotations

import os

import httpx

from .base import LLMBackend


class ApiBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        supports_n: bool = True,
        timeout_s: float = 120.0,
    ):
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"missing API key: set ${api_key_env}")
        self.model = model
        self.supports_n = supports_n
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout_s,
        )

    def generate(self, prompt: str, k: int, temperature: float, max_tokens: int) -> list[str]:
        if self.supports_n:
            return self._one_request(prompt, k, temperature, max_tokens)
        out: list[str] = []
        for _ in range(k):
            out.extend(self._one_request(prompt, 1, temperature, max_tokens))
        return out

    def _one_request(self, prompt: str, n: int, temperature: float, max_tokens: int) -> list[str]:
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": n,
            },
        )
        resp.raise_for_status()
        return [c["message"]["content"] for c in resp.json()["choices"]]
