"""Hosted-inference backend over an OpenAI-compatible chat API (Together/Fireworks/OpenRouter/…).

Config supplies base_url, model id, and the env var holding the API key. Sampling K completions
is done with `n` if the provider supports it, else K sequential requests.
"""

from __future__ import annotations

import os
import time

import httpx

from .base import LLMBackend


class ApiBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        supports_n: bool = True,
        timeout_s: float = 600.0,
        retries: int = 2,
    ):
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"missing API key: set ${api_key_env}")
        self.model = model
        self.supports_n = supports_n
        self.retries = retries
        # Reasoning models can take minutes on a single call; a big read timeout avoids
        # killing a long run over one slow completion.
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
        """Return n completions. On persistent failure return n empty strings rather than raising,
        so a single slow/failed call degrades to `invalid_syntax` samples instead of killing the run.
        """
        for attempt in range(self.retries + 1):
            try:
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
            except (httpx.HTTPError, KeyError, ValueError) as e:
                if attempt == self.retries:
                    print(f"  [api] giving up after {attempt + 1} attempts: {type(e).__name__}: {e}")
                    return [""] * n
                time.sleep(2 * (attempt + 1))
        return [""] * n
