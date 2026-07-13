"""LLM backend interface. Concrete backends are selected by config; the driver never imports one."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, k: int, temperature: float, max_tokens: int
    ) -> list[str]:
        """Return `k` completions for `prompt`. Implementations should sample independently."""
        ...
