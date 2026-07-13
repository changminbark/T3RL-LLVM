"""Offline mock backend: echoes the source IR from the prompt back as the 'rewrite'.

Lets the whole pipeline run end-to-end with no API key or GPU. Against StubVerifier, an echoed
source is normalized-equal -> `verified`, and StubPerf gives it no speedup -> `verified_no_gain`.
That exercises every stage (extract -> verify -> score -> classify) deterministically.
"""

from __future__ import annotations

import re

from .base import LLMBackend

# Grab the first fenced block in the prompt — that's the original function we echo back.
_FIRST_BLOCK = re.compile(r"```[A-Za-z0-9_+-]*\r?\n(.*?)```", re.DOTALL)


class MockBackend(LLMBackend):
    def __init__(self, lang: str = "llvm"):
        self.lang = lang

    def generate(self, prompt: str, k: int, temperature: float, max_tokens: int) -> list[str]:
        m = _FIRST_BLOCK.search(prompt)
        body = m.group(1).strip() if m else "define i32 @f() {\n  ret i32 0\n}"
        completion = f"```{self.lang}\n{body}\n```"
        return [completion] * k
