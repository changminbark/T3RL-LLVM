"""Local vLLM backend. Imported lazily so API-only installs (no `local` extra) still work."""

from __future__ import annotations

from .base import LLMBackend


class VllmBackend(LLMBackend):
    def __init__(self, model: str, dtype: str = "auto", **engine_kwargs):
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on optional 'local' extra
            raise RuntimeError(
                "vllm not installed; run `uv sync --extra local` on a GPU machine"
            ) from e
        from vllm import LLM

        self._SamplingParams = __import__("vllm", fromlist=["SamplingParams"]).SamplingParams
        self.llm = LLM(model=model, dtype=dtype, **engine_kwargs)

    def generate(self, prompt: str, k: int, temperature: float, max_tokens: int) -> list[str]:
        params = self._SamplingParams(
            n=k, temperature=temperature, max_tokens=max_tokens
        )
        outputs = self.llm.generate([prompt], params)
        return [o.text for o in outputs[0].outputs]
