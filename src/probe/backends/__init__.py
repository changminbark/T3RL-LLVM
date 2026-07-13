"""Backend factory. The driver calls `make_backend(cfg)` and never imports a concrete backend."""

from __future__ import annotations

from .base import LLMBackend


def make_backend(kind: str, **kwargs) -> LLMBackend:
    if kind == "mock":
        from .mock_backend import MockBackend

        return MockBackend(**kwargs)
    if kind == "api":
        from .api_backend import ApiBackend

        return ApiBackend(**kwargs)
    if kind == "vllm":
        from .vllm_backend import VllmBackend

        return VllmBackend(**kwargs)
    raise ValueError(f"unknown backend kind: {kind!r}")


__all__ = ["LLMBackend", "make_backend"]
