"""Normalize model-emitted LLVM IR into something Alive2 can parse.

Frontier models tend to wrap a rewrite in a whole *module* — `; ModuleID`, `target datalayout`,
`declare`s, attribute groups — and often abbreviate the boilerplate (`target datalayout = "..."`) or
reference attribute groups (`#0`, `#1`) whose definitions are truncated. Any of these makes the IR
unparseable, so the sample is wasted as `invalid_syntax` even when the actual function is fine.

`sanitize_module` fixes the boilerplate without touching the logic:
  - replace/inject `target datalayout` + `target triple` from the (valid) source module,
  - drop `attributes #N = { ... }` blocks and strip `#N` attribute-group *references*,
  - keep the model's `declare`s, globals, type defs, metadata, and the `define` itself.

It is a text transform (no LLVM needed), so it's fast and unit-testable offline. Returns None if the
output contains no function definition.
"""

from __future__ import annotations

import re

_ATTR_REF = re.compile(r"\s#\d+\b")  # e.g. " #0" on a define/declare/call line
_DEFINE_LINE = re.compile(r"(?m)^\s*define\b")


def _src_line(src_ir: str, prefix: str) -> str | None:
    for ln in src_ir.splitlines():
        if ln.lstrip().startswith(prefix):
            return ln
    return None


def sanitize_module(model_ir: str, src_ir: str) -> str | None:
    """Return a parseable module for `model_ir`, borrowing target lines from `src_ir`. None if no define."""
    src_dl = _src_line(src_ir, "target datalayout")
    src_tt = _src_line(src_ir, "target triple")

    out: list[str] = []
    have_dl = have_tt = False
    for ln in model_ir.splitlines():
        s = ln.lstrip()
        if s.startswith("attributes #"):
            continue  # drop attribute-group definitions (we strip their refs below)
        if s.startswith("target datalayout"):
            out.append(src_dl if src_dl else ln)
            have_dl = True
        elif s.startswith("target triple"):
            out.append(src_tt if src_tt else ln)
            have_tt = True
        else:
            out.append(_ATTR_REF.sub("", ln))

    prefix: list[str] = []
    if not have_dl and src_dl:
        prefix.append(src_dl)
    if not have_tt and src_tt:
        prefix.append(src_tt)

    text = "\n".join(prefix + out)
    if not _DEFINE_LINE.search(text):
        return None
    return text + "\n"
