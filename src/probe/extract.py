"""Pull code out of model completions.

Models wrap answers in prose and markdown fences. We prefer a fenced block tagged with the
expected language, fall back to any fenced block, then to the whole completion. Returning None
(no plausible code) lets the caller classify the rewrite as `invalid_syntax`.
"""

from __future__ import annotations

import re

from .schema import GenFormat

# ```llvm ... ```  /  ```c ... ```  /  ``` ... ```  (language tag optional)
_FENCE_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)

# Language tags we accept as "this is the right block" per format.
_LANG_ALIASES: dict[GenFormat, set[str]] = {
    GenFormat.ir: {"llvm", "llvm-ir", "ir", "ll"},
    GenFormat.c: {"c", "cpp", "c++"},
}


def extract_code(completion: str, fmt: GenFormat) -> str | None:
    """Return the most likely code block for `fmt`, or None if nothing plausible is present."""
    blocks = _FENCE_RE.findall(completion)  # list[(lang, body)]

    if blocks:
        wanted = _LANG_ALIASES[fmt]
        # 1) a fenced block whose tag matches the requested language
        for lang, body in blocks:
            if lang.lower() in wanted and body.strip():
                return body.strip()
        # 2) otherwise the largest non-empty fenced block (models often omit the tag)
        candidates = [b.strip() for _, b in blocks if b.strip()]
        if candidates:
            return max(candidates, key=len)

    # 3) no fences at all — treat the whole thing as code if it looks like it
    stripped = completion.strip()
    if _looks_like_code(stripped, fmt):
        return stripped
    return None


def _looks_like_code(text: str, fmt: GenFormat) -> bool:
    if not text:
        return False
    if fmt is GenFormat.ir:
        # LLVM IR function definitions and SSA registers are unmistakable.
        return "define " in text or "\n" in text and "%" in text
    # C: a function-ish signature with a brace.
    return "{" in text and "}" in text and "(" in text
