"""Resolve the LLVM toolchain and target triple.

All LLVM tools (clang, llc, llvm-mca, llvm-as) must come from the *same* install so the IR one
emits is consumable by the next. On macOS this matters twice over: Apple's `clang` on PATH emits
IR with Apple-only target features that Homebrew's `llc`/`llvm-mca` crash on, and Homebrew LLVM is
keg-only (not on PATH). So we resolve a single LLVM bin dir and take every tool from it.

Resolution order for the bin dir:
  1. $LLVM_BIN (explicit override — set this on any non-standard setup)
  2. the directory containing `llvm-mca` on PATH
  3. common Homebrew keg locations

We also target a non-Darwin triple by default (`$PROBE_TARGET`, default aarch64-linux-gnu) so
llvm-mca's asm parser does not choke on macOS directives like `.subsections_via_symbols`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

TARGET_TRIPLE = os.environ.get("PROBE_TARGET", "aarch64-linux-gnu")

_FALLBACK_DIRS = [
    "/opt/homebrew/opt/llvm/bin",  # Homebrew on Apple Silicon
    "/usr/local/opt/llvm/bin",  # Homebrew on Intel macOS
]


def llvm_bin_dir() -> str | None:
    env = os.environ.get("LLVM_BIN")
    if env and Path(env).is_dir():
        return env
    mca = shutil.which("llvm-mca")
    if mca:
        return str(Path(mca).parent)
    for d in _FALLBACK_DIRS:
        if Path(d, "llvm-mca").exists():
            return d
    return None


def find_tool(name: str) -> str | None:
    """Absolute path to an LLVM tool, taken from the resolved bin dir; PATH is a last resort."""
    d = llvm_bin_dir()
    if d:
        cand = Path(d, name)
        if cand.exists():
            return str(cand)
    return shutil.which(name)
