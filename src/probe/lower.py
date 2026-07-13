"""Format B: lower model-emitted C source to LLVM IR via clang.

Used when the chosen generation format is C. If clang is missing or the C fails to compile,
we return None so the caller records `invalid_syntax`.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

_CLANG = shutil.which("clang")


def clang_available() -> bool:
    return _CLANG is not None


def lower_c_to_ir(c_source: str, timeout_s: int = 20) -> str | None:
    """Compile C to textual LLVM IR at -O0. Returns the .ll text, or None on failure."""
    if _CLANG is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "mod.c"
        out = Path(td) / "mod.ll"
        src.write_text(c_source)
        try:
            proc = subprocess.run(
                [_CLANG, "-O0", "-emit-llvm", "-S", "-o", str(out), str(src)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if proc.returncode != 0 or not out.exists():
            return None
        return out.read_text()
