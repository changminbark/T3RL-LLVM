"""Performance scorer.

`StubPerf` returns a deterministic proxy (instruction count) so the pipeline yields a usable
ranking before llvm-mca is installed. `McaPerf` compiles IR to assembly and runs llvm-mca for a
real cycle estimate. Both return a `PerfScore`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from .schema import PerfScore

_LLC = shutil.which("llc")
_MCA = shutil.which("llvm-mca")


class PerfScorer(ABC):
    @abstractmethod
    def score(self, ir: str) -> PerfScore | None: ...


def _count_instructions(ir: str) -> int:
    """Rough IR instruction count: non-label, non-brace lines inside function bodies."""
    n = 0
    in_body = False
    for line in ir.splitlines():
        s = line.strip()
        if s.startswith("define "):
            in_body = True
            continue
        if s == "}":
            in_body = False
            continue
        if in_body and s and not s.endswith(":") and not s.startswith(";"):
            n += 1
    return n


class StubPerf(PerfScorer):
    """Proxy scorer: fewer IR instructions == fewer 'cycles'. Deterministic, no tools needed."""

    def score(self, ir: str) -> PerfScore:
        n = _count_instructions(ir)
        return PerfScore(mca_cycles=float(n), code_size_bytes=len(ir.encode()))


class McaPerf(PerfScorer):
    """Real scorer: IR -> asm (llc) -> llvm-mca total cycles."""

    def score(self, ir: str) -> PerfScore | None:
        if _LLC is None or _MCA is None:
            return None
        with tempfile.TemporaryDirectory() as td:
            ll = Path(td) / "mod.ll"
            asm = Path(td) / "mod.s"
            ll.write_text(ir)
            try:
                if subprocess.run(
                    [_LLC, "-o", str(asm), str(ll)], capture_output=True, timeout=30
                ).returncode != 0:
                    return None
                proc = subprocess.run(
                    [_MCA, str(asm)], capture_output=True, text=True, timeout=30
                )
            except (subprocess.TimeoutExpired, OSError):
                return None
            if proc.returncode != 0:
                return None
            cycles = _parse_mca_cycles(proc.stdout)
            if cycles is None:
                return None
            return PerfScore(mca_cycles=cycles, code_size_bytes=asm.stat().st_size)


def _parse_mca_cycles(mca_stdout: str) -> float | None:
    m = re.search(r"Total Cycles:\s*(\d+)", mca_stdout)
    return float(m.group(1)) if m else None


def make_perf(kind: str) -> PerfScorer:
    if kind == "stub":
        return StubPerf()
    if kind == "mca":
        return McaPerf()
    raise ValueError(f"unknown perf kind: {kind!r}")
