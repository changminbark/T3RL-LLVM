"""Performance scorer.

`StubPerf` returns a deterministic proxy (instruction count) so the pipeline yields a usable
ranking before llvm-mca is installed. `McaPerf` compiles IR to assembly and runs llvm-mca for a
real cycle estimate. Both return a `PerfScore`.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from .schema import PerfScore
from .tools import TARGET_TRIPLE, find_tool


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
    """Real scorer: IR -> asm (llc) -> llvm-mca total cycles, for the configured target triple.

    We run llvm-mca with --iterations=1. Its default (100) treats the whole function as a loop body
    and lets the out-of-order engine overlap independent copies of stack-heavy -O0 code, which hides
    its true cost: on the bootstrap corpus that makes mca rank -O0 slower than -O3 only 70% of the
    time, vs 91% at --iterations=1. This is a single-execution cycle estimate, not a loop-aware one
    (static analysis cannot see real trip counts — see docs for the remaining failure modes).
    """

    def __init__(self, triple: str = TARGET_TRIPLE, iterations: int = 1):
        self.triple = triple
        self.iterations = iterations
        self.llc = find_tool("llc")
        self.mca = find_tool("llvm-mca")

    def available(self) -> bool:
        return self.llc is not None and self.mca is not None

    def score(self, ir: str) -> PerfScore | None:
        if not self.available():
            return None
        with tempfile.TemporaryDirectory() as td:
            ll = Path(td) / "mod.ll"
            asm = Path(td) / "mod.s"
            ll.write_text(ir)
            try:
                if subprocess.run(
                    [self.llc, "-mtriple", self.triple, "-o", str(asm), str(ll)],
                    capture_output=True,
                    timeout=30,
                ).returncode != 0:
                    return None
                proc = subprocess.run(
                    [self.mca, "-mtriple", self.triple,
                     f"--iterations={self.iterations}", str(asm)],
                    capture_output=True,
                    text=True,
                    timeout=30,
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
