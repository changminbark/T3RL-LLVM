"""Equivalence-checking harness.

`StubVerifier` lets the whole pipeline run before Person A's Alive2 CLI exists: it can only
recognize trivially-equal IR (normalized textual match) as `verified`, everything else as
`unsupported`. `AliveCliVerifier` shells out to Person A's `alive-tv` wrapper and parses the
`Verdict` JSON contract.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .schema import Verdict, VerdictStatus


class VerifierHarness(ABC):
    @abstractmethod
    def check(self, src_ir: str, tgt_ir: str, timeout_s: int = 30) -> Verdict: ...


def _normalize_ir(ir: str) -> str:
    """Cheap structural normalization: drop comments/blank lines, collapse whitespace.

    Good enough for the stub to catch identical-modulo-formatting rewrites (e.g. the "output the
    input unchanged" cheat) — NOT a substitute for Alive2's semantic proof.
    """
    lines = []
    for line in ir.splitlines():
        line = re.sub(r";.*$", "", line)  # strip line comments
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


class StubVerifier(VerifierHarness):
    """Bootstrap verifier: normalized-equality only. No theorem proving."""

    def check(self, src_ir: str, tgt_ir: str, timeout_s: int = 30) -> Verdict:
        start = time.perf_counter()
        equal = _normalize_ir(src_ir) == _normalize_ir(tgt_ir)
        status = VerdictStatus.verified if equal else VerdictStatus.unsupported
        return Verdict(status=status, wall_time_s=time.perf_counter() - start)


class AliveCliVerifier(VerifierHarness):
    """Wrap Person A's alive-tv CLI. Expects it to emit the `Verdict` JSON on stdout.

    `cli_cmd` is the executable/wrapper; it is invoked as: `<cli_cmd> <src.ll> <tgt.ll> --timeout <s>`.
    Adjust the argument shape here once Person A finalizes their CLI (single point of change).
    """

    def __init__(self, cli_cmd: str = "alive-harness"):
        resolved = shutil.which(cli_cmd) or cli_cmd
        self.cli_cmd = resolved

    def check(self, src_ir: str, tgt_ir: str, timeout_s: int = 30) -> Verdict:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.ll"
            tgt = Path(td) / "tgt.ll"
            src.write_text(src_ir)
            tgt.write_text(tgt_ir)
            try:
                proc = subprocess.run(
                    [self.cli_cmd, str(src), str(tgt), "--timeout", str(timeout_s)],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s + 10,  # give the CLI its own hard timeout first
                )
            except subprocess.TimeoutExpired:
                return Verdict(
                    status=VerdictStatus.timeout, wall_time_s=time.perf_counter() - start
                )
            except OSError as e:
                return Verdict(
                    status=VerdictStatus.error,
                    counterexample=str(e),
                    wall_time_s=time.perf_counter() - start,
                )
        return self._parse(proc.stdout, proc.stderr, time.perf_counter() - start)

    @staticmethod
    def _parse(stdout: str, stderr: str, wall: float) -> Verdict:
        try:
            data = json.loads(stdout.strip().splitlines()[-1])
            v = Verdict.model_validate(data)
            if not v.wall_time_s:
                v.wall_time_s = wall
            return v
        except (json.JSONDecodeError, ValueError, IndexError):
            return Verdict(
                status=VerdictStatus.error,
                counterexample=(stderr or stdout)[:2000],
                wall_time_s=wall,
            )


def make_verifier(kind: str, **kwargs) -> VerifierHarness:
    if kind == "stub":
        return StubVerifier()
    if kind == "alive":
        return AliveCliVerifier(**kwargs)
    raise ValueError(f"unknown verifier kind: {kind!r}")
