"""Shared data models. The JSONL corpus schema here is the day-one contract with Person A.

Keep `CorpusRecord`, `Verdict`, and `PerfScore` in lockstep with `docs/phase1-partB-plan.md`.
Any change to these must be mirrored in that doc in the same commit (the sync mechanism).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CorpusRecord(BaseModel):
    """One function from the corpus. Produced by Person A, consumed by the probe."""

    function_id: str
    src_ir: str  # LLVM IR text, the -O0 source function
    source_lang: str = "c"  # "c" | "llvm" | ...
    n_instructions: int = 0
    has_loops: bool = False
    # The -O3 optimized IR — the "beat the compiler" baseline. May be None during bootstrap
    # (before Person A's corpus / local llvm tooling is available).
    o3_baseline_ir: str | None = None
    mca_cycles_o3: float | None = None

    def size_bucket(self) -> str:
        """Size buckets used for reporting, per the plan."""
        n = self.n_instructions
        if n <= 20:
            return "<=20"
        if n <= 50:
            return "20-50"
        if n <= 150:
            return "50-150"
        return ">150"


class VerdictStatus(str, Enum):
    verified = "verified"
    counterexample = "counterexample"
    timeout = "timeout"
    unsupported = "unsupported"
    error = "error"


class Verdict(BaseModel):
    """Result of an equivalence check on a (src, tgt) IR pair. Returned by the verifier harness."""

    status: VerdictStatus
    counterexample: str | None = None
    wall_time_s: float = 0.0


class PerfScore(BaseModel):
    """Performance score for a single IR module. mca_cycles is primary; code_size secondary."""

    mca_cycles: float
    code_size_bytes: int = 0


class GenFormat(str, Enum):
    """How the model is asked to emit its rewrite."""

    ir = "ir"  # format A: emit LLVM IR directly
    c = "c"  # format B: emit C, lowered to IR by clang


class RewriteOutcome(str, Enum):
    """One label per generated rewrite — the future RL reward distribution.

    Ordered from worst to best signal.
    """

    invalid_syntax = "invalid_syntax"
    counterexample = "counterexample"
    timeout = "timeout"
    unsupported = "unsupported"
    verified_no_gain = "verified_no_gain"
    verified_faster = "verified_faster"


class RewriteResult(BaseModel):
    """Full record of classifying one model completion. Written to results/ JSONL."""

    function_id: str
    sample_index: int
    outcome: RewriteOutcome
    # Speedup vs the O3 baseline (o3_cycles / rewrite_cycles), when both are known.
    speedup_vs_o3: float | None = None
    rewrite_cycles: float | None = None
    verdict_wall_time_s: float = 0.0
    counterexample: str | None = None
    raw_completion: str = Field(default="", repr=False)
    extracted_ir: str | None = Field(default=None, repr=False)
