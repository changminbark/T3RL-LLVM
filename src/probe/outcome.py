"""Classify one model completion into a RewriteOutcome — the core reward logic.

Pipeline (per the plan):
  1. extract code; for format C, lower to IR. Failure -> invalid_syntax.
  2. verifier.check(src, rewrite). timeout/unsupported/counterexample map straight through.
  3. if verified: compare rewrite cycles vs the O3 baseline ->
       verified_faster (strictly fewer) else verified_no_gain.

When the O3 baseline cycle count is unknown (bootstrap), a verified rewrite that scores strictly
fewer cycles than the *source* is treated as `verified_faster`; otherwise `verified_no_gain`.
"""

from __future__ import annotations

from .extract import extract_code
from .lower import lower_c_to_ir
from .perf import PerfScorer
from .schema import (
    CorpusRecord,
    GenFormat,
    RewriteOutcome,
    RewriteResult,
    VerdictStatus,
)
from .verifier import VerifierHarness

_VERDICT_TO_OUTCOME = {
    VerdictStatus.counterexample: RewriteOutcome.counterexample,
    VerdictStatus.timeout: RewriteOutcome.timeout,
    VerdictStatus.unsupported: RewriteOutcome.unsupported,
    VerdictStatus.error: RewriteOutcome.invalid_syntax,
}


def classify(
    record: CorpusRecord,
    completion: str,
    sample_index: int,
    fmt: GenFormat,
    verifier: VerifierHarness,
    perf: PerfScorer,
    *,
    timeout_s: int = 30,
) -> RewriteResult:
    result = RewriteResult(
        function_id=record.function_id,
        sample_index=sample_index,
        outcome=RewriteOutcome.invalid_syntax,
        raw_completion=completion,
    )

    # 1. get IR out of the completion
    code = extract_code(completion, fmt)
    if code is None:
        return result
    rewrite_ir = code if fmt is GenFormat.ir else lower_c_to_ir(code)
    if not rewrite_ir:
        return result
    result.extracted_ir = rewrite_ir

    # 2. equivalence check
    verdict = verifier.check(record.src_ir, rewrite_ir, timeout_s=timeout_s)
    result.verdict_wall_time_s = verdict.wall_time_s
    result.counterexample = verdict.counterexample
    if verdict.status is not VerdictStatus.verified:
        result.outcome = _VERDICT_TO_OUTCOME[verdict.status]
        return result

    # 3. proven equivalent -> is it actually faster?
    score = perf.score(rewrite_ir)
    if score is None:
        # verified but we cannot measure speed -> treat as no measurable gain
        result.outcome = RewriteOutcome.verified_no_gain
        return result

    result.rewrite_cycles = score.mca_cycles
    baseline = _baseline_cycles(record, perf)
    if baseline is not None and score.mca_cycles > 0:
        result.speedup_vs_o3 = baseline / score.mca_cycles
    if baseline is not None and score.mca_cycles < baseline:
        result.outcome = RewriteOutcome.verified_faster
    else:
        result.outcome = RewriteOutcome.verified_no_gain
    return result


def _baseline_cycles(record: CorpusRecord, perf: PerfScorer) -> float | None:
    """Cycles to beat: the O3 baseline if known, else the source function under the same scorer."""
    if record.mca_cycles_o3 is not None:
        return record.mca_cycles_o3
    if record.o3_baseline_ir:
        s = perf.score(record.o3_baseline_ir)
        if s is not None:
            return s.mca_cycles
    s = perf.score(record.src_ir)
    return s.mca_cycles if s is not None else None
