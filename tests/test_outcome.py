"""Classification logic against synthetic verifier/perf, covering each RewriteOutcome branch."""

from probe.outcome import classify
from probe.perf import StubPerf
from probe.schema import (
    CorpusRecord,
    GenFormat,
    RewriteOutcome,
    Verdict,
    VerdictStatus,
)
from probe.verifier import VerifierHarness

SRC = "define i32 @f(i32 %x) {\n  %a = add i32 %x, 0\n  %b = add i32 %a, 0\n  ret i32 %b\n}"
REC = CorpusRecord(function_id="f", src_ir=SRC, n_instructions=3)


class FixedVerifier(VerifierHarness):
    def __init__(self, status, cex=None):
        self.v = Verdict(status=status, counterexample=cex)

    def check(self, src_ir, tgt_ir, timeout_s=30):
        return self.v


def _completion(ir):
    return f"```llvm\n{ir}\n```"


def test_invalid_syntax_when_no_code():
    r = classify(REC, "no code here", 0, GenFormat.ir,
                 FixedVerifier(VerdictStatus.verified), StubPerf())
    assert r.outcome is RewriteOutcome.invalid_syntax


def test_counterexample_passthrough():
    r = classify(REC, _completion("define i32 @f(i32 %x){ ret i32 %x }"), 0, GenFormat.ir,
                 FixedVerifier(VerdictStatus.counterexample, cex="x=5"), StubPerf())
    assert r.outcome is RewriteOutcome.counterexample
    assert r.counterexample == "x=5"


def test_timeout_passthrough():
    r = classify(REC, _completion("define i32 @f(i32 %x){ ret i32 %x }"), 0, GenFormat.ir,
                 FixedVerifier(VerdictStatus.timeout), StubPerf())
    assert r.outcome is RewriteOutcome.timeout


def test_verified_faster_when_fewer_cycles_than_source():
    # StubPerf counts body instructions: SRC has 3, this rewrite has 1 -> strictly faster.
    rewrite = "define i32 @f(i32 %x) {\n  ret i32 %x\n}"
    r = classify(REC, _completion(rewrite), 0, GenFormat.ir,
                 FixedVerifier(VerdictStatus.verified), StubPerf())
    assert r.outcome is RewriteOutcome.verified_faster
    assert r.speedup_vs_o3 is not None and r.speedup_vs_o3 > 1.0


def test_verified_no_gain_when_not_faster():
    # rewrite == source -> equal cycles -> no gain
    r = classify(REC, _completion(SRC), 0, GenFormat.ir,
                 FixedVerifier(VerdictStatus.verified), StubPerf())
    assert r.outcome is RewriteOutcome.verified_no_gain
