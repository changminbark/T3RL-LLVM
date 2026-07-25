"""TimingPerf: signature parsing + IR normalization (offline) and native timing (guarded by clang).

Also covers the metric-agnostic reward path: classify() with a fake ns-based scorer, no clang needed.
"""

import pytest

from probe.outcome import classify
from probe.perf import PerfScorer
from probe.schema import CorpusRecord, GenFormat, PerfScore, RewriteOutcome, Verdict, VerdictStatus
from probe.timing import TimingPerf, normalize_ir, parse_signature
from probe.verifier import VerifierHarness

# ---------- signature parsing (pure, offline) ----------

def test_parse_scalar():
    sig = parse_signature("define dso_local i32 @f(i32 noundef %0) #0 {\n ret i32 0\n}")
    assert sig.name == "f" and sig.ret == "int32_t" and sig.params == ["int"]


def test_parse_ptr_with_attr_parens():
    # `captures(none)` and a trailing length int must both survive the balanced-paren scan.
    ir = "define dso_local i32 @g(ptr noundef readonly captures(none) %0, i32 noundef %1) local_unnamed_addr #0 {"
    sig = parse_signature(ir)
    assert sig.params == ["ptr", "int"] and sig.param_c == ["int32_t*", "int32_t"]


def test_parse_void_return():
    sig = parse_signature("define dso_local void @h(ptr noundef %0, i32 noundef %1) #0 {")
    assert sig.ret == "void" and sig.params == ["ptr", "int"]


def test_parse_unsupported_float_is_none():
    assert parse_signature("define dso_local float @fl(float %0) {") is None


def test_parse_no_define_is_none():
    assert parse_signature("not ir at all") is None


def test_normalize_strips_target_lines():
    ir = ('target datalayout = "e-m:o"\ntarget triple = "aarch64-linux-gnu"\n'
          "define i32 @f() {\n ret i32 0\n}")
    out = normalize_ir(ir)
    assert "target datalayout" not in out and "target triple" not in out
    assert "define i32 @f()" in out


# ---------- metric-agnostic classify (offline: fake ns scorer) ----------

_SRC = "define i32 @f(i32 %x) {\n  ret i32 %x\n}"
_O3 = "define i32 @f(i32 %x) {\n  ret i32 %x\n}"
_FAST = "define i32 @f(i32 %x) {\n  ret i32 %x ; fast\n}"
_REC = CorpusRecord(function_id="f", src_ir=_SRC, o3_baseline_ir=_O3, n_instructions=1)


class _FixedVerifier(VerifierHarness):
    def check(self, src_ir, tgt_ir, timeout_s=30):
        return Verdict(status=VerdictStatus.verified)


class _FakeTiming(PerfScorer):
    """Returns fixed wall_ns per IR string; O3 baseline = 50 ns."""
    _NS = {_FAST: 10.0}

    def score(self, ir):
        return PerfScore(wall_ns=self._NS.get(ir, 50.0))

    def cost(self, score):
        return score.wall_ns


def test_classify_uses_ns_cost_verified_faster():
    r = classify(_REC, f"```llvm\n{_FAST}\n```", 0, GenFormat.ir, _FixedVerifier(), _FakeTiming())
    assert r.outcome is RewriteOutcome.verified_faster
    assert r.rewrite_cycles == 10.0          # cost is ns here, not mca cycles
    assert abs(r.speedup_vs_o3 - 5.0) < 1e-9  # 50 ns baseline / 10 ns rewrite


def test_classify_ns_no_gain_when_not_faster():
    slow = "define i32 @f(i32 %x) {\n  ret i32 %x ; slow\n}"  # not in _NS -> 50 ns == baseline
    r = classify(_REC, f"```llvm\n{slow}\n```", 0, GenFormat.ir, _FixedVerifier(), _FakeTiming())
    assert r.outcome is RewriteOutcome.verified_no_gain


# ---------- native timing (guarded by clang) ----------

_TP = TimingPerf()
requires_clang = pytest.mark.skipif(not _TP.available(), reason="clang not on PATH")


@requires_clang
def test_timing_scores_int_function():
    s = _TP.score("define i32 @sq(i32 %x) {\n  %m = mul i32 %x, %x\n  ret i32 %m\n}")
    assert s is not None and s.wall_ns is not None and s.wall_ns >= 0.0


@requires_clang
def test_timing_pointer_function_runs():
    ir = ("define i32 @asum(ptr %0, i32 %1) {\nentry:\n  br label %l\nl:\n"
          "  %i = phi i32 [0,%entry],[%n,%l]\n  %a = phi i32 [0,%entry],[%s,%l]\n"
          "  %p = getelementptr i32, ptr %0, i32 %i\n  %v = load i32, ptr %p\n"
          "  %s = add i32 %a, %v\n  %n = add i32 %i, 1\n  %c = icmp slt i32 %n, %1\n"
          "  br i1 %c, label %l, label %d\nd:\n  ret i32 %s\n}")
    s = _TP.score(ir)
    assert s is not None and s.wall_ns is not None


@requires_clang
def test_timing_unsupported_returns_none():
    assert _TP.score("define float @fl(float %x) {\n  ret float %x\n}") is None
