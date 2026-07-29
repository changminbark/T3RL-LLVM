"""Rank-correlation helpers + audit aggregation (all offline, with a fake ns scorer)."""

from probe.perf import PerfScorer
from probe.schema import CorpusRecord, PerfScore, RewriteOutcome, RewriteResult
from probe.timing_validation import rewrite_audit, spearman


def test_spearman_perfect_monotonic():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9


def test_spearman_perfect_inverse():
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9


def test_spearman_handles_nonlinear_monotonic():
    # spearman is rank-based, so a monotonic-but-nonlinear relation is still 1.0
    assert abs(spearman([1, 2, 3, 4], [1, 4, 9, 16]) - 1.0) < 1e-9


def test_spearman_too_few_points():
    assert spearman([1], [1]) is None


class _MapTiming(PerfScorer):
    def __init__(self, table):
        self.table = table

    def score(self, ir):
        v = self.table.get(ir)
        return PerfScore(wall_ns=v) if v is not None else None

    def cost(self, score):
        return score.wall_ns


def test_rewrite_audit_confirms_and_correlates():
    o3 = "O3"
    rec = CorpusRecord(
        function_id="f", src_ir="SRC", o3_baseline_ir=o3, n_instructions=1
    )
    # two verified_faster rewrites: one really faster (5 ns < 50), one actually slower (80 > 50)
    rf = RewriteResult(
        function_id="f",
        sample_index=0,
        outcome=RewriteOutcome.verified_faster,
        speedup_vs_o3=10.0,
        extracted_ir="FAST",
    )
    rs = RewriteResult(
        function_id="f",
        sample_index=1,
        outcome=RewriteOutcome.verified_faster,
        speedup_vs_o3=1.2,
        extracted_ir="SLOW",
    )
    timing = _MapTiming({o3: 50.0, "FAST": 5.0, "SLOW": 80.0})
    out = rewrite_audit({"f": [rf, rs]}, {"f": rec}, timing)
    assert out["verified_faster_timed"] == 2
    assert out["also_faster_by_wall_clock"] == 1  # only FAST beats O3 by wall-clock
    assert abs(out["fraction_confirmed"] - 0.5) < 1e-9
    # mca speedups (10, 1.2) rank-agree with wall speedups (50/5=10, 50/80<1) -> +1.0
    assert abs(out["spearman_mca_vs_wall_speedup"] - 1.0) < 1e-9
