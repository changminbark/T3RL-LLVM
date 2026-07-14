from probe.perf_sanity import check_monotonic
from probe.perf import PerfScorer
from probe.schema import CorpusRecord, PerfScore


class FakePerf(PerfScorer):
    """Scores by mapping IR text to a preset cycle count."""

    def __init__(self, table):
        self.table = table

    def score(self, ir):
        c = self.table.get(ir)
        return PerfScore(mca_cycles=float(c)) if c is not None else None


def _rec(fid, src, o3):
    return CorpusRecord(function_id=fid, src_ir=src, o3_baseline_ir=o3)


def test_monotonic_all_hold():
    recs = [_rec("a", "o0a", "o3a"), _rec("b", "o0b", "o3b")]
    perf = FakePerf({"o0a": 10, "o3a": 4, "o0b": 8, "o3b": 8})  # >= holds (ties ok)
    out = check_monotonic(recs, perf)
    assert out["checked"] == 2
    assert out["held"] == 2
    assert out["fraction"] == 1.0
    assert out["inversions"] == []


def test_monotonic_flags_inversion():
    recs = [_rec("bad", "o0", "o3")]
    perf = FakePerf({"o0": 3, "o3": 9})  # O0 faster than O3 -> inversion
    out = check_monotonic(recs, perf)
    assert out["held"] == 0
    assert out["inversions"] == [("bad", 3.0, 9.0)]


def test_unmeasurable_records_skipped():
    recs = [_rec("x", "o0", "o3")]
    perf = FakePerf({})  # scorer returns None for everything
    out = check_monotonic(recs, perf)
    assert out["checked"] == 0
    assert out["fraction"] is None
