from probe.verify_corpus import aggregate, format_table
from probe.schema import CorpusRecord, Verdict, VerdictStatus


def _rec(fid, n, loops, has_o3=True):
    return CorpusRecord(
        function_id=fid,
        src_ir="x",
        n_instructions=n,
        has_loops=loops,
        o3_baseline_ir="y" if has_o3 else None,
    )


def test_aggregate_rate_and_median():
    records = [_rec("a", 5, False), _rec("b", 6, False), _rec("c", 7, False, has_o3=False)]
    verdicts = {
        "a": Verdict(status=VerdictStatus.verified, wall_time_s=2.0),
        "b": Verdict(status=VerdictStatus.timeout, wall_time_s=4.0),
        # "c" has no O3 baseline -> never checked -> counts as skipped
    }
    table = aggregate(records, verdicts)
    cell = table["<=20|loops=False"]
    assert cell["n_checked"] == 2
    assert cell["n_skipped"] == 1
    assert cell["verified_rate"] == 0.5
    assert cell["status_counts"]["verified"] == 1
    assert cell["status_counts"]["timeout"] == 1
    assert cell["median_wall_s"] == 3.0


def test_format_table_is_stringy():
    table = aggregate([_rec("a", 5, False)], {"a": Verdict(status=VerdictStatus.verified)})
    text = format_table(table)
    assert "<=20" in text
    assert "verified_rate" in text or "rate" in text.lower()
