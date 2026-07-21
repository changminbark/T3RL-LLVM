import json
from pathlib import Path

from probe.phase2_baseline import load_rewrites, format_curve, main
from probe.schema import RewriteResult, RewriteOutcome, CorpusRecord


def test_load_rewrites_groups_by_function(tmp_path):
    p = tmp_path / "a.rewrites.jsonl"
    rows = [
        RewriteResult(function_id="f1", sample_index=0, outcome=RewriteOutcome.verified_faster, speedup_vs_o3=1.5),
        RewriteResult(function_id="f1", sample_index=1, outcome=RewriteOutcome.invalid_syntax),
        RewriteResult(function_id="f2", sample_index=0, outcome=RewriteOutcome.verified_no_gain),
    ]
    p.write_text("\n".join(r.model_dump_json() for r in rows) + "\n")
    grouped = load_rewrites(p)
    assert set(grouped) == {"f1", "f2"}
    assert len(grouped["f1"]) == 2


def test_format_curve_has_headers():
    result = {"overall": {1: {"coverage": 0.25, "mean_speedup": 1.25, "n_functions": 2}},
              "by_bucket": {}}
    text = format_curve(result)
    assert "K" in text and "coverage" in text.lower() and "speedup" in text.lower()


def test_main_end_to_end(tmp_path):
    # rewrites
    rw = tmp_path / "run"
    rw.mkdir()
    rows = [
        RewriteResult(function_id="f1", sample_index=0, outcome=RewriteOutcome.verified_faster, speedup_vs_o3=2.0),
        RewriteResult(function_id="f1", sample_index=1, outcome=RewriteOutcome.verified_no_gain),
    ]
    (rw / "x.rewrites.jsonl").write_text("\n".join(r.model_dump_json() for r in rows) + "\n")
    # corpus (for buckets)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(CorpusRecord(function_id="f1", src_ir="x", n_instructions=5, has_loops=False).model_dump_json() + "\n")
    out = tmp_path / "out"
    rc = main(["--rewrites", str(rw), "--corpus", str(corpus), "--ks", "1,2", "--out", str(out)])
    assert rc == 0
    data = json.loads((out / "phase2_baseline.json").read_text())
    assert data["curve"]["overall"]["1"]["n_functions"] == 1
    assert (out / "phase2_baseline.txt").exists()
