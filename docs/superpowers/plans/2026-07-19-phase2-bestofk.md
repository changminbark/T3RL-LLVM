# Phase 2 — Best-of-K Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the best-of-K inference-time baseline — a pure selection+aggregation layer over existing `run_probe` rollouts that reports Coverage@K and MeanSpeedup@K over `-O3` as a K-scaling curve (K∈{1,2,4,8,16}), the number TTRL must later beat.

**Architecture:** Additive to `src/probe/`. `bestofk.py` holds pure, unbiased estimators (pass@k coverage, expected-best-over-k-subsets speedup) and a `curve` aggregator. `phase2_baseline.py` is a CLI that loads a run's `rewrites.jsonl`, joins size/loop buckets from the corpus, and emits the curve. No changes to generation/verification/scoring — Phase 2 consumes `RewriteResult` records unchanged.

**Tech Stack:** Python 3.12, `uv`, `pydantic` v2, `pytest`, stdlib `math.comb`. Reuses `probe.schema.RewriteResult`, `probe.schema.CorpusRecord`, `probe.run_probe.load_corpus`.

## Global Constraints

- Python `>=3.12`; run everything via `uv run …` (e.g. `uv run pytest`, `uv run python -m probe.phase2_baseline`).
- **Additive only.** Do NOT modify generation, verification, scoring, or the outcome pipeline (`outcome.py`, `verifier.py`, `perf.py`, `schema.py`). Phase 2 consumes `RewriteResult` records unchanged. Existing `uv run pytest tests/` must stay green.
- A rewrite is **verified** if `outcome ∈ {verified_faster, verified_no_gain}`; it **beats -O3** iff `outcome == verified_faster`.
- **Achieved speedup per sample** = `max(1.0, speedup_vs_o3)` when `outcome == verified_faster`, else `1.0` (the -O3 fallback — a non-beating sample is worth keeping -O3).
- Estimators are unbiased over all `C(n,k)` subsets (NOT "first k"): Coverage@k via pass@k `1 − C(n−c,k)/C(n,k)`; MeanSpeedup@k via expected max over random k-subsets. Use `math.comb`. When `k ≥ n`, both reduce to the observed best over all `n` samples.
- Metrics reported for K∈{1,2,4,8,16}, overall and per `(size_bucket, has_loops)`.
- TDD: failing test first, watch it fail, minimal implementation, watch it pass, commit.

---

### Task 1: Estimators + per-function speedups (`bestofk.py` core)

**Files:**
- Create: `src/probe/bestofk.py`
- Test: `tests/test_bestofk.py`

**Interfaces:**
- Consumes: `probe.schema.RewriteResult`, `probe.schema.RewriteOutcome`.
- Produces:
  - `passk_estimator(n: int, c: int, k: int) -> float`
  - `expected_best_speedup_at_k(speedups: list[float], k: int) -> float`
  - `per_function_speedups(records: list[RewriteResult]) -> list[float]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bestofk.py
import math
import pytest

from probe.bestofk import (
    passk_estimator,
    expected_best_speedup_at_k,
    per_function_speedups,
)
from probe.schema import RewriteResult, RewriteOutcome


def test_passk_known_values():
    assert passk_estimator(8, 0, 1) == 0.0          # no successes -> never covered
    assert passk_estimator(8, 8, 1) == 1.0          # all successes -> always covered
    assert passk_estimator(4, 1, 2) == pytest.approx(0.5)      # 1 - C(3,2)/C(4,2) = 1 - 3/6
    assert passk_estimator(4, 2, 2) == pytest.approx(5 / 6)    # 1 - C(2,2)/C(4,2) = 1 - 1/6


def test_passk_k_ge_available_nonsuccess():
    # If k covers more than the non-successes, coverage is certain.
    assert passk_estimator(4, 2, 3) == 1.0          # any 3 of 4 must include a success


def test_expected_best_speedup_two_samples():
    # speedups [2.0, 1.0]: k=1 -> mean 1.5 ; k=2 -> max 2.0
    assert expected_best_speedup_at_k([2.0, 1.0], 1) == pytest.approx(1.5)
    assert expected_best_speedup_at_k([2.0, 1.0], 2) == pytest.approx(2.0)


def test_expected_best_speedup_k_ge_n():
    assert expected_best_speedup_at_k([1.3, 1.1, 1.0], 5) == pytest.approx(1.3)


def _rec(idx, outcome, speedup):
    return RewriteResult(
        function_id="f", sample_index=idx, outcome=outcome, speedup_vs_o3=speedup
    )


def test_per_function_speedups_mapping():
    recs = [
        _rec(0, RewriteOutcome.verified_faster, 1.5),
        _rec(1, RewriteOutcome.verified_no_gain, None),
        _rec(2, RewriteOutcome.invalid_syntax, None),
        _rec(3, RewriteOutcome.verified_faster, 0.9),  # capped up to 1.0 by max()
    ]
    assert per_function_speedups(recs) == [1.5, 1.0, 1.0, 1.0]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_bestofk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'probe.bestofk'`.

- [ ] **Step 3: Implement the estimators**

```python
# src/probe/bestofk.py
"""Best-of-K selection metrics over existing rollouts (Phase 2 baseline).

Unbiased estimators over all C(n,k) subsets, not "the first k":
  - Coverage@k  = pass@k = 1 - C(n-c, k)/C(n, k), c = #verified_faster among n samples.
  - MeanSpeedup@k = expected max achieved speedup over a random k-subset.
Achieved speedup per sample = max(1.0, speedup_vs_o3) for verified_faster, else 1.0
(the -O3 fallback: a non-beating sample is worth keeping the compiler's -O3).
"""

from __future__ import annotations

import math

from .schema import RewriteOutcome, RewriteResult


def passk_estimator(n: int, c: int, k: int) -> float:
    """P(at least one of c successes appears in a random k-subset of n samples)."""
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if k > n - c:  # too few non-successes to fill k -> a success is guaranteed
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def expected_best_speedup_at_k(speedups: list[float], k: int) -> float:
    """Expected max over a uniformly random k-subset of `speedups`.

    P(sample ranked i (1-based, descending) is the subset max) = C(n-i, k-1) / C(n, k).
    """
    n = len(speedups)
    if n == 0:
        return 1.0
    k = min(k, n)
    ordered = sorted(speedups, reverse=True)
    denom = math.comb(n, k)
    total = 0.0
    for i, s in enumerate(ordered, start=1):  # i = 1..n
        if n - i >= k - 1:
            total += s * math.comb(n - i, k - 1) / denom
    return total


def per_function_speedups(records: list[RewriteResult]) -> list[float]:
    """Achieved speedup for each sample: max(1.0, speedup) if verified_faster, else 1.0."""
    out: list[float] = []
    for r in records:
        if r.outcome is RewriteOutcome.verified_faster and r.speedup_vs_o3 is not None:
            out.append(max(1.0, r.speedup_vs_o3))
        else:
            out.append(1.0)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_bestofk.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/probe/bestofk.py tests/test_bestofk.py
git commit -m "feat(phase2): best-of-K estimators (pass@k coverage + expected-best speedup)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: The `curve` aggregator (`bestofk.py`)

**Files:**
- Modify: `src/probe/bestofk.py`
- Test: `tests/test_bestofk.py` (extend)

**Interfaces:**
- Consumes: `passk_estimator`, `expected_best_speedup_at_k`, `per_function_speedups` (Task 1);
  `probe.schema.CorpusRecord`.
- Produces:
  - `curve(records_by_function: dict[str, list[RewriteResult]], buckets: dict[str, tuple[str, bool]], ks: list[int]) -> dict`
    where `buckets` maps `function_id -> (size_bucket, has_loops)`. Returns
    `{"overall": {k: {coverage, mean_speedup, n_functions}}, "by_bucket": {"<bucket>|loops=<bool>": {k: {...}}}}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_bestofk.py
from probe.bestofk import curve


def _recs_for(fid, outcomes_speedups):
    return [
        _rec_named(fid, i, o, s) for i, (o, s) in enumerate(outcomes_speedups)
    ]


def _rec_named(fid, idx, outcome, speedup):
    return RewriteResult(
        function_id=fid, sample_index=idx, outcome=outcome, speedup_vs_o3=speedup
    )


def test_curve_overall_and_buckets():
    VF = RewriteOutcome.verified_faster
    NG = RewriteOutcome.verified_no_gain
    # f1: 2 samples, one is 2.0x faster ; f2: 2 samples, none beat O3
    by_fn = {
        "f1": _recs_for("f1", [(VF, 2.0), (NG, None)]),
        "f2": _recs_for("f2", [(NG, None), (NG, None)]),
    }
    buckets = {"f1": ("<=20", False), "f2": ("<=20", False)}
    out = curve(by_fn, buckets, ks=[1, 2])

    # Coverage@1: f1 has c=1 of n=2 -> passk(2,1,1)=0.5 ; f2 -> 0. Mean over 2 fns = 0.25
    assert out["overall"][1]["coverage"] == pytest.approx(0.25)
    # Coverage@2: f1 -> 1.0 ; f2 -> 0 ; mean = 0.5
    assert out["overall"][2]["coverage"] == pytest.approx(0.5)
    # MeanSpeedup@1: f1 expected best of 1 over [2.0,1.0] = 1.5 ; f2 = 1.0 ; mean = 1.25
    assert out["overall"][1]["mean_speedup"] == pytest.approx(1.25)
    # MeanSpeedup@2: f1 = 2.0 ; f2 = 1.0 ; mean = 1.5
    assert out["overall"][2]["mean_speedup"] == pytest.approx(1.5)
    assert out["overall"][1]["n_functions"] == 2
    key = "<=20|loops=False"
    assert out["by_bucket"][key][2]["coverage"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_bestofk.py::test_curve_overall_and_buckets -v`
Expected: FAIL — `ImportError: cannot import name 'curve'`.

- [ ] **Step 3: Implement `curve`**

Add to `src/probe/bestofk.py`:

```python
def _agg(records_by_function, fids, ks):
    """Coverage/mean-speedup per k, averaged over the given function ids."""
    out = {}
    n_fns = len(fids)
    for k in ks:
        cov = 0.0
        spd = 0.0
        for fid in fids:
            recs = records_by_function[fid]
            n = len(recs)
            c = sum(1 for r in recs if r.outcome is RewriteOutcome.verified_faster)
            cov += passk_estimator(n, c, k)
            spd += expected_best_speedup_at_k(per_function_speedups(recs), k)
        out[k] = {
            "coverage": cov / n_fns if n_fns else 0.0,
            "mean_speedup": spd / n_fns if n_fns else 1.0,
            "n_functions": n_fns,
        }
    return out


def curve(records_by_function, buckets, ks):
    """Best-of-K curve overall and per (size_bucket, has_loops)."""
    all_fids = list(records_by_function)
    result = {"overall": _agg(records_by_function, all_fids, ks), "by_bucket": {}}

    by_bucket_fids: dict[str, list[str]] = {}
    for fid in all_fids:
        b = buckets.get(fid)
        if b is None:
            continue
        key = f"{b[0]}|loops={b[1]}"
        by_bucket_fids.setdefault(key, []).append(fid)

    for key, fids in sorted(by_bucket_fids.items()):
        result["by_bucket"][key] = _agg(records_by_function, fids, ks)
    return result
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_bestofk.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/probe/bestofk.py tests/test_bestofk.py
git commit -m "feat(phase2): best-of-K curve aggregator (overall + per bucket)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Driver + report (`phase2_baseline.py`)

**Files:**
- Create: `src/probe/phase2_baseline.py`
- Test: `tests/test_phase2_baseline.py`

**Interfaces:**
- Consumes: `probe.bestofk.curve` (Task 2); `probe.run_probe.load_corpus`; `probe.schema.RewriteResult`, `probe.schema.CorpusRecord`.
- Produces:
  - `load_rewrites(path: Path) -> dict[str, list[RewriteResult]]` — group `rewrites.jsonl` records by `function_id`; accepts a file or a run dir (globs `*.rewrites.jsonl`).
  - `format_curve(result: dict) -> str` — the K-curve text table.
  - `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_phase2_baseline.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_phase2_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'probe.phase2_baseline'`.

- [ ] **Step 3: Implement the driver**

```python
# src/probe/phase2_baseline.py
"""Phase 2 best-of-K baseline report.

Loads a run's rewrites.jsonl, joins size/loop buckets from the corpus, and emits the
Coverage@K / MeanSpeedup@K curve (the non-TTRL baseline TTRL must beat).

    uv run python -m probe.phase2_baseline --rewrites results/<run> --corpus data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bestofk import curve
from .run_probe import load_corpus
from .schema import RewriteResult


def load_rewrites(path: Path) -> dict[str, list[RewriteResult]]:
    files = sorted(path.glob("*.rewrites.jsonl")) if path.is_dir() else [path]
    grouped: dict[str, list[RewriteResult]] = {}
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = RewriteResult.model_validate_json(line)
            grouped.setdefault(r.function_id, []).append(r)
    if not grouped:
        raise SystemExit(f"no rewrite records found under {path}")
    return grouped


def format_curve(result: dict) -> str:
    def block(title: str, per_k: dict) -> list[str]:
        lines = [title, f"  {'K':>3}  {'coverage':>9}  {'mean_speedup':>13}  {'n_fns':>6}"]
        for k in sorted(per_k, key=int):
            c = per_k[k]
            lines.append(
                f"  {int(k):>3}  {c['coverage']:>9.3f}  {c['mean_speedup']:>13.3f}  {c['n_functions']:>6}"
            )
        return lines

    lines = block("OVERALL", result["overall"])
    for key in sorted(result["by_bucket"]):
        lines.append("")
        lines += block(key, result["by_bucket"][key])
    return "\n".join(lines)


def run(args) -> None:
    grouped = load_rewrites(Path(args.rewrites))
    records = load_corpus(Path(args.corpus))
    buckets = {fid: (rec.size_bucket(), rec.has_loops) for fid, rec in records.items()}
    ks = [int(x) for x in args.ks.split(",")]

    max_n = max(len(v) for v in grouped.values())
    ks = [k for k in ks if k <= max_n] or [max_n]
    if any(int(x) > max_n for x in args.ks.split(",")):
        print(f"note: capping K to available samples per function (max n={max_n})")

    result = curve(grouped, buckets, ks)
    text = format_curve(result)
    print("\n" + text + "\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase2_baseline.json").write_text(json.dumps({"ks": ks, "curve": result}, indent=2))
    (out / "phase2_baseline.txt").write_text(text + "\n")
    print(f"wrote {out}/phase2_baseline.json and .txt")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 best-of-K baseline")
    p.add_argument("--rewrites", required=True, help="run dir or *.rewrites.jsonl file")
    p.add_argument("--corpus", required=True, help="corpus JSONL (for size/loop buckets)")
    p.add_argument("--ks", default="1,2,4,8,16", help="comma-separated K values")
    p.add_argument("--out", default="results", help="output dir")
    return p


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_phase2_baseline.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/probe/phase2_baseline.py tests/test_phase2_baseline.py
git commit -m "feat(phase2): best-of-K baseline driver + K-curve report

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Run the baseline + findings note

**Files:**
- Create: `docs/phase2-findings.md`

**Interfaces:** none (experiment + documentation).

- [ ] **Step 1: Full suite green**

Run: `uv run pytest tests/ -v`
Expected: all pass (Phase 1 + Phase 2 tests). If any Phase 1 test went red, Phase 2 violated additive-only — fix first.

- [ ] **Step 2: Immediate K=1–8 curve from existing deepseek data**

Run:
```bash
uv run python -m probe.phase2_baseline \
  --rewrites results/probe_llm_deepseek --corpus /tmp/corpus21.jsonl --ks 1,2,4,8
```
Expected: prints the OVERALL + per-bucket K-curve (coverage rising with K, mean_speedup ≥ 1.0), writes `results/phase2_baseline.{json,txt}`.

- [ ] **Step 3: Firm K=16 deepseek run (full curve)**

Run (needs `source scripts/alive2/env.sh` for ALIVE_TV + LLVM_BIN, and `FIREWORKS_API_KEY` set):
```bash
source scripts/alive2/env.sh
export FIREWORKS_API_KEY=<key>
uv run python -m probe.run_probe --corpus /tmp/corpus21.jsonl \
  --backend api --model "accounts/fireworks/models/deepseek-v4-pro" \
  --base-url https://api.fireworks.ai/inference/v1 --api-key-env FIREWORKS_API_KEY \
  --format ir --k 16 --temperature 0.9 --max-tokens 8000 \
  --verifier alive --perf mca --out results/probe_llm_deepseek_k16
uv run python -m probe.phase2_baseline \
  --rewrites results/probe_llm_deepseek_k16 --corpus /tmp/corpus21.jsonl --ks 1,2,4,8,16
```
Expected: full K=1–16 curve. (This step is a real API run; if a key is unavailable, the K=1–8 curve from Step 2 stands as the interim baseline.)

- [ ] **Step 4: Write `docs/phase2-findings.md`**

Capture: the K-curve table (coverage@K, mean_speedup@K overall + per bucket); the **with-LLVM vs without** reading (best-of-1 oracle-verified vs best-of-K lift; and that without the oracle there is no *trustworthy* output); the baseline number TTRL must beat; and the standing caveats (small/trivial corpus, mca-as-proxy, loop-free scope). Use the actual numbers from Steps 2–3.

- [ ] **Step 5: Commit**

```bash
git add docs/phase2-findings.md
git commit -m "docs(phase2): best-of-K baseline findings + the number TTRL must beat

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** Task 1 = estimators + per-sample speedups; Task 2 = `curve` (overall + bucket); Task 3 = driver/report + K-capping + JSON/txt output; Task 4 = the runs (immediate K≤8 + firm K=16) and findings. All spec components, metric definitions, and the estimator requirement are covered.
- **Additive-only:** only new files (`bestofk.py`, `phase2_baseline.py`) + their tests + a findings doc; no edits to the outcome/verify/score pipeline.
- **Type consistency:** `RewriteResult` fields (`function_id`, `sample_index`, `outcome`, `speedup_vs_o3`) and `RewriteOutcome.verified_faster/verified_no_gain` match `schema.py`; `load_corpus(path) -> dict[str, CorpusRecord]` and `CorpusRecord.size_bucket()`/`.has_loops` match existing code; `curve(records_by_function, buckets, ks)` signature is identical in Tasks 2 and 3.
