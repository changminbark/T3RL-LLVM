# Phase 1 Part A — Verifier & Corpus Feasibility — Implementation Plan

> Terminology: **N** = rewrites sampled per function (the pool), **K** = selection budget (K ≤ N). Note: code snippets below keep the source's local param names `n`/`k`. See [../README.md](../README.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Part A's four deliverables — the `alive-harness` Alive2 wrapper CLI, a scaled per-function corpus builder, the verdict-rate experiment driver, and a perf sanity-check — so they satisfy Part B's frozen seams and are fully unit-tested offline (no LLVM/Alive2 installed this session).

**Architecture:** Part A is **additive** to the existing `src/probe/` package. It provides the `alive-harness` CLI that Part B's `AliveCliVerifier` already shells out to, extends the existing `build_corpus.py` with per-function `llvm-extract`, and adds two experiment drivers (`verify_corpus.py`, `perf_sanity.py`). Every component degrades gracefully when its external tool is absent, so all logic is pure-function-testable now.

**Tech Stack:** Python 3.12, `uv`, `pydantic` v2, `pytest`. Reuses `probe.schema` (`CorpusRecord`, `Verdict`, `VerdictStatus`, `PerfScore`), `probe.verifier.AliveCliVerifier`, `probe.perf.McaPerf`, and helpers already in `probe.build_corpus`.

## Global Constraints

- Python `>=3.12`; run everything via `uv run …` (e.g. `uv run pytest`, `uv run python -m probe.verify_corpus`).
- **Do NOT modify** `src/probe/schema.py`, `McaPerf` in `src/probe/perf.py`, or any Part B probe-pipeline behavior. Part A only adds/extends.
- The `alive-harness` CLI contract is frozen by `AliveCliVerifier`: it is invoked as `alive-harness <src.ll> <tgt.ll> --timeout <s>` and must print the `Verdict` JSON as the **last line of stdout** (`{"status": ..., "counterexample": ...|null, "wall_time_s": ...}`).
- A missing external tool (`alive-tv`, `llvm-extract`, `llc`, `llvm-mca`) is a logged, structured degradation — **never an exception**.
- New tests live under `tests/` and run with no LLVM/Alive2 present. Existing `uv run pytest tests/` must stay green.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit.

---

### Task 1: `alive-harness` CLI — output classifier + subprocess shell + entry point

**Files:**
- Create: `src/probe/alive_harness.py`
- Modify: `pyproject.toml` (add `[project.scripts]` entry point)
- Test: `tests/test_alive_harness.py`

**Interfaces:**
- Consumes: `probe.schema.Verdict`, `probe.schema.VerdictStatus`.
- Produces:
  - `classify_alive_output(stdout: str, stderr: str, returncode: int, timed_out: bool) -> Verdict` — pure function, `wall_time_s` left at `0.0` (the caller fills it).
  - `main(argv: list[str] | None = None) -> int` — CLI entry point; prints one line of `Verdict` JSON to stdout, returns process exit code (always `0`; the verdict carries the status).

- [ ] **Step 1: Write the failing tests for the classifier**

```python
# tests/test_alive_harness.py
from probe.alive_harness import classify_alive_output
from probe.schema import VerdictStatus

VERIFIED = """
----------------------------------------
define i32 @f(i32 %x) {
  ret i32 %x
}
=>
define i32 @f(i32 %x) {
  ret i32 %x
}

Transformation seems to be correct!

Summary:
  1 correct transformations
  0 incorrect transformations
  0 failed-to-prove transformations
  0 Alive2 errors
"""

COUNTEREXAMPLE = """
Transformation doesn't verify!
ERROR: Value mismatch

Example:
i32 %x = #x00000001 (1)

Source value: #x00000001 (1)
Target value: #x00000000 (0)

Summary:
  0 correct transformations
  1 incorrect transformations
"""

UNSUPPORTED = """
ERROR: Unsupported instruction: call
Summary:
  0 correct transformations
  0 incorrect transformations
  1 Alive2 errors
"""

ALIVE_TIMEOUT = """
ERROR: SMT Error: timeout
"""


def test_verified():
    v = classify_alive_output(VERIFIED, "", 0, False)
    assert v.status is VerdictStatus.verified


def test_counterexample_captures_example():
    v = classify_alive_output(COUNTEREXAMPLE, "", 0, False)
    assert v.status is VerdictStatus.counterexample
    assert "Target value" in (v.counterexample or "")


def test_unsupported():
    v = classify_alive_output(UNSUPPORTED, "", 0, False)
    assert v.status is VerdictStatus.unsupported


def test_wall_timeout_wins():
    # Our own wall-clock timeout tripped -> timeout regardless of stdout.
    v = classify_alive_output("", "", -9, True)
    assert v.status is VerdictStatus.timeout


def test_alive_reported_timeout():
    v = classify_alive_output(ALIVE_TIMEOUT, "", 1, False)
    assert v.status is VerdictStatus.timeout


def test_garbage_is_error():
    v = classify_alive_output("random noise", "boom", 2, False)
    assert v.status is VerdictStatus.error
    assert "boom" in (v.counterexample or "")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_alive_harness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'probe.alive_harness'`.

- [ ] **Step 3: Implement `alive_harness.py`**

```python
# src/probe/alive_harness.py
"""`alive-harness` CLI — the Alive2 wrapper Part B's AliveCliVerifier shells out to.

Invoked as `alive-harness <src.ll> <tgt.ll> --timeout <s>`; prints one line of `Verdict`
JSON (schema.Verdict) to stdout. `classify_alive_output` is a pure function over alive-tv's
text output so parsing is unit-tested without Alive2 installed. A missing/failed alive-tv
yields a `Verdict(error)`, never a crash.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .schema import Verdict, VerdictStatus

_UNSUPPORTED_MARKERS = ("unsupported",)
_TIMEOUT_MARKERS = ("smt error: timeout", "timed out", "timeout")
_INCORRECT_MARKERS = ("doesn't verify", "value mismatch", "incorrect transformation")
_CORRECT_MARKERS = ("seems to be correct",)


def _capture_counterexample(text: str, limit: int = 2000) -> str:
    """Grab the informative slice of a failing run (the Example / mismatch block)."""
    lowered = text.lower()
    for anchor in ("example:", "error:"):
        idx = lowered.find(anchor)
        if idx != -1:
            return text[idx : idx + limit].strip()
    return text[:limit].strip()


def classify_alive_output(
    stdout: str, stderr: str, returncode: int, timed_out: bool
) -> Verdict:
    """Map alive-tv's text output to a Verdict. Precedence is deliberate and ordered."""
    if timed_out:
        return Verdict(status=VerdictStatus.timeout)

    text = f"{stdout}\n{stderr}"
    low = text.lower()

    if any(m in low for m in _UNSUPPORTED_MARKERS):
        return Verdict(status=VerdictStatus.unsupported)
    if any(m in low for m in _TIMEOUT_MARKERS):
        return Verdict(status=VerdictStatus.timeout)
    if any(m in low for m in _INCORRECT_MARKERS):
        return Verdict(
            status=VerdictStatus.counterexample,
            counterexample=_capture_counterexample(text),
        )
    if any(m in low for m in _CORRECT_MARKERS):
        return Verdict(status=VerdictStatus.verified)
    return Verdict(
        status=VerdictStatus.error,
        counterexample=(stderr or stdout)[:2000].strip() or None,
    )


def _run_alive(alive_tv: str, src: Path, tgt: Path, timeout_s: int) -> Verdict:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [alive_tv, str(src), str(tgt)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        v = classify_alive_output("", "", -1, True)
        v.wall_time_s = time.perf_counter() - start
        return v
    except OSError as e:
        return Verdict(
            status=VerdictStatus.error,
            counterexample=str(e),
            wall_time_s=time.perf_counter() - start,
        )
    v = classify_alive_output(proc.stdout, proc.stderr, proc.returncode, False)
    v.wall_time_s = time.perf_counter() - start
    return v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Alive2 equivalence-check wrapper -> Verdict JSON")
    p.add_argument("src", help="source .ll file")
    p.add_argument("tgt", help="target .ll file")
    p.add_argument("--timeout", type=int, default=30, help="hard wall timeout (s)")
    p.add_argument(
        "--alive-tv",
        default=os.environ.get("ALIVE_TV") or shutil.which("alive-tv"),
        help="path to alive-tv (default: $ALIVE_TV or PATH)",
    )
    args = p.parse_args(argv)

    if not args.alive_tv:
        print(
            Verdict(
                status=VerdictStatus.error, counterexample="alive-tv not found"
            ).model_dump_json()
        )
        return 0

    verdict = _run_alive(args.alive_tv, Path(args.src), Path(args.tgt), args.timeout)
    print(verdict.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the classifier tests to verify they pass**

Run: `uv run pytest tests/test_alive_harness.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Add the console-script entry point**

In `pyproject.toml`, add after the `[project.optional-dependencies]` block:

```toml
[project.scripts]
alive-harness = "probe.alive_harness:main"
```

- [ ] **Step 6: Add the A↔B round-trip test (fake alive-tv on PATH)**

This proves the CLI mates with B's `AliveCliVerifier` end-to-end, using a tiny fake `alive-tv`.

```python
# append to tests/test_alive_harness.py
import os
import stat
import subprocess
import sys
from pathlib import Path

from probe.verifier import AliveCliVerifier
from probe.schema import VerdictStatus


def _write_fake_alive_tv(dir_path: Path, body: str) -> Path:
    """A fake alive-tv that ignores its args and prints `body` to stdout."""
    script = dir_path / "alive-tv"
    script.write_text(f"#!/usr/bin/env python3\nimport sys\nsys.stdout.write({body!r})\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def test_roundtrip_through_alivecliverifier(tmp_path, monkeypatch):
    # Fake alive-tv that always reports a correct transformation.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_fake_alive_tv(bindir, "Transformation seems to be correct!\n")
    monkeypatch.setenv("ALIVE_TV", str(bindir / "alive-tv"))

    # Invoke the real alive-harness CLI via a wrapper script AliveCliVerifier can exec.
    harness = tmp_path / "alive-harness"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        f'exec {sys.executable} -m probe.alive_harness "$@"\n'
    )
    harness.chmod(harness.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv(
        "PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src")
    )

    verifier = AliveCliVerifier(cli_cmd=str(harness))
    verdict = verifier.check("define i32 @f() {\n ret i32 0\n}", "define i32 @f() {\n ret i32 0\n}")
    assert verdict.status is VerdictStatus.verified
```

- [ ] **Step 7: Run the round-trip test**

Run: `uv run pytest tests/test_alive_harness.py::test_roundtrip_through_alivecliverifier -v`
Expected: PASS. (If `uv run` doesn't expose `src` on `PYTHONPATH` for the subprocess, the test sets it explicitly; confirm PASS.)

- [ ] **Step 8: Run the full suite and commit**

Run: `uv run pytest tests/ -v`
Expected: all green (Part B's tests + the new ones).

```bash
git add src/probe/alive_harness.py pyproject.toml tests/test_alive_harness.py
git commit -m "feat(partA): alive-harness CLI wrapping alive-tv into Verdict JSON

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Scaled corpus builder — per-function extraction, dedup, scale controls

**Files:**
- Modify: `src/probe/build_corpus.py`
- Test: `tests/test_build_corpus.py` (extend existing)

**Interfaces:**
- Consumes: existing `probe.build_corpus._clang_ir`, `count_instructions`, `has_loops`, `_DEFINE_RE`; `probe.schema.CorpusRecord`; `probe.perf.McaPerf`.
- Produces:
  - `list_defined_functions(ir: str) -> list[str]` — names of functions with a body in a module.
  - `extract_function(module_ir: str, func_name: str) -> str | None` — single-function module via `llvm-extract`, or `None` if the tool is absent/fails.
  - `_norm_hash(ir: str) -> str` — dedup key over comment/whitespace-normalized IR.
  - `build_records(src_dir: Path, mca: McaPerf | None, max_functions: int | None = None) -> list[CorpusRecord]` — the scaled builder.
  - `bucket_histogram(records: list[CorpusRecord]) -> dict[tuple[str, bool], int]`.

- [ ] **Step 1: Write failing tests for the new pure helpers**

```python
# append to tests/test_build_corpus.py
from probe.build_corpus import list_defined_functions, _norm_hash, bucket_histogram
from probe.schema import CorpusRecord

TWO_FUNCS = """define i32 @a(i32 %x) {
  ret i32 %x
}

define i32 @b(i32 %y) {
  %r = add i32 %y, 1
  ret i32 %r
}

declare i32 @ext(i32)
"""


def test_list_defined_functions_ignores_declares():
    assert list_defined_functions(TWO_FUNCS) == ["a", "b"]


def test_norm_hash_ignores_comments_and_whitespace():
    a = "define i32 @f() {\n  ret i32 0  ; note\n}"
    b = "define i32 @f() {\n\n    ret i32 0\n}"
    assert _norm_hash(a) == _norm_hash(b)


def test_norm_hash_distinguishes_bodies():
    a = "define i32 @f() {\n  ret i32 0\n}"
    b = "define i32 @f() {\n  ret i32 1\n}"
    assert _norm_hash(a) != _norm_hash(b)


def test_bucket_histogram_counts_by_size_and_loop():
    recs = [
        CorpusRecord(function_id="x", src_ir="", n_instructions=5, has_loops=False),
        CorpusRecord(function_id="y", src_ir="", n_instructions=10, has_loops=False),
        CorpusRecord(function_id="z", src_ir="", n_instructions=30, has_loops=True),
    ]
    hist = bucket_histogram(recs)
    assert hist[("<=20", False)] == 2
    assert hist[("20-50", True)] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_build_corpus.py -v`
Expected: FAIL — `ImportError: cannot import name 'list_defined_functions'`.

- [ ] **Step 3: Add the new helpers to `build_corpus.py`**

Add these imports near the top (alongside the existing `import re`, `import subprocess`):

```python
import hashlib
from collections import Counter
```

Add after the existing module-level regexes / `_CLANG` definition:

```python
_LLVM_EXTRACT = shutil.which("llvm-extract")


def list_defined_functions(ir: str) -> list[str]:
    """Names of all functions with a body (`define`) in a module, in source order."""
    return _DEFINE_RE.findall(ir)


def extract_function(module_ir: str, func_name: str) -> str | None:
    """Pull one function into its own module via llvm-extract. None if tool absent/fails."""
    if _LLVM_EXTRACT is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        mod = Path(td) / "mod.ll"
        mod.write_text(module_ir)
        try:
            proc = subprocess.run(
                [_LLVM_EXTRACT, f"-func={func_name}", "-S", "-o", "-", str(mod)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
    return proc.stdout if proc.returncode == 0 else None


def _norm_hash(ir: str) -> str:
    """Dedup key: strip comments, collapse whitespace, hash. Same normalization idea as the stub verifier."""
    lines = []
    for line in ir.splitlines():
        line = re.sub(r";.*$", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()


def bucket_histogram(records: list[CorpusRecord]) -> dict[tuple[str, bool], int]:
    """Counts per (size_bucket, has_loops) for the end-of-run summary."""
    return dict(Counter((r.size_bucket(), r.has_loops) for r in records))
```

This needs `tempfile` and `Path` (already imported) — confirm `import tempfile` is present at the top; if not, add it alongside the other stdlib imports.

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `uv run pytest tests/test_build_corpus.py -v`
Expected: PASS (existing 4 + new 4).

- [ ] **Step 5: Add `build_records` and rewire `main`**

Add the scaled builder:

```python
def build_records(
    src_dir: Path, mca: "McaPerf | None", max_functions: int | None = None
) -> list[CorpusRecord]:
    """Walk .c files recursively; compile each -O0/-O3; split every defined function.

    function_id is `<file-stem>.<func>` (unique across files). Deduped by normalized -O0 IR.
    Stops once `max_functions` records are collected.
    """
    records: list[CorpusRecord] = []
    seen: set[str] = set()
    for c_path in sorted(src_dir.rglob("*.c")):
        o0_mod = _clang_ir(c_path, "-O0")
        if o0_mod is None:
            continue
        o3_mod = _clang_ir(c_path, "-O3")
        for name in list_defined_functions(o0_mod):
            src_ir = extract_function(o0_mod, name)
            if not src_ir:  # llvm-extract absent or function empty
                continue
            h = _norm_hash(src_ir)
            if h in seen:
                continue
            seen.add(h)
            o3_ir = extract_function(o3_mod, name) if o3_mod else None
            mca_cycles = None
            if mca is not None and o3_ir:
                score = mca.score(o3_ir)
                mca_cycles = score.mca_cycles if score else None
            records.append(
                CorpusRecord(
                    function_id=f"{c_path.stem}.{name}",
                    src_ir=src_ir.strip(),
                    source_lang="c",
                    n_instructions=count_instructions(src_ir),
                    has_loops=has_loops(src_ir),
                    o3_baseline_ir=o3_ir.strip() if o3_ir else None,
                    mca_cycles_o3=mca_cycles,
                )
            )
            if max_functions is not None and len(records) >= max_functions:
                return records
    return records
```

Rewire `main` to prefer per-function extraction, with a `--max-functions` flag and a histogram. Replace the body of `main()` from the `mca = ...` line onward with:

```python
    mca = McaPerf() if args.with_mca else None
    src_dir = Path(args.src)

    if _LLVM_EXTRACT is not None:
        records = build_records(src_dir, mca, args.max_functions)
    else:
        # Fallback: no llvm-extract -> one record per whole file (original behavior).
        print("llvm-extract not found; falling back to whole-file records")
        records = []
        for c_path in sorted(src_dir.rglob("*.c")):
            rec = build_record(c_path, mca)
            if rec:
                records.append(rec)
            if args.max_functions and len(records) >= args.max_functions:
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(r.model_dump_json() for r in records) + "\n")
    print(f"wrote {len(records)} records to {out}")
    for (bucket, loops), n in sorted(bucket_histogram(records).items()):
        print(f"  {bucket:>7}  loops={str(loops):5}  {n}")
```

And add the flag to the parser (next to the existing `--with-mca` line):

```python
    p.add_argument("--max-functions", type=int, default=None, help="cap total records")
```

Note: `main`'s existing `--src`/`--out`/`--with-mca` args and the `if _CLANG is None` guard stay unchanged; only the section shown above is replaced. The existing `build_record` (whole-file) function stays for the fallback path.

- [ ] **Step 6: Run the full builder test file**

Run: `uv run pytest tests/test_build_corpus.py -v`
Expected: PASS (all helper + existing tests).

- [ ] **Step 7: Commit**

```bash
git add src/probe/build_corpus.py tests/test_build_corpus.py
git commit -m "feat(partA): per-function corpus builder via llvm-extract + dedup/histogram

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Verdict-rate experiment driver — `verify_corpus.py`

**Files:**
- Create: `src/probe/verify_corpus.py`
- Test: `tests/test_verify_corpus.py`

**Interfaces:**
- Consumes: `probe.run_probe.load_corpus`, `probe.verifier.AliveCliVerifier`, `probe.schema.CorpusRecord`, `probe.schema.Verdict`, `probe.schema.VerdictStatus`.
- Produces:
  - `aggregate(records: list[CorpusRecord], verdicts: dict[str, Verdict]) -> dict` — per `(size_bucket, has_loops)` stats: `n_checked`, `n_skipped`, `verified_rate`, per-status counts, `median_wall_s`, `p90_wall_s`.
  - `format_table(table: dict) -> str` — the one-page text table.
  - `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write failing tests for `aggregate`**

```python
# tests/test_verify_corpus.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_verify_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'probe.verify_corpus'`.

- [ ] **Step 3: Implement `verify_corpus.py`**

```python
# src/probe/verify_corpus.py
"""Part A headline experiment: Alive2 verdict rates on the compiler's own -O0 -> -O3.

For every corpus record with an O3 baseline, ask the alive-harness whether (src_ir, o3_baseline_ir)
verify. Aggregate verdict rate + latency per (size bucket, loop presence) and emit the one-page
table. Uses the *same* AliveCliVerifier Part B uses, so this also exercises the harness end-to-end.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tqdm import tqdm

from .run_probe import load_corpus
from .schema import CorpusRecord, Verdict, VerdictStatus
from .verifier import AliveCliVerifier

_STATUSES = [s.value for s in VerdictStatus]


def _key(rec: CorpusRecord) -> str:
    return f"{rec.size_bucket()}|loops={rec.has_loops}"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct * (len(ordered) - 1))))
    return ordered[idx]


def aggregate(records: list[CorpusRecord], verdicts: dict[str, Verdict]) -> dict:
    """Per (size bucket, loop) stats. Records without a checked verdict count as skipped."""
    buckets: dict[str, dict] = {}
    for rec in records:
        cell = buckets.setdefault(
            _key(rec),
            {
                "n_checked": 0,
                "n_skipped": 0,
                "status_counts": {s: 0 for s in _STATUSES},
                "_walls": [],
            },
        )
        v = verdicts.get(rec.function_id)
        if v is None:
            cell["n_skipped"] += 1
            continue
        cell["n_checked"] += 1
        cell["status_counts"][v.status.value] += 1
        cell["_walls"].append(v.wall_time_s)

    for cell in buckets.values():
        walls = cell.pop("_walls")
        n = cell["n_checked"]
        cell["verified_rate"] = (cell["status_counts"]["verified"] / n) if n else 0.0
        cell["median_wall_s"] = statistics.median(walls) if walls else 0.0
        cell["p90_wall_s"] = _percentile(walls, 0.9)
    return buckets


def format_table(table: dict) -> str:
    header = (
        f"{'bucket|loops':<22}{'n_chk':>6}{'n_skip':>7}"
        f"{'ver_rate':>9}{'med_s':>8}{'p90_s':>8}  status_counts"
    )
    lines = [header, "-" * len(header)]
    for key in sorted(table):
        c = table[key]
        nz = {s: n for s, n in c["status_counts"].items() if n}
        lines.append(
            f"{key:<22}{c['n_checked']:>6}{c['n_skipped']:>7}"
            f"{c['verified_rate']:>9.2f}{c['median_wall_s']:>8.2f}"
            f"{c['p90_wall_s']:>8.2f}  {nz}"
        )
    return "\n".join(lines)


def run(args) -> None:
    records = list(load_corpus(Path(args.corpus)).values())
    verifier = AliveCliVerifier(cli_cmd=args.cli)
    verdicts: dict[str, Verdict] = {}
    for rec in tqdm(records, desc="verify O0->O3"):
        if not rec.o3_baseline_ir:
            continue
        verdicts[rec.function_id] = verifier.check(
            rec.src_ir, rec.o3_baseline_ir, timeout_s=args.timeout
        )

    table = aggregate(records, verdicts)
    text = format_table(table)
    print("\n" + text + "\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict_rates.json").write_text(
        json.dumps({"timeout_s": args.timeout, "table": table}, indent=2)
    )
    (out / "verdict_rates.txt").write_text(text + "\n")
    print(f"wrote {out}/verdict_rates.json and .txt")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Part A: Alive2 verdict-rate experiment")
    p.add_argument("--corpus", required=True, help="JSONL file or dir of *.jsonl")
    p.add_argument("--timeout", type=int, default=30, help="per-check timeout (s)")
    p.add_argument("--cli", default="alive-harness", help="alive-harness command")
    p.add_argument("--out", default="results", help="output dir")
    return p


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_verify_corpus.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/probe/verify_corpus.py tests/test_verify_corpus.py
git commit -m "feat(partA): verdict-rate experiment driver (Alive2 on O0->O3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Perf sanity-check — `perf_sanity.py`

**Files:**
- Create: `src/probe/perf_sanity.py`
- Test: `tests/test_perf_sanity.py`

**Interfaces:**
- Consumes: `probe.perf.PerfScorer`, `probe.perf.make_perf`, `probe.schema.CorpusRecord`, `probe.schema.PerfScore`, `probe.run_probe.load_corpus`.
- Produces:
  - `check_monotonic(records: list[CorpusRecord], perf: PerfScorer) -> dict` — `{checked, held, fraction, inversions}` where `inversions` is a list of `(function_id, o0_cycles, o3_cycles)` for records where `cycles(O0) < cycles(O3)`.
  - `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write failing tests for `check_monotonic`**

```python
# tests/test_perf_sanity.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_perf_sanity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'probe.perf_sanity'`.

- [ ] **Step 3: Implement `perf_sanity.py`**

```python
# src/probe/perf_sanity.py
"""Sanity-check the perf scorer: it must rank -O0 cycles >= -O3 cycles.

If llvm-mca ranks optimized code as *slower*, the speed signal is untrustworthy and
`verified_faster` rewards are noise. Reports the fraction of functions where the ranking
holds and lists any inversions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .perf import PerfScorer, make_perf
from .run_probe import load_corpus
from .schema import CorpusRecord


def check_monotonic(records: list[CorpusRecord], perf: PerfScorer) -> dict:
    checked = 0
    held = 0
    inversions: list[tuple[str, float, float]] = []
    for rec in records:
        if not rec.o3_baseline_ir:
            continue
        s0 = perf.score(rec.src_ir)
        s3 = perf.score(rec.o3_baseline_ir)
        if s0 is None or s3 is None:
            continue
        checked += 1
        if s0.mca_cycles >= s3.mca_cycles:
            held += 1
        else:
            inversions.append((rec.function_id, s0.mca_cycles, s3.mca_cycles))
    return {
        "checked": checked,
        "held": held,
        "fraction": (held / checked) if checked else None,
        "inversions": inversions,
    }


def run(args) -> None:
    records = list(load_corpus(Path(args.corpus)).values())
    perf = make_perf(args.perf)
    out = check_monotonic(records, perf)
    if out["checked"] == 0:
        print("no measurable (O0, O3) pairs — is llvm-mca/llc installed and are O3 baselines present?")
    else:
        print(
            f"monotonic (O0>=O3) on {out['held']}/{out['checked']} = {out['fraction']:.2f}"
        )
        for fid, c0, c3 in out["inversions"][:20]:
            print(f"  INVERSION {fid}: O0={c0} < O3={c3}")
    print(json.dumps(out, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Part A: perf scorer monotonicity sanity-check")
    p.add_argument("--corpus", required=True, help="JSONL file or dir of *.jsonl")
    p.add_argument("--perf", default="mca", choices=["stub", "mca"])
    return p


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_perf_sanity.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/probe/perf_sanity.py tests/test_perf_sanity.py
git commit -m "feat(partA): perf scorer monotonicity sanity-check

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Full-suite verification + sync-doc note

**Files:**
- Modify: `../partB-plan.md` (append a short Part A confirmation note) — OR add it to `../partA-plan.md`; pick the shared sync doc both sessions read.

**Interfaces:** none (documentation + verification only).

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL green — Part B's original tests plus the four new test files. If anything in Part B went red, Part A violated the "additive only" constraint; fix before proceeding.

- [ ] **Step 2: Confirm the CLIs are importable and wired**

Run:
```bash
uv run python -m probe.alive_harness --help
uv run python -m probe.verify_corpus --help
uv run python -m probe.perf_sanity --help
uv sync && uv run alive-harness --help
```
Expected: each prints usage; `uv run alive-harness --help` confirms the console-script entry point resolves (so `shutil.which("alive-harness")` will work for `AliveCliVerifier` after install).

- [ ] **Step 3: Append the interface-confirmation note to the shared sync doc**

Add to the bottom of `../partB-plan.md` (the doc both sessions read), under a new heading:

```markdown
## Part A status (Person A confirmation)

- `alive-harness` CLI implemented (`src/probe/alive_harness.py`, console-script entry point).
  Contract matches `AliveCliVerifier`: `alive-harness <src.ll> <tgt.ll> --timeout <s>` ->
  last-line `Verdict` JSON. Missing `alive-tv` -> `Verdict(error)`, never crashes.
- Corpus builder scaled to per-function extraction (`build_corpus.build_records`, `--max-functions`),
  deduped, bucketed. `function_id` is `<file-stem>.<func>`.
- Verdict-rate experiment: `python -m probe.verify_corpus --corpus <dir> --timeout 30`
  -> `results/verdict_rates.{json,txt}` (the go/no-go table).
- Perf sanity-check: `python -m probe.perf_sanity --corpus <dir> --perf mca`.
- All Part A logic unit-tested offline (no LLVM/Alive2 needed). Real corpus + real verdict table
  pending a later session that installs LLVM + builds Alive2.
```

- [ ] **Step 4: Commit**

```bash
git add ../partB-plan.md
git commit -m "docs(partA): confirm alive-harness contract + Part A status in sync doc

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** Task 1 = component A (alive-harness) + A↔B round-trip test; Task 2 = component B (scaled corpus builder); Task 3 = component C (verdict-rate driver); Task 4 = component D (perf sanity); Task 5 = the offline test-suite gate + sync-doc deliverable. All four spec deliverables + the "B suite stays green" constraint are covered.
- **No new tools installed** — every subprocess path (`alive-tv`, `llvm-extract`, `llc`, `llvm-mca`) has a tool-absent branch, so all tests pass on this machine.
- **Type consistency:** `Verdict`/`VerdictStatus`/`CorpusRecord`/`PerfScore` come straight from `probe.schema` unchanged; `AliveCliVerifier.check(src_ir, tgt_ir, timeout_s=...)` and `load_corpus(path) -> dict[str, CorpusRecord]` match Part B's existing signatures.
```
