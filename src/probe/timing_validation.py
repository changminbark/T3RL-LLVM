"""Audit llvm-mca against real wall-clock timing — the Phase 2 hardening.

Two checks (both reuse the existing corpus/rewrites loaders and scorers):

1. Corpus: for every function with an -O3 baseline, compare mca cycles and measured ns for the
   -O0 and -O3 versions. Reports how often each metric ranks -O0 slower than -O3 (they should),
   and the rank correlation between mca cycles and ns.
2. Rewrite audit (--rewrites): for each rewrite mca called `verified_faster` (faster than -O3),
   time it and the -O3 baseline. Reports the fraction that are *actually* faster by wall-clock and
   the correlation between mca-predicted speedup and real speedup — i.e. how much to trust Phase 2's
   mca-based coverage.

    uv run python -m probe.timing_validation --corpus data/corpus [--rewrites results/<run>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .perf import make_perf
from .phase2_baseline import load_rewrites
from .run_probe import load_corpus
from .schema import CorpusRecord, RewriteOutcome, RewriteResult
from .timing import TimingPerf


# ---------- pure-python rank correlation (no scipy) ----------
def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average rank for ties (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


# ---------- checks ----------
def corpus_audit(records: list[CorpusRecord], mca, timing: TimingPerf) -> dict:
    checked = wall_mono = mca_mono = 0
    mca_vals: list[float] = []
    ns_vals: list[float] = []
    for rec in records:
        if not rec.o3_baseline_ir:
            continue
        m0, m3 = mca.score(rec.src_ir), mca.score(rec.o3_baseline_ir)
        t0, t3 = timing.score(rec.src_ir), timing.score(rec.o3_baseline_ir)
        if None in (m0, m3, t0, t3):
            continue
        checked += 1
        wall_mono += t0.wall_ns >= t3.wall_ns
        mca_mono += m0.mca_cycles >= m3.mca_cycles
        mca_vals += [m0.mca_cycles, m3.mca_cycles]
        ns_vals += [t0.wall_ns, t3.wall_ns]
    return {
        "timeable_pairs": checked,
        "wall_clock_monotonic": wall_mono / checked if checked else None,
        "mca_monotonic": mca_mono / checked if checked else None,
        "spearman_mca_vs_ns": spearman(mca_vals, ns_vals),
    }


def rewrite_audit(
    grouped: dict[str, list[RewriteResult]], by_id: dict[str, CorpusRecord], timing: TimingPerf
) -> dict:
    total = faster_by_wall = 0
    mca_speedups: list[float] = []
    wall_speedups: list[float] = []
    for fid, recs in grouped.items():
        rec = by_id.get(fid)
        if not rec or not rec.o3_baseline_ir:
            continue
        t3 = timing.score(rec.o3_baseline_ir)
        if t3 is None:
            continue
        for r in recs:
            if r.outcome is not RewriteOutcome.verified_faster or not r.extracted_ir:
                continue
            tr = timing.score(r.extracted_ir)
            if tr is None or tr.wall_ns <= 0:
                continue
            total += 1
            faster_by_wall += tr.wall_ns <= t3.wall_ns
            if r.speedup_vs_o3 is not None:
                mca_speedups.append(r.speedup_vs_o3)
                wall_speedups.append(t3.wall_ns / tr.wall_ns)
    return {
        "verified_faster_timed": total,
        "also_faster_by_wall_clock": faster_by_wall,
        "fraction_confirmed": faster_by_wall / total if total else None,
        "spearman_mca_vs_wall_speedup": spearman(mca_speedups, wall_speedups),
    }


def format_report(corpus: dict, rewrites: dict | None) -> str:
    L = ["=== mca vs wall-clock audit ===", "", "Corpus (-O0 vs -O3):"]
    c = corpus
    L.append(f"  timeable pairs:        {c['timeable_pairs']}")
    for k in ("wall_clock_monotonic", "mca_monotonic"):
        v = c[k]
        L.append(f"  {k:22} {v:.1%}" if v is not None else f"  {k:22} n/a")
    s = c["spearman_mca_vs_ns"]
    L.append(f"  spearman(mca, ns):     {s:.3f}" if s is not None else "  spearman(mca, ns):     n/a")
    if rewrites is not None:
        r = rewrites
        L += ["", "Rewrite audit (mca-`verified_faster` rewrites):",
              f"  timed:                 {r['verified_faster_timed']}",
              f"  also faster by wall:   {r['also_faster_by_wall_clock']}"]
        fc, sp = r["fraction_confirmed"], r["spearman_mca_vs_wall_speedup"]
        L.append(f"  fraction confirmed:    {fc:.1%}" if fc is not None else "  fraction confirmed:    n/a")
        L.append(f"  spearman(mca,wall Δ):  {sp:.3f}" if sp is not None else "  spearman(mca,wall Δ):  n/a")
    return "\n".join(L)


def run(args) -> None:
    timing = TimingPerf()
    if not timing.available():
        raise SystemExit("no clang on PATH — timing needs a host compiler")
    records = load_corpus(Path(args.corpus))
    corpus = corpus_audit(list(records.values()), make_perf("mca"), timing)
    rewrites = None
    if args.rewrites:
        grouped = load_rewrites(Path(args.rewrites))
        rewrites = rewrite_audit(grouped, records, timing)

    report = format_report(corpus, rewrites)
    print("\n" + report + "\n")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing_validation.json").write_text(
        json.dumps({"corpus": corpus, "rewrites": rewrites}, indent=2)
    )
    (out / "timing_validation.txt").write_text(report + "\n")
    print(f"wrote {out}/timing_validation.{{json,txt}}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit llvm-mca against real wall-clock timing")
    p.add_argument("--corpus", required=True, help="corpus JSONL file or dir")
    p.add_argument("--rewrites", help="optional run dir / rewrites.jsonl to audit verified_faster")
    p.add_argument("--out", default="results")
    return p


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
