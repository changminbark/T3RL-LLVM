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
