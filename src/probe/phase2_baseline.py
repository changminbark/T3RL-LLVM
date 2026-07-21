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
