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
