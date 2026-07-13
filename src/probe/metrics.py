"""Aggregate per-rewrite results into the headline metrics: solve@K and the outcome distribution.

Both are reported per size bucket and by loop presence, matching the plan's deliverable table.
"""

from __future__ import annotations

from collections import defaultdict

from .schema import CorpusRecord, RewriteOutcome, RewriteResult

_ALL_OUTCOMES = list(RewriteOutcome)


def summarize(
    records: dict[str, CorpusRecord], results: list[RewriteResult]
) -> dict:
    """Return a nested summary: per bucket -> {solve_at_k, n_functions, outcome_distribution}."""
    by_fn: dict[str, list[RewriteResult]] = defaultdict(list)
    for r in results:
        by_fn[r.function_id].append(r)

    buckets: dict[str, list[str]] = defaultdict(list)  # bucket key -> function_ids
    for fid, rec in records.items():
        if fid not in by_fn:
            continue
        loop = "loops" if rec.has_loops else "loopfree"
        buckets[f"{rec.size_bucket()}|{loop}"].append(fid)
        buckets["ALL"].append(fid)

    summary = {}
    for bucket, fids in sorted(buckets.items()):
        solved = 0
        outcome_counts = defaultdict(int)
        total_rewrites = 0
        for fid in fids:
            fn_results = by_fn[fid]
            if any(r.outcome is RewriteOutcome.verified_faster for r in fn_results):
                solved += 1
            for r in fn_results:
                outcome_counts[r.outcome] += 1
                total_rewrites += 1
        n = len(fids)
        summary[bucket] = {
            "n_functions": n,
            "solve_at_k": solved / n if n else 0.0,
            "outcome_distribution": {
                o.value: outcome_counts[o] / total_rewrites if total_rewrites else 0.0
                for o in _ALL_OUTCOMES
            },
        }
    return summary


def format_table(summary: dict) -> str:
    """Render the summary as a fixed-width text table for quick scanning."""
    header = f"{'bucket':<18}{'n':>4}{'solve@K':>10}  outcome distribution"
    lines = [header, "-" * len(header)]
    for bucket, s in summary.items():
        dist = s["outcome_distribution"]
        top = ", ".join(
            f"{o}={dist[o]:.2f}" for o in (
                "verified_faster", "verified_no_gain", "counterexample",
                "invalid_syntax", "timeout", "unsupported",
            )
        )
        lines.append(
            f"{bucket:<18}{s['n_functions']:>4}{s['solve_at_k']:>10.3f}  {top}"
        )
    return "\n".join(lines)
