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
