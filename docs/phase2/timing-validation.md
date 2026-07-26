# Phase 2 hardening — real wall-clock timing vs llvm-mca

> Terminology: N = rewrites sampled per function (the pool), K = selection budget (K ≤ N) — see
> [README.md](README.md).

Phase 2's coverage/speedup numbers ride entirely on `llvm-mca` cycle estimates, which the
[findings](findings.md) flag as the reviewer-critical weakness (reliable loop-free, weak on loops;
cycles ≠ wall-clock). This adds a **real timing** path — as both an audit of mca and a selectable
reward metric.

## What was added

- **`TimingPerf`** (`src/probe/timing.py`) — a `PerfScorer` that compiles a function's IR + a
  synthesized, seeded C benchmark driver into a **native host** executable and measures median
  per-call nanoseconds. It strips the IR's linux target lines and lets the resolved clang retarget to
  the host, so it runs natively on the Linux box **and** the macOS dev machine. Selectable as
  `run_probe --perf timing`; scope is integer/pointer signatures (float/struct/vararg → not timeable).
- **Metric-agnostic reward** — `PerfScorer.cost()` / `cached_baseline()` let `outcome.classify`
  compare on whichever metric the scorer produces; `speedup_vs_o3` stays a unit-free ratio, so
  `bestofk` / `phase2_baseline` are unchanged.
- **`timing_validation`** (`src/probe/timing_validation.py`) — the audit: correlate mca vs ns on the
  corpus, and (with `--rewrites`) check what fraction of mca-`verified_faster` rewrites are *really*
  faster by wall-clock.

## Result (bootstrap corpus, 64 functions, macOS arm64)

| metric | ranks -O0 slower than -O3 |
|---|--:|
| **wall-clock** | **100%** (64/64) |
| llvm-mca (`--iterations=1`) | 90.6% |

Rank correlation **Spearman(mca cycles, wall ns) = 0.51** — mca is a *directionally useful but loose*
proxy: it almost always agrees on the O0-vs-O3 direction, but its magnitude only moderately tracks
real time, and its ~9% ranking errors concentrate on loops (it can't see trip counts). Real timing,
by contrast, ranks every pair correctly and exposes the true loop wins (e.g. `bubble_pass`
4185 ns → 1195 ns, `sum_to_n` loop → closed-form ≈ 0 ns).

**Takeaway:** trust mca for coarse loop-free selection; use `--perf timing` (or the audit) when a
number needs to survive review, especially on loops.

## Run it

```bash
# audit mca vs wall-clock on a corpus (+ optionally an existing run's verified_faster rewrites)
uv run python -m probe.timing_validation --corpus data/corpus --rewrites results/<run>
# or drive selection by real time
uv run python -m probe.run_probe --corpus data/corpus --backend api --model <id> \
    --k 16 --verifier alive --perf timing
```

Note: the perf toolchain must match the corpus's LLVM (source `scripts/alive2/env.sh` for the
llvm@21 corpus; the bootstrap seed is llvm@22, so run it without `env.sh`). Laptop timing is noisy on
sub-nanosecond functions — the audit reports *rank* agreement (Spearman) for that reason; rerun on a
quiet Linux box for paper-grade magnitudes.
