# Phase 2 — Best-of-K Inference-Time Baseline

The non-TTRL baseline: sample **N** rewrites per function, keep the Alive2-**verified** ones, and
select the **fastest** by llvm-mca. Its speedup over the compiler's `-O3` is **the number a Phase 3
TTRL loop must beat**, and — per the master plan — this pipeline *becomes* the RL reward function
unchanged. It's also the first experiment where LLVM *improves* the output (via selection), not just
grades it.

## Terminology

- **N** — rewrites sampled per function (the pool; `run_probe --k`).
- **K** — the selection budget, K ≤ N. Phase 2 generates N=16 and reports a **curve over
  K ∈ {1,2,4,8,16}** using unbiased estimators over all C(N,K) subsets (not "the first K").
- **Coverage@K** — fraction of functions whose best-of-K beats `-O3` (has a verified-faster rewrite).
- **MeanSpeedup@K** — mean achieved speedup over `-O3`, `max(1.0, best verified speedup)` per function
  (you keep `-O3` when the model can't beat it).
- **pass@K** — the coverage estimator, `pass@K = 1 − C(N−c, K) / C(N, K)` with `c` = verified-faster
  samples among the N.

## Findings summary

From a K=16 deepseek run over the clean corpus ([details](findings.md)):

- **Best-of-K saturates at ~28% coverage / 1.38× over `-O3`** — K=8→16 barely moves (26.5%→28.1%),
  a clean, non-runaway target that leaves well-defined headroom for TTRL.
- **Gains concentrate where there's headroom:** 75% coverage / **2.0×** on 20–50 loops; tiny
  loop-free functions are mostly already optimal at `-O3`.
- **LLVM as a tool:** without the oracle you have zero *trustworthy* speedups (raw samples can't be
  verified); oracle verify-and-select roughly **doubles coverage from K=1→16 at zero training cost**.

Figures: [`figures/plot1_bestofk_curve.png`](figures/plot1_bestofk_curve.png) (the K-curve),
[`plot3_coverage_by_bucket.png`](figures/plot3_coverage_by_bucket.png),
[`plot4_base_vs_llm_tool.png`](figures/plot4_base_vs_llm_tool.png).

**Caveat:** the whole signal rides on llvm-mca cycles (reliable loop-free, weaker on loops; cycles ≠
wall-clock) — a real-timing validation is the recommended next hardening.

## Contents

- [`findings.md`](findings.md) — full results, per-bucket tables, reproduce commands.
- [`plans/`](plans/) · [`specs/`](specs/) — best-of-K implementation design records.
