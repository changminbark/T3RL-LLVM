# Phase 2 — Best-of-K Inference-Time Baseline — Design

> Terminology: **N** = rewrites sampled per function (the pool), **K** = selection budget (K ≤ N). Note: code snippets below keep the source's local param names `n`/`k`. See [../README.md](../README.md).

> **Status:** approved design, pre-implementation.
> **Companion docs:** `../../phase1/partA-plan.md` (master brief), `../../phase1/partA-findings.md`
> (Phase 1 verifier results), Phase 1 Workstream-B `solve@K` results in `results/probe_*`.

## Context

Phase 1 answered the two viability kill-questions (verifier usable? model prior nonzero?) → GO,
scoped to loop-free functions. **Phase 2 establishes the non-TTRL baseline**: best-of-K sampling
with Alive2 verification + llvm-mca scoring used as pure *selection*. Per the master plan, "this
number is what TTRL must beat, and the pipeline becomes the RL reward function unchanged."

This is also the first experiment that measures **LLVM improving the result over the base model**:
Phase 1 used the oracle only to *grade*; Phase 2 uses it to *select*, and reports the lift.

## Key reuse — Phase 2 is a layer, not a new pipeline

The rollout + grading pipeline already exists. `run_probe` samples K rewrites per function,
classifies each via `outcome.classify`, and writes `results/<run>/<tag>.rewrites.jsonl` where each
record (`schema.RewriteResult`) carries: `function_id`, `sample_index`, `outcome`
(`verified_faster` | `verified_no_gain` | `counterexample` | `invalid_syntax` | `timeout` |
`unsupported`), `speedup_vs_o3` (float | null), `rewrite_cycles` (float | null).

Phase 2 adds a **selection + aggregation layer** over those records. It does not touch generation,
verification, or scoring. Verified data already exists: the deepseek K=8 run
(`results/probe_llm_deepseek/*.rewrites.jsonl`, 512 rewrites, 64 verified-faster, speedups 1.04–2.6×).

## Metric definitions

For one function with `n` sampled rewrites:

- A rewrite is **verified** if `outcome ∈ {verified_faster, verified_no_gain}` (Alive2 proved
  equivalence). It **beats -O3** if `outcome == verified_faster` (strictly fewer mca cycles than
  the O3 baseline, i.e. `speedup_vs_o3 > 1`).
- **best-of-K selection**: among K samples, keep the verified ones, pick the one with the fewest
  `rewrite_cycles` (fastest). Its achieved speedup over -O3 is `max(1.0, speedup_vs_o3)` — in
  deployment you keep -O3 when the model can't beat it, so speedup never drops below 1.0.

Reported as a curve over **K ∈ {1, 2, 4, 8, 16}**, overall and per (size bucket × has_loops):

- **Coverage@K** = fraction of functions whose best-of-K beats -O3 (has ≥1 `verified_faster` in K).
- **MeanSpeedup@K** = mean over functions of the best-of-K achieved speedup (`max(1.0, …)`).

### Statistical estimator (required — not "first k")

With only `n` samples per function (8, later 16), taking "the first k" is high-variance. Use
unbiased estimators over all `C(n,k)` subsets:

- **Coverage@k** uses the pass@k estimator, per function then averaged:
  `passk(n, c, k) = 1 − C(n−c, k) / C(n, k)`, where `c` = number of `verified_faster` among the
  function's `n` samples (`0` when `k > n − c` guard; `1.0` when `c ≥ n − k + 1`). Compute in log
  space or with `math.comb` to avoid overflow.
- **MeanSpeedup@k** uses the **expected best-of-k speedup** over random k-subsets. Per function,
  let `s_1 ≥ s_2 ≥ … ≥ s_n` be the per-sample achieved speedups (`max(1.0, speedup_vs_o3)` for
  verified samples, `1.0` for non-verified — a non-verified sample contributes no usable rewrite,
  so its selectable value is the -O3 fallback = 1.0). The probability that `s_i` is the max of a
  random k-subset is `C(n−i, k−1) / C(n, k)` (ties broken by index). Expected best-of-k =
  `Σ_i s_i · C(n−i, k−1) / C(n, k)`. Average across functions.
- When `k ≥ n`, both reduce to the observed best over all `n` samples.

These are closed-form (no Monte-Carlo needed at n ≤ 16), deterministic, and low-variance.

## Components

### `src/probe/bestofk.py` (pure, no I/O)

- `passk_estimator(n: int, c: int, k: int) -> float` — the coverage estimator above.
- `expected_best_speedup_at_k(speedups: list[float], k: int) -> float` — expected max over random
  k-subsets, given a function's per-sample achieved speedups.
- `per_function_speedups(records: list[RewriteResult]) -> list[float]` — map a function's rewrite
  records to per-sample achieved speedups (`max(1.0, speedup_vs_o3)` if `verified_faster`, else
  `1.0`).
- `curve(records_by_function: dict[str, list[RewriteResult]], buckets: dict[str, tuple[str, bool]], ks: list[int]) -> dict`
  — per-K `{coverage, mean_speedup, n_functions}`, overall and per `(size_bucket, has_loops)`
  (`buckets` maps `function_id -> (size_bucket, has_loops)`; the driver joins from the corpus).

### `src/probe/phase2_baseline.py` (driver / report)

- CLI: `phase2-baseline --rewrites <run-dir-or-jsonl> --corpus <jsonl> [--ks 1,2,4,8,16] --out results`.
- Loads `rewrites.jsonl` (reuse a small loader; group by `function_id`), joins buckets from the
  corpus (`load_corpus` from `run_probe`), calls `curve`, prints a K-curve table, writes
  `results/phase2_baseline.{json,txt}`.
- Handles a run whose max samples < requested K by capping (report the actual K available and warn).

## Data flow

```
run_probe (K samples) ─▶ results/<run>/<tag>.rewrites.jsonl  (RewriteResult[])
                                        │
              corpus.jsonl ─▶ phase2_baseline.py ─▶ bestofk.curve ─▶ results/phase2_baseline.{json,txt}
              (buckets)          (group by fn, join buckets)         (coverage@K, mean_speedup@K, per bucket)
```

## Run plan

1. **Immediate curve (free):** run `phase2_baseline` over the existing deepseek K=8 data → K=1–8
   curve now, no new API calls.
2. **Firm baseline:** one `run_probe` at **K=16** with deepseek on the clean corpus
   (`--verifier alive --perf mca`, `env.sh` for ALIVE_TV + LLVM_BIN) → full K=1–16 curve.
3. Report the "with-LLVM vs without" reading: best-of-1 (oracle-verified single sample) vs
   best-of-K lift; and the honest note that without the oracle there is no *trustworthy* output.

## Error handling

- Missing/empty rewrites file → clear exit message. Records with `speedup_vs_o3 == null` on a
  `verified_faster` outcome (shouldn't happen) → treated as speedup 1.0 with a logged warning.
- `k > n_samples` → cap to `n`, warn once. `math.comb` used for exact combinatorics.

## Testing (all offline, no LLVM/model needed)

- `passk_estimator`: hand-computed values — `passk(8,0,k)=0`, `passk(8,8,1)=1`, `passk(4,1,2)=0.5`,
  `passk(4,2,2)=5/6`.
- `expected_best_speedup_at_k`: tiny known set, e.g. speedups `[2.0, 1.0]`, k=1 → 1.5; k=2 → 2.0.
- `per_function_speedups`: verified_faster → its speedup; other outcomes → 1.0.
- `curve`: synthetic 3-function set, assert coverage/mean per K and per bucket.
- Driver smoke: run over a tiny fixture `rewrites.jsonl` + corpus, assert table + JSON written.

## Out of scope (YAGNI)

- Any RL / weight updates (that's Phase 3).
- New generation formats or a format-A-vs-C study (separate experiment).
- Changing the reward/outcome pipeline — Phase 2 consumes it unchanged.
- A larger/loop-rich corpus rebuild (Linux job; the caveat is carried, not fixed here).

## Deliverables checklist

- [ ] `bestofk.py` (estimators + curve) with offline unit tests.
- [ ] `phase2_baseline.py` driver emitting `results/phase2_baseline.{json,txt}`.
- [ ] K=1–8 curve from existing deepseek data; K=16 run for the full curve.
- [ ] Short findings note: the best-of-K baseline (the number TTRL must beat) + with/without-LLVM read.
