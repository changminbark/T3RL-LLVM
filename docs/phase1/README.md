# Phase 1 — Viability

Inference-only experiments (no RL, no training) answering the two kill-questions before investing in
a TTRL loop. Two independent workstreams:

- **Part A — verifier & corpus feasibility.** Can Alive2 verify the compiler's own `-O0`→`-O3`
  optimizations, often enough and fast enough to serve as a reward oracle?
- **Part B — model capability probe.** Does an accessible base model have a nonzero prior on the task
  (producing verified-and-faster rewrites) that TTRL could amplify?

## Terminology

- **N** — rewrites sampled per function (the pool; what `run_probe --k` sets).
- **K** — the selection/evaluation budget in `@K` metrics, K ≤ N. Phase 1 generated N and reported
  **`solve@K` at K=N** (fraction of functions with ≥1 verified-and-faster rewrite in the pool).
- **Verdict outcomes** (Alive2 per `(src, tgt)` pair): `verified` · `counterexample` (proven
  different) · `timeout` · `unsupported` · `couldn't-prove` (prover abstains — distinct from a tool
  error).
- **Buckets** — functions grouped by size (`n_instructions`: ≤20, 20–50, 50–150) and `has_loops`.

## Findings summary

**Part A ([details](partA-findings.md)) — verifier is usable, scoped to loop-free.** Over 784
functions (llvm-test-suite, verified at `(-O0, -O3)`):

- **79% verified overall**, but **81% on loop-free vs 4% on loops** — loops are effectively out of
  reach for pure Alive2 (only checked to a few unrolled iterations).
- **Timeout just 0.4%** and **median 0.02–1.3 s** — the feared "Alive2 times out on real functions"
  kill-question does not materialize; the oracle is cheap enough for an RL reward loop.
- **Perf sanity: 98% monotonic** (llvm-mca ranks `-O0` ≥ `-O3`), inversions confined to loops.
- **Verdict: GO, scoped to loop-free functions (≤~150 instrs).**

**Part B ([plan](partB-plan.md)) — a real, un-gamed prior exists.** `solve@K` needs a model backend;
the later Phase 2 deepseek run confirms a frontier model produces genuinely verified-faster rewrites,
while a small model mostly fails on IR syntax (see `figures/plot2_slm_vs_llm.png`).

**Perf scorer ([details](perf-scorer-findings.md)).** `llvm-mca` is a reliable speed proxy for
loop-free code (98%) but weak on loops (69%); run it at `--iterations=1` (the default 100 distorts
whole-function comparisons).

## Contents

- [`partA-plan.md`](partA-plan.md) · [`partA-findings.md`](partA-findings.md) — verifier/corpus.
- [`partB-plan.md`](partB-plan.md) — model capability probe.
- [`perf-scorer-findings.md`](perf-scorer-findings.md) — llvm-mca as a speed proxy.
- [`alive2-build.md`](alive2-build.md) — building the `alive-tv` oracle (also used by Phase 2).
- [`plans/`](plans/) · [`specs/`](specs/) — Part A implementation design records.
