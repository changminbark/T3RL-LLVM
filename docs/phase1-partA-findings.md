# Phase 1 · Part A — Verifier & Corpus Feasibility — Findings

**Date:** 2026-07-18 · **Machine:** macOS arm64, 8-core, 16 GB.
**Toolchain:** Alive2 v21.0 + LLVM/clang 21.1.8 (Homebrew `llvm@21`). Build: `docs/alive2-build.md`.

## What was run

- **Corpus:** 800 single functions extracted (`llvm-extract`) from **llvm-test-suite/SingleSource**
  (2,519 `.c` files; header-using files skip on macOS under the `aarch64-linux-gnu` target — see
  Caveats). Compiled `-O0` (`src_ir`) and `-O3` (`o3_baseline_ir`) with clang 21; 784 have an O3
  baseline. Skewed small (696 of ≤20 instrs) because the walk hits small functions first under the
  `--max-functions 800` cap.
- **Verdict-rate experiment** (`verify_corpus`, 30 s timeout): run Alive2 on `(-O0, -O3)` for every
  function — "can the oracle verify the compiler's own optimization?" Verdict rate + latency per
  size/loop bucket.
- **Perf sanity** (`perf_sanity`): does `llvm-mca` rank `-O0` cycles ≥ `-O3` cycles?

## Headline numbers

| bucket | n | verified | counterex. | **timeout** | unsupported | couldn't-prove* |
|---|--:|--:|--:|--:|--:|--:|
| ≤20  loop-free | 687 | **85%** | 27 | 1 | 15 | 59 |
| ≤20  loops     | 2   | 1 | 1 | 0 | 0 | 0 |
| 20–50 loop-free| 46  | **41%** | 6 | 0 | 1 | 20 |
| 20–50 loops    | 10  | **0%** | 3 | 0 | 1 | 6 |
| 50–150 loop-free|25  | **60%** | 1 | 0 | 1 | 8 |
| 50–150 loops   | 11  | **0%** | 3 | **2** | 3 | 3 |
| >150 loop-free | 3   | 1 | 0 | 0 | 0 | 2 |

\* "couldn't-prove" = Alive2 `ERROR: Couldn't prove the correctness of the transformation` (91) plus
a few genuine edge errors (6 "source doesn't reach a return", 1 SMT-incomplete). These are the
prover **abstaining**, not the harness crashing — currently classified `error`.

**Aggregates (784 checked):** verified 79% · counterexample 5% · **timeout 0.4% (3/784)** ·
unsupported 3% · couldn't-prove 12.5%.
**Loop-free vs loops:** 81% verified on loop-free (620/761) vs **4% on loops (1/23)**.
**Median verification time:** 0.02–1.3 s per bucket (p90 ≤ 10.5 s). **Perf sanity: 98% (764/783)**
monotonic — inversions confined to loops (llvm-mca can't see trip counts).

## Go / no-go (pre-registered thresholds)

- **"Median verification time ≤ 30 s"** → **PASS, decisively.** Sub-second medians. The feared
  kill-question — *Alive2 times out on realistic functions* — **does not materialize**: 0.4% timeout
  rate. The oracle is cheap enough for an RL reward loop.
- **"Verdict on ≥ 60% of the two smaller buckets"** → **PASS for ≤20 (85%), borderline for 20–50
  loop-free (41% verified / 54% with counterexamples).** The limiter is not timeouts but the prover
  **abstaining** ("couldn't prove"), which rises with size.
- **Loops:** effectively out of reach for pure Alive2 (≈0% verified; fails via couldn't-prove /
  counterexample / unsupported, as the plan predicted — loops are only checked to a few unrolled
  iterations).

**Verdict: GO, scoped to loop-free functions (up to ~150 instrs) for Phase 2.** The oracle is fast,
reliable on straight-line code, and the timeout risk is retired. Loops need a different strategy
(bounded-unroll acceptance tier, or exclude) before they contribute reward signal.

## Caveats (affect interpretation, not the go/no-go)

- **Corpus skew & macOS limits.** The 800 are mostly tiny loop-free functions; only 23 loops. A
  loop-rich, larger corpus needs a Linux box where clang can compile header-using SingleSource files
  natively (the runbook already routes the corpus build there). Numbers here are a floor, not the
  final calibrated corpus.
- **`llvm-extract` artifacts.** Pulling one function out of context can make the `-O3` copy diverge
  from `-O0` (O3 specialized on callers / globals / signature), producing **spurious
  counterexamples**. Some of the 41 counterexamples are likely artifacts, not real miscompiles — to
  confirm in Phase 2.
- **"couldn't-prove" deserves its own label.** It is a distinct oracle outcome (prover abstains ≠
  tool error ≠ proven-different). Recommend adding a `failed_to_prove` value to `RewriteOutcome` /
  `VerdictStatus` (a shared-schema change — coordinate with Part B) so the reward distribution is
  honest. For reward purposes it behaves like timeout/unsupported (not verified → reward 0).

## Status of the other half (Workstream B)

`solve@K` (does a base model produce verified-and-faster rewrites) was **not run** — it needs a
model backend (no API key / GPU in this environment) and is Person B's deliverable. The reward
pipeline it depends on is now proven working end-to-end with real Alive2.

## Reproduce

```bash
source scripts/alive2/env.sh                 # ALIVE_TV + LLVM_BIN (llvm@21)
uv run python -m probe.build_corpus --src "$(./scripts/fetch-corpus.sh -q)" \
    --out data/corpus/corpus.jsonl --with-mca --max-functions 800
uv run python -m probe.verify_corpus --corpus data/corpus --timeout 30   # -> results/verdict_rates.{json,txt}
uv run python -m probe.perf_sanity  --corpus data/corpus --perf mca
```
