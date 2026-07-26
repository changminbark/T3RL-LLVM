# Phase 2 hardening — IR robustness (recovering `invalid_syntax`)

> Terminology: N = rewrites sampled per function (the pool), K = selection budget (K ≤ N) — see
> [README.md](README.md).

**What it does:** normalizes the LLVM IR a model emits so Alive2 can parse it, instead of discarding
it as `invalid_syntax`. **Why it exists:** a real run showed the model's *raw-IR* rewrites were mostly
being thrown away for format reasons, not because the model lacked ability — masking real speedups.

## The problem

A deepseek-v4-pro K=16 run (Alive2 v21 + llvm@21, 64-function `data/c_sources` corpus) produced
**61% `invalid_syntax`**. Two causes:

- **Full-module / placeholder output.** The model wraps a rewrite in a whole module and abbreviates
  boilerplate (`target datalayout = "..."`) or references attribute groups (`#0`, `#1`) whose defs
  are truncated → unparseable.
- **Truncation.** Verbose reasoning + a full module overran `max_tokens=2048` (39% of the invalids
  were literally cut off mid-output).

## The fix

- **`ir_utils.sanitize_module`** (in `outcome.classify` for `--format ir`): inject a valid
  `target datalayout`/`triple` from the source module, drop `attributes #N = {...}` blocks and strip
  `#N` references, keep the model's declares/globals/metadata/`define`. Text-only, no LLVM needed.
- **`--max-tokens` default 2048 → 8192**, and a prompt that forbids `...` and lets the model omit
  module boilerplate (so completions are short and clean — avg 594 chars, down from ~4800).

## Result

Same corpus and model, K=16:

| | `invalid_syntax` | `verified_faster` | Coverage@16 | MeanSpeedup@16 |
|---|--:|--:|--:|--:|
| before | 61% | 8 (1%) | 10.9% | 1.18× |
| sanitize only (replayed on the *same* completions) | 41% | 33 (3%) | 21.9% | 1.45× |
| **sanitize + 8192 tokens (fresh run)** | **10%** | **150 (15%)** | **23.4%** | **1.39×** |

- **Coverage doubled from the sanitizer alone** (11% → 22%), re-verifying completions the model had
  already produced — a clean ablation with no new sampling.
- The token bump then cleared the truncation: only 8% of the remaining invalids are truncated (vs
  39%), so the leftover ~10% is mostly genuinely-bad IR — roughly the real floor.
- The result lands on the [reference baseline](findings.md) (28% / 1.38×). The remaining gap is
  corpus, not pipeline: `data/c_sources` skews small/loop-free.

Where the wins concentrate (best-of-16): **20–50 loops 75% / 2.06×**, **≤20 loops 33% / 4.52×**;
≤20 loop-free (42 of 64 functions) only 10% / 1.03× — small straight-line code is already near-optimal
at `-O3`.

## Takeaway

For `--format ir`, parseability — not model ability — was the binding constraint. The curve also
**saturates early** (best-of-1 already 14.6% coverage; K=8→16 barely moves), so more sampling buys
little. That's the setup Phase 3 targets: shift the whole K-curve up by making the base model itself
emit more `verified_faster` rewrites, rather than sampling more.
