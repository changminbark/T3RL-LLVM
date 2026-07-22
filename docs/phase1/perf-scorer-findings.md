# Perf-scorer findings (llvm-mca as a speed proxy)

The plan's Step 0 sanity check: does `llvm-mca` rank `-O0` cycles > `-O3` cycles reliably? If the
scorer can't even see the compiler's own optimizations as faster, it can't reward a model for
beating them. Measured on the 64-function bootstrap corpus (Homebrew LLVM 22, target
`aarch64-linux-gnu`).

## Headline

| scorer setting            | O0 ranked slower than O3 (correct) |
|---------------------------|------------------------------------|
| `llvm-mca` default (`--iterations=100`) | **70%** |
| `llvm-mca --iterations=1` | **91%** |
| `--iterations=1`, loop-free only | **98% (47/48)** |
| `--iterations=1`, loops only | **69% (11/16)** |

**We use `--iterations=1`** (baked into `McaPerf`).

## Why the default is wrong

`llvm-mca` treats its input as a loop body executed `--iterations` times (default 100) and models
out-of-order execution across those iterations. Whole-function `-O0` code recomputes everything
from the stack, so its "iterations" are mutually independent and the OoO engine overlaps them
heavily — hiding the real per-call cost. At 100 iterations that artifact makes `-O0` look as fast
as, or faster than, tight `-O3` code. One iteration measures a single execution and removes the
artifact.

## Residual failures at `--iterations=1` (6/64)

`array_max, array_sum, dot_product, factorial, str_len` (loops) and `unpack_sum` (loop-free).

Five of six are loops, and the cause is fundamental, not tunable: `llvm-mca` analyzes a **static**
instruction stream and cannot know a loop's **trip count**. `-O3` vectorizes/unrolls, so the
analyzed region has *more* static instructions; mca reads that as "more cycles" even though real
runtime (many iterations of a tighter body) is lower. Static analysis simply cannot compare loop
performance without trip-count information. (`unpack_sum` is a lone loop-free outlier: `-O3` picks a
wider but higher-latency instruction sequence that mca prices above the `-O0` version.)

## Implications for the reward

- **`llvm-mca` is a trustworthy speed signal for loop-free functions (98%).** Reward built on it
  is sound there.
- **It is not reliable for loops (69%)** — it will sometimes reward *de*-optimizing a loop. This
  converges with Alive2's own limitation (loops are only checked to a few unrolled iterations):
  **both halves of the oracle are strongest on loop-free code.** Phase 1's go/no-go therefore
  rightly targets the loop-free buckets.
- Recommendation for Phase 2/3: gate the mca-based reward to loop-free functions, or pair it with a
  loop-aware signal (trip-count-weighted cost, or actual runtime on sampled inputs) before trusting
  it on loops. `code_size_bytes` (already in `PerfScore`) is a cheap, monotonic secondary check.

## Toolchain note

Apple's system `clang` emits IR with Apple-only target features that crash Homebrew's `llc`/`mca`
("Unsupported stack probing method"), and `llvm-mca` also crashes on macOS asm directives
(`.subsections_via_symbols`). Both are avoided by using one consistent Homebrew toolchain and a
non-Darwin triple (`aarch64-linux-gnu`). See `src/probe/tools.py`; override with `$LLVM_BIN` and
`$PROBE_TARGET`.
