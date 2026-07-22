# Phase 1 · Part A — Verifier & Corpus Feasibility — Design

> Terminology: **N** = rewrites sampled per function (the pool), **K** = selection budget (K ≤ N). Note: code snippets below keep the source's local param names `n`/`k`. See [../README.md](../README.md).

> **Status:** approved design, pre-implementation.
> **Companion docs:** `../partA-plan.md` (master brief), `../partB-plan.md`
> (Chang Min's Part B, already implemented — the frozen interface contract).

## Context

We are testing viability of an unsupervised **TTRL superoptimizer**: a model rewrites LLVM IR
functions to be provably-equivalent-but-faster, using **Alive2** (theorem-prover translation
validator) as an un-gameable reward oracle and **llvm-mca** as the speed scorer. Phase 1 is
inference-only viability testing, split into two parallel workstreams:

- **Part A (this doc):** Is the *verifier* usable in practice? Produce the corpus + Alive2
  harness + the verdict-rate/latency table.
- **Part B (done):** Does any base model have a nonzero prior on the task? (`solve@K`.)

Part A's headline deliverable is the **one-page table of Alive2 verdict rates and median
verification times per size/loop bucket**, measured by asking Alive2 to verify the compiler's
*own* `-O0`→`-O3` optimization on every corpus function. That rate upper-bounds how often we
get oracle signal on model outputs.

## Guiding constraint — Part A is additive to Part B

Part B froze the contracts in `src/probe/schema.py` and built stub/real seams. Part A satisfies
those seams **exactly**; it does not redesign them.

- **Verifier seam:** `AliveCliVerifier` (in `verifier.py`) shells out to a CLI named
  `alive-harness`, invoked as `alive-harness <src.ll> <tgt.ll> --timeout <s>`, and parses the
  **last line of stdout** as `Verdict` JSON (`status`, `counterexample`, `wall_time_s`). Part A
  provides that CLI.
- **Perf seam:** `McaPerf` (in `perf.py`) already wraps `llc` + `llvm-mca` → `PerfScore`. Part A
  does **not** rewrite it; it only adds a sanity-check that it ranks `-O0` cycles ≥ `-O3` cycles.
- **Corpus seam:** the corpus is JSONL of `CorpusRecord`. `build_corpus.py` (B's bootstrap) already
  computes `n_instructions` / `has_loops` / O3 baseline from a directory of `.c` files. Part A
  **extends** this file, reusing its helpers, rather than forking a parallel builder.

### Environment note (this session)

This machine has only Apple's `/usr/bin/clang`. No `opt`, `llc`, `llvm-mca`, `llvm-extract`,
`llvm-as`, and no `alive-tv`/Alive2. **Session scope (agreed): code-complete + swap-ready.** All
Part A code is built and fully unit-tested offline (tools stubbed / output fixtures), and drops
into Part B's seams unchanged. Installing the real toolchain and running the 500-function
experiment happens in a later session. Every component therefore **degrades gracefully when its
external tool is absent** — a crash is a bug.

## Components

### A. `alive-harness` CLI — `src/probe/alive_harness.py`

The Alive2 wrapper B's `AliveCliVerifier` shells out to. Exposed as a **console script** in
`pyproject.toml` (`alive-harness = "probe.alive_harness:main"`) so `shutil.which("alive-harness")`
resolves after `uv sync`.

- **Invocation (frozen by B):** `alive-harness src.ll tgt.ll --timeout <s>`. Optional
  `--alive-tv <path>` / `ALIVE_TV` env override for locating `alive-tv`.
- **Behavior:** locate `alive-tv`; run it sandboxed (`alive-tv src.ll tgt.ll`) with a hard wall
  timeout; capture stdout/stderr; classify into one `VerdictStatus`; print exactly one line of
  `Verdict` JSON to stdout.
- **Classification** from `alive-tv` output text:
  - "Transformation seems to be correct" / all-correct summary → `verified`
  - "Transformation doesn't verify" + a counterexample block → `counterexample` (capture the
    example text, truncated)
  - our wall timeout tripped, or alive-tv reports a timeout → `timeout`
  - "ERROR: ... unsupported" / unsupported-feature markers → `unsupported`
  - alive-tv missing, nonzero exit without a recognized verdict, or unparseable → `error`
    (with stderr excerpt in `counterexample`)
- **Missing `alive-tv` → `error` verdict, exit 0.** Never raises. This is what makes the CLI
  unit-testable now: parsing logic is a pure function over captured `alive-tv` text fixtures.

The parser is a pure function `classify_alive_output(stdout, stderr, timed_out) -> Verdict`
kept separate from the subprocess/IO shell, so tests target it directly.

### B. Scaled corpus builder — extend `src/probe/build_corpus.py`

Add an LLVM test-suite adapter and per-function extraction on top of B's existing file.

- **Input:** a directory tree of self-contained `.c` files (canonical source: an
  `llvm-test-suite/SingleSource` clone; any `.c` dir also works). Walk `.c` files **recursively**.
- **Per file:** compile whole module `-O0` and `-O3` (reuse `_clang_ir`), then split **each
  defined function** into its own single-function module via `llvm-extract -func=<name> -S`.
  Function names come from the existing `_DEFINE_RE`. The O0 extraction is the record's `src_ir`;
  the matching function extracted from the O3 module is `o3_baseline_ir`.
- **Per function record:** reuse `count_instructions` / `has_loops` for `n_instructions` /
  `has_loops`; `size_bucket()` already lives on `CorpusRecord`. Fill `mca_cycles_o3` via `McaPerf`
  when `--with-mca` and tools present (existing path).
- **Dedup** by a hash of normalized `src_ir` (drop identical functions common across test files).
  **Skip** functions with no body (declarations) and functions that fail to extract.
- **Scale controls:** `--max-functions N` (target 500–1000), keep the per-bucket counts balanced
  where possible; print a bucket histogram at the end.
- **Tool-absent guard:** if `llvm-extract` is missing, fall back to B's existing whole-file
  behavior and log; if `clang` is missing, exit with a clear message (existing behavior).

Function extraction is factored into a pure-ish helper `list_defined_functions(ir) -> list[str]`
(regex over `define`) so splitting is testable on a synthetic multi-function IR string without
`llvm-extract`.

### C. Verdict-rate experiment driver — `src/probe/verify_corpus.py` (headline)

Produces Part A's deliverable table.

- **CLI:** `verify-corpus --corpus data/corpus --timeout 30 --out results` (also a `--timeouts
  30,120` sweep to mirror the plan's 30s/120s variants).
- **Per `CorpusRecord`:** if `o3_baseline_ir` is present, run the equivalence check on
  `(src_ir, o3_baseline_ir)` through the **same** `AliveCliVerifier` B uses (import it — proves
  the harness end-to-end), recording `status` and `wall_time_s`. Records without an O3 baseline
  are counted as `skipped`.
- **Aggregate** per (size bucket × has_loops):
  - verdict rate = fraction with `status == verified`,
  - a full status breakdown (verified / counterexample / timeout / unsupported / error),
  - **median** and p90 `wall_time_s`.
- **Output:** `results/verdict_rates.json` (machine) + `results/verdict_rates.txt` (the one-page
  table). Aggregation math is a pure function `aggregate(records, verdicts) -> table` — unit-tested
  on synthetic verdicts, no tools needed.

### D. Perf sanity-check — `src/probe/perf_sanity.py`

- Over the first N corpus records with both `src_ir` and `o3_baseline_ir`, score both with
  `McaPerf` and check `cycles(O0) >= cycles(O3)`.
- Report the fraction that holds and any inversions (a red flag that the scorer is untrustworthy).
- Degrades to a clear "llvm-mca not available" message when tools are absent.

## Data flow

```
LLVM test-suite .c tree ─▶ build_corpus.py ─▶ data/corpus/*.jsonl  (CorpusRecord[])
   clang -O0/-O3, llvm-extract per func,                    │
   bucket, dedup, --max-functions                           ▼
                                        verify_corpus.py ─▶ results/verdict_rates.{json,txt}
                                        check(src_ir, o3_baseline_ir) per record   │
                    ┌── alive_harness.py (console script) ◀── subprocess ──────────┘
                    │      wraps alive-tv → Verdict JSON  (the CLI AliveCliVerifier calls)
                    │
perf_sanity.py ─▶ McaPerf (B's) over O0 vs O3 ─▶ monotonicity fraction
```

## Error handling

Uniform rule: **a missing external tool is a logged, structured degradation, never an
exception.** `alive-tv` absent → `Verdict(error)`; `llvm-extract` absent → whole-file fallback;
`llc`/`llvm-mca` absent → `None`/skip with reason. Subprocess timeouts are caught and mapped to
`timeout`. Malformed tool output maps to `error` with an excerpt, never a stack trace.

## Testing (all offline, no LLVM/Alive2 required)

- **`alive-tv` parsing fixtures:** captured example outputs for verified / counterexample /
  timeout / unsupported / malformed → `classify_alive_output` returns the right `VerdictStatus`
  and captures counterexample text.
- **A↔B round-trip:** `AliveCliVerifier(cli_cmd="alive-harness").check(src, tgt)` invoked against
  the real `alive_harness` entry point with a **fake `alive-tv`** on PATH (a tiny script emitting
  canned output) yields the expected verdict — proves the two halves mate.
- **Corpus builder:** `list_defined_functions` splits a synthetic multi-function IR string
  correctly; dedup and bucket-histogram logic on hand-built records.
- **Verdict-rate aggregation:** `aggregate` over synthetic (record, verdict) pairs produces
  correct per-bucket rates, medians, and status breakdowns.
- **Regression:** existing `uv run pytest tests/` (B's suite) stays green — Part A must not alter
  B's behavior.

## Out of scope (YAGNI for Phase 1 Part A)

- Installing/building LLVM or Alive2 (later session).
- Mining GitHub repos or ingesting prebuilt bitcode datasets (test-suite `.c` tree is the source).
- Any change to `schema.py`, `perf.py`'s `McaPerf`, or Part B's probe pipeline.
- Multi-turn / counterexample-repair logic (that's Part B's ablation).

## Deliverables checklist

- [ ] `alive-harness` console-script CLI (`alive_harness.py`) + `pyproject.toml` entry point.
- [ ] Extended `build_corpus.py`: test-suite adapter, per-function `llvm-extract`, dedup,
      `--max-functions`, bucket histogram.
- [ ] `verify_corpus.py` driver emitting `results/verdict_rates.{json,txt}`.
- [ ] `perf_sanity.py`.
- [ ] Offline test suite for all of the above; B's suite still green.
- [ ] Short note in `../partA-plan.md` or the shared doc confirming the `alive-harness`
      CLI contract matches what B calls.
