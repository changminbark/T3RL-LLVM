# T3RL-LLVM

Testing the viability of an unsupervised **TTRL superoptimizer**: a model that rewrites LLVM IR
functions to be provably-equivalent-but-faster, using Alive2 (a theorem-prover-based translation
validator) as an un-gameable reward oracle and `llvm-mca` as the speed scorer. No labels, no
gameable grader — the compiler infrastructure is the labeler.

This repo currently holds **Phase 1, Part B — the model capability probe**: an inference-only
experiment measuring whether an accessible base model has a nonzero prior on the task (the ability
TTRL would amplify). See [`docs/phase1/partB-plan.md`](docs/phase1/partB-plan.md) for the full plan
and the shared interface with Part A (verifier & corpus).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Everything runs behind stubs, so no LLVM/Alive2/API key
is needed to exercise the pipeline end-to-end.

```bash
uv sync --extra dev          # create .venv, install deps (add --extra local for the vLLM backend)
uv run pytest                # unit tests
# Full pipeline on the offline mock backend + stub verifier/perf:
uv run python -m probe.run_probe --corpus data/bootstrap --backend mock --k 8 \
    --verifier stub --perf stub
```

That prints a bucketed `solve@K` table and writes per-run artifacts to `results/`
(`*.rewrites.jsonl`, `*.summary.json`, `*.table.txt`).

## End-to-end (real run)

The offline quickstart above uses stubs. A real run needs three external things: the **LLVM
toolchain**, the **Alive2 oracle**, and a **model backend**. Ordered runbook:

```bash
# 0. Python env
uv sync --extra dev

# 1-2. Alive2 oracle + LLVM 21 toolchain (~15 min, no from-source LLVM). Full guide: docs/phase1/alive2-build.md
git submodule update --init --recursive
./scripts/alive2/01-prereqs.sh && ./scripts/alive2/02-build-alive2.sh
source scripts/alive2/env.sh            # exports ALIVE_TV (oracle) + LLVM_BIN (llvm@21 for clang/llc/llvm-mca)

# 3. Build the real corpus. fetch-corpus.sh shallow-clones llvm-test-suite and prints
#    its SingleSource path. Run this on the Linux box where you built LLVM/Alive2 — clang
#    emits IR for aarch64-linux-gnu, which is native there; on macOS, files that include
#    Linux system headers fail to compile and get skipped.
uv run python -m probe.build_corpus --src "$(./scripts/fetch-corpus.sh -q)" \
    --out data/corpus/corpus.jsonl --with-mca --max-functions 800

# 4. Part A — verifier feasibility (the go/no-go table) + perf sanity
uv run python -m probe.verify_corpus --corpus data/corpus --timeout 30   # -> results/verdict_rates.{json,txt}
uv run python -m probe.perf_sanity  --corpus data/corpus --perf mca      # llvm-mca ranks O0 >= O3?

# 5. Part B — model probe (real solve@K) against the same oracle
export TOGETHER_API_KEY=...
uv run python -m probe.run_probe --corpus data/corpus --backend api \
    --model "Qwen/Qwen2.5-Coder-32B-Instruct" --base-url https://api.together.xyz/v1 \
    --api-key-env TOGETHER_API_KEY --k 16 --verifier alive --perf mca
```

Steps 1–2 are independent (Alive2 builds its own LLVM) and can run in either order. You get partial
signal before finishing the slow parts: a model key alone gives Part B's syntactic-validity/format
numbers (`--verifier stub --perf stub`); adding `brew install llvm` gives real `llvm-mca` scoring;
only Alive2 unlocks real equivalence verdicts. Details in the sections below.

## Running against a real model

```bash
# Hosted OpenAI-compatible API (set the key env named by --api-key-env):
export TOGETHER_API_KEY=...
uv run python -m probe.run_probe --corpus data/bootstrap --backend api \
    --model "Qwen/Qwen2.5-Coder-32B-Instruct" --base-url https://api.together.xyz/v1 \
    --api-key-env TOGETHER_API_KEY --k 16 --temperature 0.9

# Local GPU via vLLM (install the extra first: uv sync --extra local):
uv run python -m probe.run_probe --corpus data/bootstrap --backend vllm \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" --k 16
```

See `configs/models.yaml` for preset backend/sampling combinations.

## Building the corpus

`data/bootstrap/seed.jsonl` is generated from single-function C files in `data/c_sources/` using
the local clang (`-O0` → `src_ir`, `-O3` → `o3_baseline_ir`, with `n_instructions`/`has_loops`
computed from the IR). Regenerate it after adding C files:

```bash
uv run python -m probe.build_corpus --src data/c_sources --out data/bootstrap/seed.jsonl
# add --with-mca to also fill mca_cycles_o3 (requires llvm-mca + llc)
```

Person A's `data/corpus/` (git-ignored) is the source of truth for the real runs.

## Real verifier & performance scorer

The `stub` verifier only recognizes trivially-equal IR; `stub` perf uses an instruction-count
proxy. Once LLVM tools and the Alive2 harness are installed (build Alive2 with
[`docs/phase1/alive2-build.md`](docs/phase1/alive2-build.md); `--verifier alive` shells out to `alive-harness`,
which finds `alive-tv` via `$ALIVE_TV`), swap them in with no other code changes:

```bash
uv run python -m probe.run_probe --corpus data/corpus --backend api --model <id> \
    --k 16 --verifier alive --perf mca
```

**LLVM toolchain:** `--perf mca` and `--with-mca` need `llc` + `llvm-mca` (e.g. `brew install
llvm`). All LLVM tools (clang, llc, llvm-mca) must come from the *same* install — on macOS, Apple's
clang emits IR that Homebrew's tools crash on. `tools.py` auto-resolves the Homebrew keg; override
with `LLVM_BIN=/path/to/llvm/bin` and the target triple with `PROBE_TARGET` (default
`aarch64-linux-gnu`, chosen so llvm-mca doesn't choke on macOS asm directives).

**Caveat — `llvm-mca` is only a reliable speed proxy for loop-free functions (98%), not loops
(69%).** We run it at `--iterations=1`; see [`docs/phase1/perf-scorer-findings.md`](docs/phase1/perf-scorer-findings.md)
for the measurements and why.

## Key details

- **Generation formats:** `--format ir` (model emits LLVM IR directly) or `--format c` (model
  emits C, lowered to IR via `clang`).
- **Ablation:** `--include-o3` also shows the `-O3` baseline in the prompt as a starting point.
- **Metrics:** `solve@K` (fraction of functions with ≥1 verified-and-faster rewrite) and the
  per-outcome distribution, reported per size/loop bucket — see `src/probe/metrics.py`.
- **Layout:** interfaces live in `src/probe/` (backends, verifier, perf, outcome, metrics);
  corpus schema is the day-one contract with Part A in `src/probe/schema.py`.
