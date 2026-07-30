# T3RL-LLVM

**A model that teaches itself to out-optimize the compiler — graded by the compiler's own proof
tools, with no training labels.**

## What it does

Compilers (`-O2`/`-O3`) optimize code using a fixed catalog of safe rewrites; a language model can
search more creatively. So: take a function in **LLVM IR** (the intermediate language C, C++, and
Rust all compile to) and ask a model to *rewrite it to do exactly the same thing, but faster*. This
is **superoptimization**.

The catch that normally kills this: grading "exactly the same thing" is hard — tests miss edge
cases, and labeled data doesn't exist. This project sidesteps that by using **formal compiler tools
as the grader**, making the reward label-free and very hard to game — which is what unlocks
unsupervised / test-time RL (TTRL).

## How the reward works

> **reward = (provably correct) × (measurably faster)**

- **Alive2 — the reward oracle (correctness).** A *translation validator*: given the original
  function and the model's rewrite, it uses a theorem prover to either **prove** they return
  identical outputs for *every* input, or hand back a concrete input where they differ. A rewrite
  scores only if Alive2 proves equivalence — no test suite to miss cases, no labels. This blocks the
  *wrong-but-fast* cheat.
- **llvm-mca — the speed scorer (magnitude).** It statically models a CPU pipeline to estimate how
  many **cycles** a code sequence takes. Once Alive2 says "correct," mca says "how much faster,"
  measured against `-O3`. This blocks the *output-the-input-unchanged* cheat (zero speedup → zero
  reward).

The two easy ways to fool the grader cancel out: to score, a rewrite must be **both** proven correct
**and** faster — rare, and hard to fake. (`llvm-mca` is a proxy — accurate loop-free, weaker on
loops; `--perf timing` measures real wall-clock when a number must survive review.)

## Status

| phase | question | result |
|---|---|---|
| **1** — viability | Is the oracle usable? Does the model have a prior? | **GO**, scoped to loop-free — [docs/phase1](docs/phase1/README.md) |
| **2** — best-of-K baseline | How good is verify-and-select, before any training? | ~**23–28%** of functions beaten past `-O3`, ~**1.4×** mean speedup — [docs/phase2](docs/phase2/README.md) |
| **3** — TTRL loop | Can the model *learn* to beat that baseline? | planned — [docs/phase3](docs/phase3/README.md) |
| 4 — writeup | | future |

Docs index + terminology: [docs/README.md](docs/README.md).

## Quickstart (offline — no keys, no LLVM)

Requires [uv](https://docs.astral.sh/uv/). Runs the whole pipeline against stubs:

```bash
uv sync --extra dev
uv run pytest
uv run python -m probe.run_probe --corpus data/bootstrap --backend mock \
    --k 8 --verifier stub --perf stub
```

Prints a per-bucket `solve@K` table; writes artifacts to `results/` (git-ignored).

## Real run

Three ingredients: the **Alive2 oracle**, the **LLVM 21 toolchain**, and a **model backend**.

```bash
# 1. Build Alive2 + LLVM 21 (~15 min; full guide: docs/phase1/alive2-build.md)
git submodule update --init --recursive
./scripts/alive2/01-prereqs.sh && ./scripts/alive2/02-build-alive2.sh
source scripts/alive2/env.sh          # exports ALIVE_TV + LLVM_BIN (llvm@21)

# 2. Build a corpus (uses llvm@21 from env.sh, so it matches the oracle)
uv run python -m probe.build_corpus --src data/c_sources \
    --out data/corpus/corpus.jsonl --with-mca

# 3. Sample K rewrites per function, verify with Alive2, score with mca (Fireworks example)
export FIREWORKS_API_KEY=...
uv run python -m probe.run_probe --corpus data/corpus/corpus.jsonl --backend api \
    --model accounts/fireworks/models/deepseek-v4-pro \
    --base-url https://api.fireworks.ai/inference/v1 --api-key-env FIREWORKS_API_KEY \
    --k 16 --verifier alive --perf mca --out results/run1

# 4. Best-of-K baseline (Coverage@K / MeanSpeedup@K)
uv run python -m probe.phase2_baseline --rewrites results/run1 \
    --corpus data/corpus/corpus.jsonl --ks 1,2,4,8,16
```

**Other backends** (any OpenAI-compatible API): swap `--model` / `--base-url` / `--api-key-env` —
e.g. Together (`https://api.together.xyz/v1`, `Qwen/Qwen2.5-Coder-32B-Instruct`), OpenAI, OpenRouter.
Add `--no-supports-n` if the provider ignores `n` (samples K sequentially). Local GPU:
`--backend vllm --model <hf-id>` (needs `uv sync --extra local`). Presets in `configs/models.yaml`.

**Partial signal without the full stack:** a model key alone (`--verifier stub --perf stub`) gives
IR-validity numbers; `brew install llvm` adds real `mca` scoring; only Alive2 unlocks real
equivalence verdicts.

## Notes

- **Real timing instead of mca:** `--perf timing` measures wall-clock ns on the native host (macOS or
  Linux). Audit mca against it: `uv run python -m probe.timing_validation --corpus <corpus>
  --rewrites <run>`. See [docs/phase2/timing-validation.md](docs/phase2/timing-validation.md).
- **Toolchain consistency:** the corpus, `alive-tv`, and mca must all use the same LLVM (source
  `env.sh` for llvm@21). Apple's system clang emits IR Homebrew's tools reject; `tools.py` resolves
  everything from one install (`$LLVM_BIN` / `$PROBE_TARGET` to override). The oracle toolchain costs
  ~5.4 GB on disk; to reclaim it see
  [Uninstalling](docs/phase1/alive2-build.md#uninstalling-reclaiming-54-gb).
- **IR is normalized** before verification (`ir_utils.sanitize_module`) — models emit full modules /
  `...` placeholders Alive2 would reject; this doubled coverage on a real run
  ([docs/phase2/ir-robustness.md](docs/phase2/ir-robustness.md)).
- **Terminology:** **N** = rewrites sampled per function (`--k`); **K** = the selection budget in
  `@K` metrics (K ≤ N). `solve@K` / `Coverage@K` = fraction of functions with ≥1 verified-faster
  rewrite.

## Repo layout

`src/probe/` is the pipeline: `backends/` (mock/api/vllm), `verifier.py` (Alive2), `perf.py` +
`timing.py` (scorers), `outcome.py` (classify a rewrite), `bestofk.py` + `phase2_baseline.py`
(metrics), `ir_utils.py` (IR normalization), `build_corpus.py`. The corpus schema — the day-one
contract between components — is in `schema.py`. Design docs and findings live in `docs/`.
