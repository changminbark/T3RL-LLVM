# Phase 1 · Part B — Model Capability Probe

> **Shared sync doc.** Both teammates' Claude sessions read this file to stay aligned.
> This is **Part B (Model Capability Probe)**. Person A's Part A doc lives alongside it as
> `docs/phase1-partA-plan.md`.
>
> **For Person A's session:** the sections you care about are **Shared interface**,
> **Workstream A responsibilities (what B needs from you)**, and **Go/no-go (both halves)**.
> Everything under "Repo scaffold" and "Work steps" is Person B's implementation and won't
> block you — but the interface contracts there are the exact shape B will call your harness
> with, so confirm them.

## Context

We are testing the viability of an unsupervised **TTRL superoptimizer**: a model that rewrites
LLVM IR functions to be provably-equivalent-but-faster, using Alive2 (a theorem-prover-based
translation validator) as an un-gameable reward oracle and `llvm-mca` as the speed scorer.
No RL or training happens in Phase 1 — it is inference-only viability testing.

The project has two independent kill risks, de-risked in parallel by two people:

- **Workstream A (Person A):** Is the *verifier* usable? (Alive2 verdict rate + latency on
  real functions.) → produces the corpus + Alive2 harness.
- **Workstream B (Person B, this doc):** Does *any accessible base model* have a nonzero prior
  on the task? TTRL amplifies existing ability; if the base model almost never produces a
  verified-and-faster rewrite, there is nothing to reinforce and the project stops here.

**Part B deliverable:** `solve@K` per model per size/loop bucket, the chosen generation format
(raw IR vs. C→IR), the full outcome distribution (the future reward distribution), and reusable
prompt templates.

### Key decisions locked in

- **Model backend:** undecided → all generation goes behind one `LLMBackend` interface with two
  implementations (local vLLM, hosted API). Nothing else in the code knows which is used.
- **Person A's corpus + harness:** not ready → B bootstraps on ~50 hand-picked functions
  (Alive2's own test suite) and codes against a **stubbed `VerifierHarness`** matching the agreed
  JSONL schema. When A's real CLI lands, B swaps the stub for a subprocess call — no other code
  changes.

---

## Shared interface (agree with Person A on day one)

Corpus is JSONL, one record per function:

```json
{
  "function_id": "str (unique)",
  "src_ir": "str (LLVM IR text, the -O0 source function)",
  "source_lang": "c | llvm | ...",
  "n_instructions": 0,
  "has_loops": false,
  "o3_baseline_ir": "str (the -O3 optimized IR — the 'beat the compiler' baseline)",
  "mca_cycles_o3": 0.0
}
```

Harness verdict (what Person A's CLI returns for a `(src, tgt)` pair):

```json
{ "status": "verified | counterexample | timeout | unsupported | error",
  "counterexample": "str | null",
  "wall_time_s": 0.0 }
```

Performance score (mca): `{ "mca_cycles": float, "code_size_bytes": int }` for any IR.

---

## Workstream A responsibilities (what B needs from you) — for Person A's session

Person A owns the verifier & corpus. B is a pure consumer. To let B build in parallel against
stubs, A should deliver, in priority order:

1. **The JSONL schema, frozen (day one).** Exactly the `CorpusRecord` fields above. Confirm how
   `n_instructions` and `has_loops` are computed (e.g., count IR instructions post-`-O0`;
   loop = presence of a back-edge / `llvm.loop` metadata) so buckets are consistent across both
   sessions.
2. **A harness CLI** B can shell out to: `alive-tv`-based, taking `src.ll tgt.ll` + timeout,
   emitting the `Verdict` JSON above (`verified | counterexample | timeout | unsupported | error`,
   plus `wall_time_s`). Sandboxed. B wraps this as `AliveCliVerifier` (identical interface to the
   `StubVerifier` B builds first).
3. **A perf CLI/wrapper** over `llvm-mca` emitting the `PerfScore` JSON (`mca_cycles`,
   `code_size_bytes`) for any IR. B wraps this as the real `perf.py`; B uses `StubPerf` until it
   exists.
4. **The corpus JSONL**, bucketed by size (≤20, 20–50, 50–150 IR instrs) and loop presence,
   target ~500–1,000 functions, with `o3_baseline_ir` + `mca_cycles_o3` populated.

A's own deliverable (per the brief) is the verdict-rate / median-verification-time table per
bucket — that's the *other* half of the go/no-go and is independent of B's code.

**Sync mechanism:** both sessions read the committed `docs/phase1-partB-plan.md` (and
`docs/phase1-partA-plan.md`). When A changes the schema or a CLI's output shape, update the doc
in the same commit so B's session picks it up on next read.

---

## Repo scaffold (Part B)

Dependency & env management: **uv**. `pyproject.toml` declares deps; `uv sync` creates `.venv`
and writes `uv.lock` (committed). Run everything via `uv run …` (e.g.
`uv run python -m probe.run_probe`, `uv run pytest`). `vllm` goes behind an optional extra
(`[project.optional-dependencies] local = ["vllm"]`) so API-only users skip the heavy GPU dep:
`uv sync --extra local`.

```
T3RL-LLVM/
  pyproject.toml               # deps: pydantic, tqdm, httpx, pyyaml, pytest
                               # optional-dependencies.local: vllm (GPU-only)
  uv.lock                      # committed lockfile
  .gitignore                   # .venv/, results/, __pycache__/, *.ll scratch, data/corpus/*
  .python-version              # pin (3.12)
  docs/
    phase1-partB-plan.md       # this file
    phase1-partA-plan.md       # Person A's doc (they own it)
  data/
    bootstrap/                 # ~50 hand-picked functions (Alive2 test cases) as JSONL
    corpus/                    # drop-in for Person A's real corpus JSONL (git-ignored)
  src/probe/
    schema.py                  # pydantic models: CorpusRecord, Verdict, PerfScore, outcomes
    backends/
      base.py                  # LLMBackend ABC: .generate(prompt, k, temperature) -> list[str]
      vllm_backend.py          # local vLLM / HF impl
      api_backend.py           # hosted API impl (OpenAI-compatible)
    prompts.py                 # template builders (format A/B, -O0-only vs +O3, retry-with-CEX)
    extract.py                 # pull IR/C out of completions (fenced ```llvm / ```c blocks)
    lower.py                   # format B: C -> IR via clang -O0 -emit-llvm -S
    verifier.py                # VerifierHarness: StubVerifier + AliveCliVerifier
    perf.py                    # mca scorer wrapper + StubPerf for bootstrap
    outcome.py                 # classify one rewrite -> RewriteOutcome + score
    run_probe.py               # main experiment driver (CLI)
    metrics.py                 # aggregate -> solve@K, outcome distribution, tables
  configs/
    models.yaml                # backend + model id + sampling defaults per experiment
  results/                     # per-run JSONL + summary tables (git-ignored)
  tests/                       # schema/extract/outcome/stub-verifier unit tests
```

### Interface contracts

**`LLMBackend`** (`backends/base.py`)

```python
class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, k: int, temperature: float,
                 max_tokens: int) -> list[str]: ...
```

Selected by `configs/models.yaml`; `run_probe.py` never imports a concrete backend directly
(factory in `backends/__init__.py`).

**`VerifierHarness`** (`verifier.py`)

```python
class VerifierHarness(ABC):
    @abstractmethod
    def check(self, src_ir: str, tgt_ir: str, timeout_s: int) -> Verdict: ...

class StubVerifier(VerifierHarness):      # bootstrap: normalized-IR equality only
class AliveCliVerifier(VerifierHarness):  # subprocess -> alive-tv, parse -> Verdict
```

The stub lets the whole pipeline run end-to-end before Alive2 exists (it can only say `verified`
on trivially-equal IR or `unsupported` otherwise — enough to exercise plumbing and metrics).

**`RewriteOutcome`** (`outcome.py`) — one label per generated rewrite, the future reward
distribution:
`invalid_syntax` · `counterexample` · `timeout` · `unsupported` · `verified_no_gain` ·
`verified_faster`

Classification pipeline for a single rewrite:

1. extract IR (format A) or C→lower→IR (format B); fails to parse/assemble → `invalid_syntax`.
2. `verifier.check(src_ir, rewrite_ir)` → map `timeout`/`unsupported`/`counterexample` through.
3. if `verified`: compare `perf.score(rewrite_ir).mca_cycles` vs `mca_cycles_o3` →
   `verified_faster` (strictly fewer cycles than **O3**) else `verified_no_gain`.

---

## Work steps (Part B)

### Step 0 — Scaffold + schema (unblocks everything)
- `uv` project: author `pyproject.toml`, `.gitignore`, `.python-version`; `uv sync` → `.venv` +
  `uv.lock` (commit the lock).
- Stand up repo layout, `schema.py`, backend factory, `StubVerifier`, `StubPerf`.
- Commit this doc as the shared sync artifact.
- Confirm the JSONL schema with Person A in writing before building consumers.

### Step 1 — Bootstrap corpus (~50 functions)
- Pull ~50 small functions from Alive2's test suite / hand-pick; store as bootstrap JSONL matching
  the schema (leave `o3_baseline_ir`/`mca_cycles_o3` best-effort — generate locally with
  `clang -O3` + `llvm-mca` once installed, else mark null and gate O3 comparisons).
- Purpose: unblock all prompting/plumbing work before Person A's corpus lands.

### Step 2 — Generation format A/B decision
- Implement `prompts.py` format A (emit LLVM IR) and format B (emit C, then `lower.py` → IR).
- Run both over the bootstrap set; measure **syntactic validity rate** (fraction assembling with
  `llvm-as`/parsing cleanly).
- Decision rule: if format A validity is poor even for a strong open model (Qwen-coder-class),
  format B is the default carrier for the rest of Phase 1.

### Step 3 — `solve@K` experiment (the headline number)
- For each function: sample **K = 8–16** rewrites at **temperature 0.8–1.0**, classify each via
  the outcome pipeline.
- Report per size bucket (≤20, 20–50, 50–150 IR instrs) and by loop presence:
  - **`solve@K`** = fraction of functions with ≥1 `verified_faster` rewrite. *(primary metric)*
  - **Outcome distribution** = mean fraction of the K rewrites in each label. *(reward sparsity)*
  - Wall-clock per function (sampling + verification) for later RL-loop cost estimation.

### Step 4 — Cheap prompt ablations
- Context ablation: `-O0` src only **vs.** also include `-O3` output as a starting point.
- One-turn repair: feed Alive2's counterexample back for a single retry turn. Measure `solve@K`
  lift.

### Step 5 — Swap stub → real harness
- When Person A's CLI is ready, implement `AliveCliVerifier` and real `perf.py` against
  `llvm-mca`.
- Re-run Steps 3–4 on the real corpus. No changes above the verifier/perf boundary.

---

## Go/no-go criteria (both halves — pre-register before seeing results)

**Workstream B (Person B):**
- **Model prior:** some model reaches **`solve@16` ≥ ~5–10%** on the loop-free buckets. Below
  this, the prior is too weak to amplify → recommend light SFT warm-up or a different base model
  before Phase 2.
- Record the outcome distribution regardless — even a failing `solve@K` tells us *where* rollouts
  die (mostly `invalid_syntax`? mostly `verified_no_gain`?), which changes the fix.

**Workstream A (Person A):**
- Alive2 returns a verdict (not timeout/unsupported) on **≥ ~60%** of functions in at least the
  two smaller buckets. Else: shrink function size or accept a hybrid verifier and rescope.
- **Median verification time ≤ ~30s**, so RL-loop reward calls are affordable.

Treat all thresholds as pre-registered calibration points, decided before results, so a weak
signal isn't rationalized.

---

## Verification

- **Unit:** `uv run pytest tests/` — schema round-trips, extraction on messy completions,
  outcome classification against synthetic verdicts, stub verifier.
- **End-to-end on stubs (no LLVM/Alive2 needed):**
  `uv run python -m probe.run_probe --corpus data/bootstrap --backend <api|vllm> --k 8 --verifier stub`
  runs the whole pipeline and emits a summary table — proves plumbing before external tools exist.
- **End-to-end real:** same command with `--verifier alive --perf mca` once tools + Person A's
  harness are installed; sanity-check that `llvm-mca` ranks `-O0` cycles > `-O3` cycles on a few
  functions before trusting the scorer.
- **Backend swap check:** identical invocation with `--backend vllm` vs `--backend api` produces
  the same schema of outputs — confirms the abstraction holds.

## Open items to confirm with Person A
- Lock the JSONL schema wording (esp. how loop presence and instruction count are computed).
- Confirm which concrete model(s) to prioritize first (Qwen2.5-Coder class) so `configs/models.yaml`
  has real ids.
- `PerfScore` already carries `code_size_bytes` (secondary metric) to avoid a later schema change.

## Part A status (Person A confirmation)

- `alive-harness` CLI implemented (`src/probe/alive_harness.py`, console-script entry point).
  Contract matches `AliveCliVerifier`: `alive-harness <src.ll> <tgt.ll> --timeout <s>` ->
  last-line `Verdict` JSON. Missing `alive-tv` -> `Verdict(error)`, never crashes.
- Corpus builder scaled to per-function extraction (`build_corpus.build_records`, `--max-functions`),
  deduped, bucketed. `function_id` is `<file-stem>.<func>`.
- Verdict-rate experiment: `python -m probe.verify_corpus --corpus <dir> --timeout 30`
  -> `results/verdict_rates.{json,txt}` (the go/no-go table).
- Perf sanity-check: `python -m probe.perf_sanity --corpus <dir> --perf mca`.
- All Part A logic unit-tested offline (no LLVM/Alive2 needed). Real corpus + real verdict table
  pending a later session that installs LLVM + builds Alive2.
