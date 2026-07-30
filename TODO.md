# TODO — toward a complete research story

Phases 1–2 established a working, un-gameable reward pipeline and a saturating best-of-K baseline
(~23–28% of functions beaten past `-O3`, ~1.4×). What's left to make this paper-grade and to reach
the actual claim (a model that *learns* to beat that baseline). Ordered roughly by leverage.

> **Phase 3 is now planned:** platform decision (Fireworks RFT + Eval Protocol), the
> Fireworks-GPU / Debian-VM split, and the per-person work split live in
> **[docs/phase3/README.md](docs/phase3/README.md)**. Items below marked **[P3-blocking]** are on its
> critical path — do those first. Start with the `qwen3-8b` baseline in that doc: if a <16B model has
> no prior, GRPO has no gradient and nothing else here matters.

## 1. Bigger, loop-rich, realistic corpus  ← highest priority · **[P3-blocking]**

The current corpora are small and skew tiny/loop-free (64 functions; several buckets have n=1), so the
numbers are directional, not benchmark-grade.

- [ ] Build the real corpus on the **Debian VM** (native linux-gnu, no header-skip): the
      llvm-test-suite path via `./scripts/fetch-corpus.sh` → `build_corpus --max-functions 800+`.
      GRPO over 64 functions with n=1 buckets will just memorize, so this gates Phase 3.
- [ ] Add **real-world code**: functions mined from permissively-licensed GitHub C/C++/Rust repos, not
      just the test-suite (distribution TTRL would actually be deployed on).
- [ ] **Balance the buckets** — deliberately oversample loops and 50–150 / >150-instr functions so
      each bucket has enough n to report a stable number.
- [ ] Target ~500–1,000 deduped functions; publish the corpus + build recipe for reproducibility.

## 2. Handle loops — the central limitation

Loops have the biggest speedups but **both** halves of the oracle are unreliable there: Alive2 verifies
only ~4% of loops (bounded unrolling), and `llvm-mca` ranks loops correctly only ~69% (can't see trip
counts). Today the honest scope is loop-free.

- [ ] **Loop-aware speed metric:** use `--perf timing` (real wall-clock, handles loops) as the scorer
      for loop functions; validate at scale on a quiet Linux box.
- [ ] **Bounded-unroll acceptance tier:** treat Alive2's bounded-equivalence as a weaker-but-usable
      reward for loops, clearly labeled vs full proofs.
- [ ] Decide the paper's stance: scope to loop-free (clean, weaker claim) vs. a tiered-evidence reward
      (broader, needs a hacking audit).

## 3. Speed metric validation & hardening

- [ ] Run `timing_validation` on the **big corpus on a quiet Linux box** for paper-grade
      mca-vs-wall-clock numbers (bootstrap gave 100% wall-clock vs 90.6% mca, Spearman 0.51).
- [ ] Consider **real timing as the primary reward**; keep `code_size_bytes` as a cheap secondary.
- [ ] Quantify how often mca's `verified_faster` calls survive wall-clock (the `--rewrites` audit) on
      a real run — the reviewer-critical number.

## 4. Model & prompt coverage

- [ ] **`qwen3-8b` best-of-K baseline · [P3-blocking, do first]** — the RFT-free tier is <16B, so this
      is both the Phase 3 go/no-go (no prior ⇒ all-zero GRPO groups ⇒ no gradient) and the mandatory
      adapted-vs-base comparison point. Command + fallbacks: [docs/phase3](docs/phase3/README.md).
- [ ] **Multiple models / sizes:** Qwen2.5-Coder (7B/32B), DeepSeek, GPT, Claude — establish which have
      a real prior (SLM-vs-LLM is a known contrast; small models fail on IR syntax).
- [ ] **Format ablation:** `--format ir` (raw IR) vs `--format c` (emit C, lower with clang) — the C
      path sidesteps most IR-parseability failures; measure the coverage trade-off.
- [ ] **Multi-turn repair:** feed Alive2's counterexample back for one retry turn; measure `solve@K`
      lift (previews the multi-turn value for Phase 3).
- [ ] `--include-o3` ablation: does showing the `-O3` starting point help or anchor?

## 5. Phase 3 — the TTRL loop (the actual claim)

**Plan, platform decision, and work split: [docs/phase3/README.md](docs/phase3/README.md).** Summary:
Fireworks RFT (free <16B, we have credits) drives the policy; the reward box runs **unchanged** on the
Debian VM behind an Eval Protocol `RemoteRolloutProcessor` `/init` endpoint. The oracle, not the GPU,
is the throughput bottleneck.

- [ ] **Reward shaping spec:** outcome → scalar in [0,1]. `verified_no_gain` must score **0** or we
      reopen the output-the-input-unchanged cheat; proposal is `1 - 1/speedup` for `verified_faster`.
      Decide before coding — it's a reviewer-visible choice tied to §6.
- [ ] `src/probe/reward.py` (+ tests) and `scripts/rft/env_server.py` (the `/init` service). Prove the
      contract against `--backend mock --verifier stub` before wiring the real oracle.
- [ ] Expose the env server over HTTPS (Fireworks dials *in*; needs a tunnel behind NAT) and bound
      rollout concurrency — they drive it, our box has 14 GB.
- [ ] Wrap the Phase 2 reward (unchanged) in an **online RL loop** (GRPO-style, LoRA rank ≤32)
      over an unlabeled function stream.
- [ ] **Adapted-vs-base at equal K:** the headline — does the fine-tuned model clear best-of-K at the
      same sample budget, ideally shifting the whole K-curve up? (Base = the §4 `qwen3-8b` run, not
      the deepseek Phase 2 numbers.)
- [ ] **Per-function dynamics:** flip rates, mode-collapse indicators, extinction-window analysis.
- [ ] **Held-out generalization:** does adapting on one code distribution transfer to another?

## 6. Reward-hacking audit (a core selling point)

- [ ] Empirically show the reward can't be gamed: proof-gated reward vs weaker evidence tiers; confirm
      the two cheats (wrong-but-fast, unchanged) stay at reward 0.
- [ ] Investigate **spurious counterexamples**: `llvm-extract` pulls a function out of context, so the
      `-O3` copy can specialize on callers/globals and diverge from `-O0` — some counterexamples are
      artifacts, not real miscompiles. Quantify and filter.

## 7. Engineering / infra

- [ ] **Parallelize Alive2** (the throughput bottleneck) and **cache verdicts** by (src, tgt) hash.
      **[P3-blocking]** — not cleanup: at `timeout_s=30` × G=16 rollouts an RL step can sit in Z3 for
      minutes, and the policy re-emits near-identical rewrites constantly, so the cache is a large
      multiplier. On the VM cap at ~6 memory-limited workers: 14 workers on 14 GB will hit swap and
      silently turn proofs into `timeout` verdicts, which corrupts the reward signal. **Report
      verified-rewrites/sec** — that number sets the rollout ceiling and the budget.
- [ ] Add a `failed_to_prove` outcome (Alive2 abstains ≠ tool error ≠ proven-different) — Part A flagged
      this; it makes the reward distribution honest. Shared-schema change; coordinate both workstreams.
- [ ] Cost/latency tracking per run (tokens, API $, Alive2 seconds) for the RL-loop budget.

## 8. Paper positioning

- [ ] Related-work pass: STOKE / Souper (classical superopt), PIE / "performance-improving edits"
      (LLM, test-based), RLVR (test/answer rewards), Alive2 (the tool). Sharpen the novelty:
      **formal-proof reward for LLM superoptimization + test-time adaptation.**
- [ ] Pick a venue shape (MLSys / PL / strong workshop) and decide the fallback: if Phase 3's lift is
      marginal, a "feasibility + measurement" paper on proof-gated rewards still stands on Phase 1–2.
