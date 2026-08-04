# TODO — toward a complete research story

Phases 1–2 established a working, un-gameable reward pipeline and a saturating best-of-K baseline
(~23–28% of functions beaten past `-O3`, ~1.4×). What's left to make this paper-grade and to reach
the actual claim (a model that *learns* to beat that baseline). Ordered roughly by leverage.

> **Phase 3 is now planned:** platform decision (Fireworks RFT + Eval Protocol), the
> Fireworks-GPU / Debian-VM split, and the per-person work split live in
> **[docs/phase3/README.md](docs/phase3/README.md)**. Items below marked **[P3-blocking]** are on its
> critical path — do those first. Start with the `qwen3-8b` baseline in that doc: if a <16B model has
> no prior, GRPO has no gradient and nothing else here matters.

**The LLVM run box** (corpus build, Alive2, mca, timing): Debian, **14 threads / 14 GB RAM / 4 GB swap
/ 1.3 TB disk** (8 GB VRAM — unused; the policy lives on Fireworks). Everything in this file that
touches LLVM must fit that envelope. The two limits that actually bind:

- **RAM, not cores.** Z3 can take 1–2 GB+ per `alive-tv`; **cap parallelism at ~6 workers**, memory-
  limited. Touching the 4 GB swap turns proofs into `timeout` verdicts *silently* — that corrupts the
  reward signal, so it must fail loudly instead (§7).
- **Set `PROBE_TARGET` *and* `PROBE_CPU` together — they must match.** Defaults are
  `aarch64-linux-gnu` + `cortex-a72` (`tools.py:26-27`), an ARM pair chosen as a *macOS* workaround.
  Setting only `PROBE_TARGET=x86_64-linux-gnu` leaves `-mcpu cortex-a72` against an x86 triple, and
  **`llc` hard-fails** (`LLVM ERROR: 64-bit code requested on a subtarget that doesn't support it`) →
  `McaPerf.score` returns `None` → **`mca_cycles_o3: null` on every record**, with no error from
  `build_corpus`. On x86_64 pick the real family from `lscpu` (`znver3`, `skylake`, …); generic
  `x86-64` works but has a thin scheduling model. The pair is part of the corpus's identity — cycle
  counts differ by 2–3× across CPU models — so **record both, and never compare mca numbers across
  them.**

Disk is a non-issue (LLVM+Alive2 ~5.4 GB, llvm-test-suite clone ~1 GB, corpus tens of MB).

## 1. Bigger, loop-rich, realistic corpus  ← highest priority · **[P3-blocking]**

The current corpora are small and skew tiny/loop-free (64 functions; several buckets have n=1), so the
numbers are directional, not benchmark-grade.

Run order on the VM (details + one-time setup: [docs/phase3](docs/phase3/README.md)):

- [x] **Toolchain first.** `01-prereqs.sh` → `02-build-alive2.sh` → `source scripts/alive2/env.sh`;
      verify `alive-tv --version` and `$LLVM_BIN/llvm-mca --version`. Export `PROBE_TARGET` (above).
      *Done on the VM 2026-08-02.*
- [x] **Build the whole thing, uncapped.** `./scripts/fetch-corpus.sh -q` → `build_corpus --with-mca`
      **without** `--max-functions`. Rationale below — the cap is not a sampling strategy. Expect most
      of llvm-test-suite/SingleSource to fail `clang` standalone (missing harness headers); those are
      skipped silently, which is fine, but **log the skip rate** so we know the yield.
      *Run on the VM 2026-08-02 — **record count + bucket histogram not yet captured here** (§1 note).*
- [x] **Stratified sampler + filters written** (`probe/make_corpora.py`, 9 tests). Emits **both**
      corpora in one pass: `report` (stratum-capped, balanced) and `train` (oracle-verifiable,
      artifact-filtered). `--max-functions` was never a sampling strategy — it truncates in
      `sorted(rglob("*.c"))` order (`build_corpus.py:180`), yielding the alphabetically-first N.
- [x] **Ran on the VM 2026-08-03: train 1543 · report 695, and perf sanity went 88.0% → 96.6%**
      (Phase 1 reference: 98%). The `main`/libc diagnosis held.
- [x] **Re-ran with the intrinsic fix + 300/stratum cap: train 577 · report 678, perf sanity 95.7%.**
      Lower than the uncapped 96.6% purely by denominator (the cap removed the cleanest 943 tiny
      functions; absolute inversions fell 52 → 25). The 25 that remain are two documented llvm-mca
      limits — call-swamped scores (~19) and loop unrolling vs `--iterations=1` (~6) — not corpus
      defects. **§1 is done; the corpus is good enough to train on.**
- [ ] **Fix the train corpus shape — the oracle filter made the skew worse.** Verified-only is
      **81% tiny/loop-free** (1251/1543) vs 66% raw, because Alive2 verifies exactly the functions
      Phase 2 found are *already optimal at -O3*. No headroom ⇒ reward 0 on every rollout, same dead
      weight as unverifiable. `--train-cap-per-bucket` (default 300) is a blunt stopgap; the real
      filter is **headroom data from the `qwen3-8b` baseline** — keep functions that ever produced a
      `verified_faster` sample. Do this *after* §4's baseline run, as its second output.
- [ ] Add **real-world code**: functions mined from permissively-licensed GitHub C/C++/Rust repos, not
      just the test-suite (distribution TTRL would actually be deployed on).
- [ ] Target ~500–1,000 deduped functions; publish the corpus + build recipe + **`PROBE_TARGET` and
      LLVM version** for reproducibility.
- [x] Sanity-check the result before trusting it: `verify_corpus.py`, plus the Phase 1 Part A
      perf-sanity check (is `-O0` ≥ `-O3` cycles?) on the new corpus. *Both run on the VM 2026-08-02;
      **verdicts vs Phase 1's 79% / 98% not yet compared** — see the open numbers below.*

**Open — the four numbers that decide what happens next.** The commands have been run; their outputs
live only on the VM and are not recorded anywhere in this repo. Until they are, §1 is "executed" but
not "known", and the items below can't be scoped:

- [x] **Record the corpus build result.** 2026-08-02: **3414 records, 3236 scored (94.8%)**, only 3
      real mca failures (the target/cpu pair was right). Distribution is still **66% tiny loop-free**,
      but every bucket now clears n=33 and small loops went 12 → 213. Table + reading:
      [docs/phase3](docs/phase3/README.md#corpus-build-result--2026-08-02). Still unrecorded: the skip
      rate (`.c` files walked vs records produced).
- [x] **`verify_corpus` on the new corpus: 51.9% verified (1680/3239)** vs Phase 1's 79% — a
      *composition* difference, not a regression. Loop-free 58%, loops 14.1% (Phase 1's 4% was 1/23).
      **1,680 functions form the train-corpus pool.** Oracle throughput: 0.62 s mean/check serial →
      ~1.6 s per G=16 step at 6 workers, so the oracle is **not** the feared bottleneck (excluding
      >150-instr functions, p90 22 s). Full table:
      [docs/phase3](docs/phase3/README.md#oracle--perf-sanity-on-the-new-corpus--2026-08-02).
- [x] **`perf_sanity` on the new corpus: 88% monotonic (2847/3236)** vs Phase 1's 98%. Explained, not
      a regression: gcc-c-torture is full of **libc reimplementations** (`-O3` turns a hand-written
      `strlen` loop into a *call to `strlen`* — `O0=15 → O3=110`) and **driver `main`s**. Details:
      [docs/phase3](docs/phase3/README.md#oracle--perf-sanity-on-the-new-corpus--2026-08-02).
- [ ] **Filter the corpus accordingly:** drop `main`, and drop functions whose `-O3` IR calls a
      function the `-O0` IR does not. Re-measure perf sanity after; it should approach Phase 1's 98%.
      Same root cause as §6's spurious-counterexample audit (`llvm-extract` strips context).
- [ ] Commit these into `docs/phase3/` (or a `docs/phase1/partA-findings.md` addendum) with the
      `PROBE_TARGET`, LLVM version, and llvm-test-suite commit, so the corpus is reproducible.

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
      **The 2026-08-03 corpus work strengthened this a lot:** mca is unreliable on call-containing
      functions (one call ≈ 104 cycles swamps everything; **24% of the train corpus**) *and* on loops
      (unrolling vs `--iterations=1`) — between them, most of what has interesting headroom. Every
      residual perf-sanity inversion is one of these two.
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
- [ ] **Minimum relative speedup before `verified_faster`.** llvm-mca prices one call at **~104
      cycles** vs ~4 for plain arithmetic (measured), so any call-containing function's score is
      swamped by a constant and a 1-cycle change reads as a 1.009× "win" — a free, meaningless reward
      the policy can farm. Require e.g. ≥5% before granting `verified_faster`, or score this
      population with `--perf timing`. **Also re-check whether Phase 2's coverage numbers include
      such artifacts.**
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
      artifacts, not real miscompiles. Quantify and filter. **Now measurable: 421 counterexamples
      (13.0%) on the new corpus, up from Phase 1's 5%, 283 of them in the ≤20 loop-free bucket.** 421
      real LLVM miscompiles is not credible, so this is close to a pure artifact population — a good
      thing, because it means a filter can recover them rather than a scope reduction.

## 7. Engineering / infra

- [ ] **Parallelize Alive2** (the throughput bottleneck) and **cache verdicts** by (src, tgt) hash.
      **[P3-blocking]** — not cleanup: at `timeout_s=30` × G=16 rollouts an RL step can sit in Z3 for
      minutes, and the policy re-emits near-identical rewrites constantly, so the cache is a large
      multiplier. On the VM cap at ~6 memory-limited workers: 14 workers on 14 GB will hit swap and
      silently turn proofs into `timeout` verdicts, which corrupts the reward signal. **Report
      verified-rewrites/sec** — that number sets the rollout ceiling and the budget.
- [ ] **`build_corpus --jobs N`** — `build_records` is fully serial (`build_corpus.py:151`): two
      `clang` invocations per file, then `llvm-extract` + `llvm-mca` per function, all on one core.
      It uses 1 of the VM's 14 threads. Per-file work is independent, so a process pool over the
      file list is a near-linear win on the uncapped build. Keep dedup (`_norm_hash`) and
      `function_id` ordering deterministic so the corpus stays reproducible.
- [x] **Fail loudly when the oracle is absent.** `verify_corpus` now aborts after 20 consecutive tool
      errors with a message pointing at `env.sh` — the 2026-08-02 run completed and reported
      `verified_rate 0.00` as if it were a finding. Still open: the same guard in `run_probe`
      (refuse to start when `--verifier alive` is asked for and `alive-tv` doesn't resolve).
- [x] Added a `failed_to_prove` outcome (Alive2 abstains ≠ tool error ≠ proven-different) — Part A
      flagged this; it makes the reward distribution honest. Shared-schema change, mirrored in
      `docs/phase1/partB-plan.md` per the `schema.py` rule. **Also split `error` out of
      `invalid_syntax`**: a tool failure was being reported as the model emitting garbage. All three
      are reward 0, so no metric moves — but `invalid_syntax` numbers are no longer comparable with
      pre-2026-08-03 runs, including [ir-robustness.md](docs/phase2/ir-robustness.md)'s 61%→10%.
- [ ] Cost/latency tracking per run (tokens, API $, Alive2 seconds) for the RL-loop budget.

## 8. Paper positioning

- [ ] Related-work pass: STOKE / Souper (classical superopt), PIE / "performance-improving edits"
      (LLM, test-based), RLVR (test/answer rewards), Alive2 (the tool). Sharpen the novelty:
      **formal-proof reward for LLM superoptimization + test-time adaptation.**
- [ ] Pick a venue shape (MLSys / PL / strong workshop) and decide the fallback: if Phase 3's lift is
      marginal, a "feasibility + measurement" paper on proof-gated rewards still stands on Phase 1–2.
