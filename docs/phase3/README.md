# Phase 3 — TTRL loop (plan)

Wrap the **Phase 2 reward box, unchanged**, in an online GRPO+LoRA loop over an unlabeled function
stream. The headline claim: **adapted-vs-base at equal K** — does the fine-tuned model clear the
best-of-K baseline (~28% coverage / 1.38× over `-O3`) at the same sample budget?

Status: **not started.** This is the plan, the platform decision, and who does what.

## The one thing to settle first (blocking)

**Does a small, RFT-eligible model have a prior at all?** Phase 2's numbers come from
`deepseek-v4-pro`; the free RFT tier only covers models <16B (`qwen3-8b`). If `qwen3-8b`'s
verified-faster rate is near zero, **every GRPO group is all-zero reward and there is no gradient** —
the loop cannot bootstrap, and no amount of infra fixes it.

The test needs no new code, and its output is *also* the required Phase 3 baseline (adapted-vs-base
only means something against the same base model):

```bash
uv run python -m probe.run_probe --corpus data/corpus/corpus.jsonl --backend api \
    --model accounts/fireworks/models/qwen3-8b \
    --base-url https://api.fireworks.ai/inference/v1 --api-key-env FIREWORKS_API_KEY \
    --k 16 --verifier alive --perf mca --out results/qwen3-8b-base
uv run python -m probe.phase2_baseline --rewrites results/qwen3-8b-base \
    --corpus data/corpus/corpus.jsonl --ks 1,2,4,8,16
```

**Go/no-go:** enough functions with ≥1 `verified_faster` sample to give GRPO non-degenerate groups.
If it's too low, in order of preference: (1) `--format c` (emit C, lower with clang — sidesteps IR
parseability, see [ir-robustness](../phase2/ir-robustness.md)); (2) pay on-demand GPU rates for a
≥16B model; (3) SFT-warmstart on verified rewrites harvested from the deepseek Phase 2 runs.

## Two workloads, two machines

Phase 3 splits cleanly, and the split is what makes it affordable:

| | what it does | where |
|---|---|---|
| **A. Policy** | sample G rollouts/function, LoRA/GRPO update | Fireworks (GPU) |
| **B. Reward oracle** | `sanitize` → `alive-tv` (Z3) → `llvm-mca` | Debian VM (CPU only) |

**The oracle is the throughput bottleneck, not the GPU.** With `timeout_s=30`
(`src/probe/verifier.py:25`) × G=16 rollouts, one RL step can sit in Z3 for minutes. That makes the
Alive2 parallelism + verdict cache (TODO §7) a **Phase 3 prerequisite**, not cleanup — the policy
re-emits near-identical rewrites constantly, so caching by `(src, tgt)` hash is a large multiplier.

## Platform: Fireworks RFT + Eval Protocol

Chosen because we already have credits and the Phase 2 API path, and **RFT is free for models <16B**
(`qwen3-4b` / `qwen3-8b` are tunable; LoRA rank ≤32, default 8).

A Fireworks-*hosted* evaluator cannot work for us — the reward needs LLVM 21 + `alive-tv` + Z3, and
their sandbox targets lightweight scoring ("evaluations should complete in seconds"). The escape
hatch is **`RemoteRolloutProcessor`**, which delegates rollout execution to an HTTP service we run:

```
Fireworks (policy sampling + LoRA/GRPO update)
   │  POST /init  ─────────────────────────────►   Debian VM (env server)
   │  ◄── sample(model_base_url) ───────────────    ir_utils.sanitize_module
   │                                                verifier.py → alive-tv (Z3)
   │  ◄── Status.rollout_finished(reward) ──────     perf.py → llvm-mca
```

`/init` receives `completion_params`, `messages`, a traced `model_base_url` (call the *current*
policy through it), and correlation metadata. Our code then runs the existing pipeline locally and
signals completion. `outcome.py` becomes the reward function, unchanged. Because `/init` is our code,
multi-turn counterexample repair (TODO §4) drops in later for free.

Reference: [`remote-rollout-processor-hello-world`](https://github.com/eval-protocol/remote-rollout-processor-hello-world)
(minimal contract), [`mcpmark-lite-rft-example`](https://github.com/eval-protocol/mcpmark-lite-rft-example)
(closest precedent: Docker-packaged local env, deterministic verifier over post-rollout *state*, not
assistant text).

Rejected alternatives: **Tinker** (clean fit — reward stays client-side — but no credits and a
different model roster), **Prime Intellect / verl on rented GPUs** (cheapest per GPU-hour, most infra
work; the fallback if Fireworks' constraints bite), **Modal** (best for fanning out the *oracle*,
expensive as a trainer), **OpenPipe/ART** (its headline RULER LLM-judge reward is irrelevant — our
whole thesis is that the reward is a *proof*).

## The Debian VM: yes for the oracle, no for the GPU

Specs: Debian, 14 threads, 14 GB RAM, 4 GB swap, 8 GB VRAM, 1.3 TB disk.

**Good for it.** `scripts/alive2/01-prereqs.sh` already has the apt path (`clang-21 llvm-21-dev`,
`z3`), and `02-build-alive2.sh` builds *only* `alive-tv` — no LLVM from source, so the build is a few
GB of RAM, not 30+. `env.sh` auto-resolves `/usr/lib/llvm-21`. Native linux-gnu also unblocks the
real corpus (no header-skip hacks). 1.3 TB is ample.

**Constraints, in order of how likely they are to bite:**

1. **RAM, not cores, is the limit.** Do *not* run 14 `alive-tv` workers on 14 GB — each Z3 instance
   can take 1–2 GB+, and touching the 4 GB swap silently turns proofs into `timeout` verdicts, which
   **corrupts the reward signal**. Cap at ~6 workers with a per-worker memory limit so an OOM
   surfaces as a clean tool error.
2. **Public reachability.** Fireworks initiates the connection to `/init`, so the VM needs a
   publicly-reachable HTTPS endpoint (Cloudflare Tunnel or Tailscale Funnel behind NAT). It also
   inverts control: *they* drive concurrency, and their checklist assumes autoscaling — we must bound
   it ourselves.
3. **Verify `llvm-21-dev` is RTTI-enabled**, or the alive2 link fails. This is the one build-time
   gotcha.
4. **`PROBE_TARGET` must be set to the VM's real arch.** It defaults to `aarch64-linux-gnu`
   (`tools.py:23`) — that default exists to stop llvm-mca choking on *macOS* asm directives, and it is
   wrong on an x86_64 box. `build_corpus` compiles with `--target=$PROBE_TARGET` and `McaPerf` scores
   at that triple (`perf.py:68`), so getting it wrong silently models a CPU we are not running on.
   `timing.py` is unaffected (it strips target lines and retargets to host), which is precisely why a
   mismatched triple makes the §3 mca-vs-wall-clock audit meaningless. **mca cycles are only
   comparable within one target** — treat the triple as part of the corpus's identity.
5. **`--perf timing` needs a quiet host.** If the VM shares a hypervisor with other tenants,
   wall-clock is noisy. Keep `mca` as the RL reward; use `timing` for offline audits only, pinned
   cores, low load.
6. **8 GB VRAM cannot serve the policy** (a 7–8B model is ~14 GB in bf16 before optimizer state).
   Treat the VM as a dedicated oracle server.

### VM runbook (TODO §1)

```bash
# 0. one-time: toolchain. Confirm arch FIRST — it decides PROBE_TARGET.
uname -m                                   # x86_64 -> x86_64-linux-gnu ; aarch64 -> aarch64-linux-gnu
git submodule update --init --recursive
./scripts/alive2/01-prereqs.sh             # apt: cmake ninja re2c z3 libz3-dev clang-21 llvm-21-dev
./scripts/alive2/02-build-alive2.sh        # builds alive-tv only (~15 min, a few GB RAM)
source scripts/alive2/env.sh               # exports ALIVE_TV + LLVM_BIN
# Triple AND cpu must match, or llc hard-fails and every mca_cycles_o3 comes out null:
export PROBE_TARGET=x86_64-linux-gnu       # <- match `uname -m`
export PROBE_CPU=znver3                    # <- match `lscpu` family (aarch64 default: cortex-a72)
"$ALIVE_TV" --version && "$LLVM_BIN/llvm-mca" --version
# prove the pair before building anything (must print a cycle count, not an error):
printf 'define i32 @f(i32 %%a){\n %%b = add i32 %%a, 1\n ret i32 %%b\n}\n' > /tmp/t.ll
"$LLVM_BIN/llc" -mtriple "$PROBE_TARGET" -mcpu "$PROBE_CPU" -o /tmp/t.s /tmp/t.ll \
  && "$LLVM_BIN/llvm-mca" -mtriple "$PROBE_TARGET" -mcpu "$PROBE_CPU" /tmp/t.s | grep 'Total Cycles'

# 1. sanity: the whole pipeline offline, no keys, before touching the real corpus
uv sync --extra dev && uv run pytest
uv run python -m probe.run_probe --corpus data/bootstrap --backend mock \
    --k 8 --verifier stub --perf stub

# 2. corpus source (~1 GB shallow clone into ~/.cache/t3rl-corpus; CORPUS_SRC_DIR to relocate)
./scripts/fetch-corpus.sh

# 3. build UNCAPPED (--max-functions truncates alphabetically; it is not a sampling strategy).
#    Serial today (TODO §7 --jobs), so expect this to take a while on one core.
uv run python -m probe.build_corpus --src "$(./scripts/fetch-corpus.sh -q)" \
    --out data/corpus/testsuite-full.jsonl --with-mca

# 3b. FIRST check: did mca actually score anything? A nonzero count here means the
#     triple/cpu pair was wrong and the corpus must be rebuilt after fixing it.
grep -c '"mca_cycles_o3":null' data/corpus/testsuite-full.jsonl

# 4. is the corpus trustworthy? (both are read-only checks on the new corpus)
uv run python -m probe.perf_sanity  --corpus data/corpus/testsuite-full.jsonl   # -O0 cycles >= -O3?
uv run python -m probe.verify_corpus --corpus data/corpus/testsuite-full.jsonl  # Alive2 verdict rate
#    ^ writes results/verdicts.jsonl (per function), which step 5 needs.
#      Aborts early if alive-tv is missing rather than reporting 0% verified as a "result".

# 5. derive the two corpora, then confirm the filter worked
uv run python -m probe.make_corpora --corpus data/corpus/testsuite-full.jsonl \
    --verdicts results/verdicts.jsonl \
    --out-train data/corpus/train.jsonl --out-report data/corpus/report.jsonl
uv run python -m probe.perf_sanity --corpus data/corpus/train.jsonl   # expect ~98%, up from 88%
```

Step 4 is not optional: `perf_sanity` is the check that the *speed* half of the reward is not noise,
and `verify_corpus` re-runs the Phase 1 Part A headline on the new distribution. Both numbers should
be compared against Phase 1's (79% verified, 98% perf sanity) — a large drop means the new corpus is
harder than what the oracle was validated on, and the honest scope needs revisiting *before* any RL.

Run `verify_corpus` with the bounded worker pool (§7) once it exists; at ~6 workers and a 30 s
timeout, a few thousand functions is hours, not minutes.

### Corpus build result — 2026-08-02

llvm-test-suite/SingleSource, uncapped, on the Debian VM (`scripts/corpus_report.py`):

```
records             3414
no o3_baseline_ir     175  (  5.1%)   llvm-extract failed; null mca expected
null mca_cycles       178  (  5.2%)
  had O3, mca fail      3  (  0.1%)   <- llc/mca genuinely failed
scored               3236  ( 94.8%)
```

**94.8% scored, 3 real mca failures — the `PROBE_TARGET`/`PROBE_CPU` pair was correct.** Buckets over
the 3236 scored records:

| size | loop-free | loops | total |
|---|---:|---:|---:|
| ≤20 | 2120 | 33 | 2153 |
| 20–50 | 517 | 213 | 730 |
| 50–150 | 112 | 110 | 222 |
| >150 | 33 | 98 | 131 |
| **total** | **2782** | **454** | **3236** |

Read: **the raw distribution is still 66% tiny loop-free** — the same skew as the old 64-function
corpus, just 50× more of it. Phase 2 found those are mostly already optimal at `-O3`, so most of this
corpus is *dead weight as a reward target*. What's new and valuable is the absolute counts in the
buckets that matter: 213 small loops (Phase 2's best bucket had 12) and 110 mid-size loops. Every
bucket now clears n=33, so per-bucket numbers can finally be reported without n=1 caveats.

### Oracle + perf sanity on the new corpus — 2026-08-02

**`verify_corpus` (valid run, 35 min, 3239 checked):**

| | this corpus | Phase 1 (784 checked) |
|---|---:|---:|
| verified | **51.9%** (1680) | 79% |
| counterexample | **13.0%** (421) | 5% |
| couldn't-prove / abstain (`error`) | 19.0% (617) | 12.5% |
| unsupported | 13.0% (420) | 3% |
| timeout | 3.1% (101) | 0.4% |
| **loop-free** | **58.0%** (1616/2784) | 81% |
| **loops** | **14.1%** (64/455) | 4% (n=23) |

Per bucket, verified rate: ≤20 loop-free **63%** (1331) · 20–50 loop-free 47% (243) · 50–150 loop-free
35% (39) · ≤20 loops 21% · 20–50 loops 19% (40) · 50–150 loops 15% (16) · >150 loop-free 9% ·
>150 loops **1%** (89/98 `unsupported`).

Three things to take from this:

1. **1,680 functions the oracle provably handles.** That is the train-corpus pool, and it is ~26× the
   entire old corpus. The headline drop (79% → 52%) is *composition*, not regression: Phase 1's corpus
   was hand-curated single-function C files; this is a compiler stress-test suite.
2. **`error` is Alive2 abstaining, not crashing** — Phase 1 established this is mostly
   `Couldn't prove the correctness of the transformation` (partA-findings). It behaves like
   timeout/unsupported for reward purposes (not verified → 0), which is exactly why TODO §7's
   `failed_to_prove` outcome matters: 19% of the corpus is currently filed under a name that reads
   like a bug.
3. **Counterexamples tripled, 5% → 13% (421).** These claim `-O0` and `-O3` differ, which is not
   plausible as 421 LLVM miscompiles. They are `llvm-extract` context artifacts — the `-O3` copy
   specializes on callers/globals the extracted `-O0` copy cannot see. 283 of them are in the ≤20
   loop-free bucket, where out-of-context specialization is most likely. **TODO §6 is no longer
   theoretical; it now has an n=421 target.**

**Loops are not as dead as Phase 1 suggested** — 14.1% vs 4%, but Phase 1's loop sample was n=23, so
4% was 1/23 and meaningless. At n=455 the honest statement is: loops verify at ~14%, and above 150
instructions they are hopeless (1%, almost entirely `unsupported`).

**Oracle throughput (the RL budget number):** 0.62 s mean per check serial (median 0.02–0.05 s, p90 up
to 22 s on >150 loop-free). At the planned ~6 workers that is ~9.7 checks/s → **~1.6 s of oracle time
per G=16 GRPO step**. The oracle is *not* the bottleneck we feared, provided the corpus excludes the
>150 buckets where p90 blows out.

**`perf_sanity`: 88% monotonic (2847/3236), vs Phase 1's 98%.** This one is real, and the inversion
list explains itself:

- **libc reimplementations.** `builtins/lib/strlen::strlen` goes `O0=15 → O3=110`; same shape for
  `memcpy`, `strcmp`, `strcat`, `abs`. At `-O3` LLVM's loop-idiom recognition spots the hand-written
  loop and replaces it with a **call to the very libc function being defined**. mca models the call as
  expensive, so `-O3` scores "slower". The rewrite is legitimate; the *measurement* is meaningless.
- **`main` drivers.** A large share of the rest are `main`, clustered at ~105–120 cycles at both
  levels — test harness scaffolding, not optimizable kernels.

Both are artifacts of mining a *compiler test suite*: gcc-c-torture is full of libc reimplementations
and driver `main`s. Neither existed in the old 64-function corpus, which is why 98% → 88%. Filter them
out (drop `main`; drop functions whose `-O3` IR calls a function the `-O0` IR does not) rather than
treating 88% as the oracle degrading. This is the same family as TODO §6's spurious-counterexample
audit: `llvm-extract` pulls a function out of context and the `-O3` copy specializes on things the
`-O0` copy cannot see.

### Filter result — 2026-08-03

`make_corpora` on the 3414-record build, with `verdicts.jsonl` from the (valid) verify run:

```
train  1543 functions   dropped: 842 driver main · 625 unverifiable · 178 unscoreable
                                 121 libc idiom · 105 >150-instr
report  695 functions   (cap 150/stratum)
```

**perf sanity on `train`: 96.6% (1491/1543), up from 88.0%** on the raw corpus — Phase 1's reference
was 98%. The `main`/libc diagnosis was right.

Final corpus after the intrinsic fix + `--train-cap-per-bucket 300`: **train 577 · report 678**, perf
sanity **95.7% (552/577)**. Note that is *lower* than the uncapped 96.6%: the cap removed 943
tiny/loop-free functions, which were the cleanest ones, so the inversion **rate** rose (3.4% → 4.3%)
even though the absolute count fell 52 → 25. Denominator effect, not a regression.

**But llvm-mca prices a single call at ~104 cycles** (measured: `call` → 104, plain arithmetic → 4,
aarch64/cortex-a72). That explains the entire residual: the leftover inversions cluster at
`O0=107 → O3=108`, i.e. one cycle of real difference on top of ~104 cycles of constant call overhead.
Two consequences, and the second is the serious one:

1. Those ~52 remaining "inversions" are not inversions; they are noise on a swamped measurement.
2. **For any call-containing function, a rewrite that shaves one cycle scores `verified_faster` at
   1.009× — a "win" that is entirely a static-model artifact.** Phase 2's coverage numbers may include
   some of these. The fix is a **minimum relative speedup** before `verified_faster` is granted (see
   TODO §5) and/or `--perf timing` for this population. `make_corpora` now reports the call-containing
   fraction so the scale is visible.

The 25 residual inversions are **two known llvm-mca limitations, not corpus defects**, and the split
is visible in the numbers:

- **call-swamped** (~19): the `O0=107 → O3=108` cluster, one real cycle on ~104 of call overhead.
- **loop unrolling** (~6): `memset 13 → 23`, `my_memcpy 16 → 30`, `pr60960::f1 16 → 17`. These are
  *not* the idiom-recognition artifact — the magnitudes are far too small for a call. `-O3` unrolls or
  vectorizes, so a single pass through the code does more work while the loop runs fewer times, and
  `llvm-mca --iterations=1` cannot see trip counts. This is exactly the 69%-on-loops weakness in
  [perf-scorer-findings](../phase1/perf-scorer-findings.md).

Together with the 24% of `train` that contains a call, this is the strongest argument yet for TODO §3's
"real timing as the primary reward": mca is unreliable on both call-containing *and* loop functions,
which between them are most of what has interesting headroom.

**The train corpus is still the wrong shape, and the oracle filter made it worse.** Verified-only left
1251/1543 (**81%**) in ≤20/loop-free, versus 66% in the raw corpus — Alive2 verifies tiny loop-free
functions most easily, and those are exactly the ones Phase 2 found are *already optimal at -O3*. A
function with no headroom returns reward 0 on every rollout just as surely as one that never verifies.
`--train-cap-per-bucket` (default 300) blunts this, but the real filter needs headroom data we do not
have yet: **the `qwen3-8b` baseline run produces it** — keep the functions that ever yielded a
`verified_faster` sample, and that becomes the v2 train stream.

### Two derived corpora, not one

The balanced corpus TODO §1 asks for is for **reporting**; the RL stream wants something different,
and conflating them will waste most of the rollout budget:

- **Report corpus** (~900, balanced): downsample ≤20/loop-free to ~150, keep everything in 50–150 and
  >150, keep all loops. Gives stable per-bucket numbers for the paper's tables.
- **Train corpus** (RL stream): filter to functions where **the oracle actually returns a verdict**
  *and* there is headroom over `-O3`. A function Alive2 can never verify contributes reward 0 to every
  rollout forever — it is not a hard example, it is a dead one, and GRPO learns nothing from a group
  that is uniformly zero. Alive2 verifies only ~4% of loops (Phase 1), so a size/loop-balanced corpus
  is close to the *worst* choice for training. Build this one from the `verify_corpus` output, not
  from the bucket histogram.

## Reward shaping (design decision, not yet made)

Fireworks evaluators return **0.0–1.0**; our speedup is unbounded. The squashing function interacts
directly with the hacking audit (TODO §6): **do not give partial credit to `verified_no_gain`**, or we
reopen the *output-the-input-unchanged* cheat the design closes. Proposed:

- `invalid_syntax` / `counterexample` / `timeout` / `unsupported` → **0.0**
- `verified_no_gain` → **0.0** (correct but worthless; keeps reward multiplicative)
- `verified_faster` → **`1 - 1/speedup`** (0.33 at 1.5×, 0.5 at 2×, asymptotes to 1)

Write this down in a spec before coding; it's a reviewer-visible choice.

## Work split

**Pipeline owner (macOS, no GPU):**
- [ ] Run the `qwen3-8b` baseline above; publish the go/no-go.
- [ ] `src/probe/reward.py` — outcome → scalar in [0,1], per the shaping above (+ tests).
- [ ] `scripts/rft/env_server.py` — the `/init` service. Build it against the mock backend + stub
      verifier first, so the contract is proven before the oracle is involved.
- [ ] Launch/monitor the RFT job via the eval-protocol CLI.

**VM owner (Debian box):**
- [x] `01-prereqs.sh` + `02-build-alive2.sh`; confirm RTTI, `alive-tv --version`, `source env.sh`.
      *2026-08-02.*
- [x] Build the real corpus (uncapped) + `perf_sanity` + `verify_corpus`. *2026-08-02 — **outputs not
      yet recorded in the repo**; see TODO §1 "the four numbers".*
- [ ] Stratified sampler: full corpus → balanced subset by `(size_bucket, has_loops)`. Blocks the
      balanced corpus TODO §1 asks for; `--max-functions` cannot do this.
- [ ] Parallel Alive2 worker pool (~6 workers, memory-capped) + verdict cache keyed on
      `(src, tgt)` hash (TODO §7). **Report verified-rewrites/sec** — that number sets the rollout
      throughput ceiling and therefore the budget.
- [ ] Expose the env server over HTTPS; document the tunnel setup.

## Open questions for Fireworks

- Can an environment/evaluator ship as **custom Docker on their side**? (mcpmark-lite includes a
  Dockerfile.) If yes, `alive-tv` + `llvm-mca` run on their infra and the tunnel disappears — keep
  the VM for `--perf timing` audits only.
- Hard timeout / concurrency limits on `/init` rollouts?
- Do credits cover the **dedicated deployment** GPU-hours? Fine-tuned models serve only through
  dedicated deployments, so post-training eval runs bill even when RFT itself is free.

Pricing and free-tier details checked **2026-07-29** — RFT-free-under-16B has had promo windows
before, so re-verify before planning around it.
