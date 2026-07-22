> Terminology: **N** = rewrites sampled per function (the pool), **K** = selection budget (K ≤ N) — see [README.md](README.md).

https://claude.ai/share/1e6efd07-84f9-4c40-8eff-8dc388cb3dd3 
The task. Take a function's code (in LLVM IR — the compiler's intermediate language that C, C++, Rust all compile into). Ask the model: "rewrite this so it does exactly the same thing, but faster." That's superoptimization — finding better code than the standard compiler (-O2/-O3) can find, since compilers only apply a fixed catalog of safe rewrites while a model can search more creatively.
The problem with training on this. "Does exactly the same thing" is really hard to grade. Tests can miss edge cases, and human labels don't exist. This is normally what kills unsupervised RL on code.
The trick. For this specific task, a perfect grader already exists: Alive2. Give it the original function and the model's rewrite, and it uses a theorem prover to either prove they're equivalent or hand back a concrete input where they differ. So the reward is simple and un-fakeable:
Model generates N rewrites of the function.
Alive2 checks each one.
Reward = (proven equivalent) × (how much faster it is, estimated by LLVM's own performance tools like llvm-mca).
Do an RL update on those rollouts. Repeat on the next function.
Why the reward is hard to hack. The two obvious cheats cancel each other out. Output wrong-but-fast code → Alive2 catches it, reward 0. Output the input unchanged → provably equivalent, but zero speedup, reward 0. To score, the model must produce code that is both provably correct and measurably better. That's rare — most RL-for-code setups have a gameable grader, and this one basically doesn't.
Where TTRL comes in. You do this at deployment time, on unlabeled functions from whatever codebase you actually care about — no training labels ever needed, because the compiler infrastructure is the labeler. The model adapts to the code distribution it's being used on. And unlike the math-TTRL papers, you don't need majority voting or consensus for most rollouts, because you have proofs instead of guesses — which sidesteps the mode-collapse problem those papers spend all their effort fighting.
The main practical risks, so you know what to check first:
Alive2 times out on big or loop-heavy functions (loops are only checked up to a few unrolled iterations). If it times out on most realistic functions, the oracle disappears and you're back to weak-verifier land. Measuring this timeout rate on real code is the first experiment.
The base model has to be able to write valid LLVM IR (or you let it write C and compile that down — slightly weaker but easier).
The model's prior has to be decent to begin with — TTRL amplifies existing ability rather than creating new ability, so if the base model almost never finds a genuinely better rewrite, there's nothing to reinforce.
One-sentence pitch version: "A model that teaches itself to out-optimize the compiler, using the compiler's own proof tools as the reward — no labels, no gameable grader."
==========================================================
Here's a phased breakdown designed so you two can work in parallel with minimal blocking. The key structural insight: the project has two independent risk factors — is the verifier usable in practice? and can the model actually do the task at all? — and they can be de-risked simultaneously by different people.
Phase overview
Phase 1 — Viability (2–4 weeks). Answer the two kill questions above with inference-only experiments. No RL, no training. Output: a go/no-go decision and a calibrated task corpus.
Phase 2 — Inference-time baseline (2–3 weeks). Build the full reward pipeline and establish the non-TTRL baseline: best-of-K sampling with Alive2 verification + llvm-mca scoring as pure selection. This number is what TTRL must beat, and the pipeline becomes the RL reward function unchanged.
Phase 3 — TTRL loop (4–6 weeks). Wrap Phase 2's reward in an online RL loop (GRPO-style, LoRA updates to keep it cheap) over an unlabeled function stream. Track per-function dynamics (extinction-window style: flip rates, collapse indicators).
Phase 4 — Analysis & writeup. Adapted-vs-base comparisons, held-out generalization, ablations (proof-gated reward vs. weaker evidence tiers), hacking audit.
Phases 2–4 depend on Phase 1's artifacts, but within each phase the infra/model split below persists, so you're never serialized on each other.
Phase 1 in detail
Split it into two workstreams with one shared interface: a JSONL corpus where each record is {function_id, src_ir, source_lang, n_instructions, has_loops, o3_baseline_ir, mca_cycles_o3}. Person A produces it; Person B consumes it. Agree on this schema on day one and you can otherwise work independently.
Workstream A — Verifier & corpus feasibility (Person A)
Goal: measure whether Alive2 gives verdicts often enough, and fast enough, on realistic functions to serve as a reward oracle.
Build the corpus. Mine single functions from the LLVM test-suite, and/or C files from permissively-licensed GitHub repos. Lower with clang -O0 -emit-llvm, extract individual functions with llvm-extract. Also store the -O3 version of each — that's the "beat the compiler" baseline. Bucket by size (≤20, 20–50, 50–150 IR instructions) and by loop presence. Target ~500–1,000 functions.
Build the verification harness: a wrapper around alive-tv src.ll tgt.ll with a hard timeout (start at 30s and 120s variants), parsing output into {verified, counterexample, timeout, unsupported}. Sandbox it; make it a simple CLI/function Person B can also call.
The key measurement: run Alive2 on (src at -O0) vs (src at -O3) for every function — i.e., can it verify the compiler's own optimizations? This gives the verdict rate per size/loop bucket, which upper-bounds how often you'll get oracle signal on model outputs.
Set up llvm-mca (and code-size as a secondary metric) as the performance scorer; sanity-check that its cycle estimates rank -O0 < -O3 reliably.
Deliverables: the corpus JSONL, the harness, and a one-page table of verdict rates and median verification times per bucket.
Workstream B — Model capability probe (Person B)
Goal: measure whether any accessible base model has a nonzero prior on this task — the thing TTRL would amplify.
While waiting for the real corpus, start with ~50 hand-picked functions (Alive2's own test cases work) so prompting work isn't blocked.
Test the two generation formats: (a) model emits LLVM IR directly, (b) model emits C that you lower to IR. Measure syntactic validity rate for (a) — if it's poor even for strong open models (try Qwen-coder-class), format (b) wins by default.
Once Person A's corpus lands: for each function, sample N=8–16 rewrites at temperature ~0.8–1.0, run them through Person A's harness, and measure the number that matters: fraction of functions with ≥1 verified-and-faster-than-O3 rewrite (call it solve@K). Also record the distribution of outcomes (invalid / counterexample / timeout / verified-no-gain / verified-faster) — this is your future reward distribution and tells you how sparse the RL signal will be.
Cheap prompt ablations: include -O3 output in the prompt as a starting point vs. -O0 only; include Alive2 counterexamples in a single retry turn (previews the multi-turn value).
Deliverables: solve@K per model per bucket, chosen generation format, prompt templates.
Go/no-go criteria (agree on these up front):
Alive2 returns a verdict (not timeout/unsupported) on ≥ ~60% of functions in at least the two smaller buckets. If not, shrink function size or accept a hybrid verifier and rescope.
Some model achieves solve@16 ≥ ~5–10% on the loop-free buckets. Below that, the prior is too weak for TTRL to amplify — consider a light SFT warm-up phase or a different model before proceeding.
Median verification time ≤ ~30s, so an RL loop's reward calls are affordable.
Treat these thresholds as calibration points rather than hard laws — the point is deciding them before seeing results so you don't rationalize a weak signal.
One coordination suggestion: keep a shared doc with three sections — interface changes, blocking questions, current numbers — and a twice-weekly 15-minute sync. Everything else in this phase genuinely doesn't require synchronous work.

