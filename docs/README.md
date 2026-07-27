# T3RL-LLVM docs

Design docs, plans, and findings for the TTRL superoptimizer — a model that rewrites LLVM IR to be
provably-equivalent-but-faster, using Alive2 as an un-gameable reward oracle and llvm-mca as the
speed scorer.

- **[Phase 1 — viability](phase1/README.md):** is the verifier usable (Part A) and does a base
  model have a prior worth amplifying (Part B)? → **GO, scoped to loop-free functions.**
- **[Phase 2 — best-of-K baseline](phase2/README.md):** the non-TTRL number TTRL must beat →
  **~28% coverage / 1.38× over -O3** (reference corpus). Hardened two ways:
  [real-timing audit](phase2/timing-validation.md) of the llvm-mca speed proxy (wall-clock ranks
  -O0/-O3 correctly 100% vs mca 90.6%), and [IR robustness](phase2/ir-robustness.md) — normalizing
  the model's IR cut `invalid_syntax` 61%→10% and doubled coverage (11%→23.4%) on a local run.
- Phase 3 (TTRL loop) and Phase 4 (writeup) are future.

**Terminology:** **N** = rewrites sampled per function (the pool; `run_probe --k`). **K** = the
selection/evaluation budget in `@K` metrics (`solve@K`, `Coverage@K`, `best-of-K`, `pass@K`), K ≤ N.
Each phase README defines the terms it uses.
