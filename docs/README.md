# T3RL-LLVM docs

Design docs, plans, and findings for the TTRL superoptimizer — a model that rewrites LLVM IR to be
provably-equivalent-but-faster, using Alive2 as an un-gameable reward oracle and llvm-mca as the
speed scorer.

- **[Phase 1 — viability](phase1/README.md):** is the verifier usable (Part A) and does a base
  model have a prior worth amplifying (Part B)? → **GO, scoped to loop-free functions.**
- **[Phase 2 — best-of-K baseline](phase2/README.md):** the non-TTRL number TTRL must beat →
  **~28% coverage / 1.38× over -O3.**
- Phase 3 (TTRL loop) and Phase 4 (writeup) are future.

**Terminology:** **N** = rewrites sampled per function (the pool; `run_probe --k`). **K** = the
selection/evaluation budget in `@K` metrics (`solve@K`, `Coverage@K`, `best-of-K`, `pass@K`), K ≤ N.
Each phase README defines the terms it uses.
