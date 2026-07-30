# Pipeline diagrams

Open this in a Markdown preview (IDE or GitHub) to see the diagrams rendered.

The reward pipeline (the box that classifies one rewrite) is **the same** in both phases: Phase 1
uses it to *grade*, Phase 2 adds a best-of-K *selection* layer on top, and Phase 3 will wrap an RL
update around it — nothing inside the box changes.

## Phase 1 — Viability
![phase1 diagram](phase1/figures/phase1.png)
```mermaid
flowchart TD
    C["C functions (llvm-test-suite / data/c_sources)"]
    C -->|"clang -O0"| SRC["src_ir"]
    C -->|"clang -O3"| O3["o3_baseline_ir"]

    subgraph PA["Part A — is the oracle usable?"]
        AV{{"Alive2: is -O0 ≡ -O3?"}}
        PS{{"llvm-mca: cycles -O0 ≥ -O3?"}}
        AV --> VR["verdict rate / bucket<br/>79% verified · 0.4% timeout"]
        PS --> PSR["perf sanity 98%"]
    end

    subgraph PB["Part B — does the model have a prior?"]
        GEN["model samples N rewrites"]
        AV2{{"Alive2: src ≡ rewrite?"}}
        MCA["llvm-mca vs -O3"]
        SK["solve@K per bucket"]
        GEN --> AV2 -->|proven| MCA --> SK
    end

    SRC --> AV
    O3 --> AV
    SRC --> PS
    O3 --> PS
    SRC --> GEN
    SRC --> AV2

    VR --> GO{{"Go / No-Go"}}
    PSR --> GO
    SK --> GO
    GO --> RES["GO — scoped to loop-free (≤~150 instrs)"]
```

## Phase 2 — Best-of-K inference-time baseline
![phase2 diagram](phase2/figures/phase2.png)
```mermaid
flowchart TD
    F["function: src_ir + o3_baseline_ir"] --> P["build prompt"]
    P --> M["model backend<br/>samples N rewrites"]
    M --> S["sanitize IR (ir_utils)"]
    S -->|"no define"| INV1["invalid_syntax"]
    S -->|"IR"| V{{"Alive2: src ≡ rewrite?"}}
    V -->|"counterexample"| CE["counterexample"]
    V -->|"timeout / unsupported / error"| INV2["timeout / unsupported / invalid"]
    V -->|"proven equivalent"| SC["score: llvm-mca or timing, vs -O3"]
    SC -->|"not faster"| NG["verified_no_gain"]
    SC -->|"strictly faster"| VF["verified_faster (speedup_vs_o3)"]

    VF --> AGG
    NG --> AGG
    CE --> AGG
    INV1 --> AGG
    INV2 --> AGG

    subgraph AGG["best-of-K aggregation (phase2_baseline.py)"]
        R["per-function outcomes × N samples"] --> BK["best-of-K: keep verified, pick fastest"]
        BK --> CURVE["Coverage@K · MeanSpeedup@K<br/>~23–28% / ~1.4× over -O3"]
    end
```

## Phase 3 — TTRL loop (planned)

Same reward box, now closed into a GRPO+LoRA loop. The dashed line is the network boundary: Fireworks
holds the policy, our Debian VM holds the oracle. Plan: [phase3/README.md](phase3/README.md).

```mermaid
flowchart LR
    subgraph FW["Fireworks RFT (GPU)"]
        POL["policy (LoRA)"] --> ROLL["sample G rollouts"]
        UPD["GRPO update"] --> POL
    end

    subgraph VM["Debian VM (CPU) — env server /init"]
        RW["reward box, unchanged<br/>sanitize → alive-tv → llvm-mca"]
        CACHE[("verdict cache<br/>(src,tgt) hash")]
        RW <--> CACHE
        RW --> SC["reward ∈ [0,1]<br/>0 unless proven; 1 − 1/speedup"]
    end

    ROLL -.->|"POST /init"| RW
    SC -.->|"rollout_finished(reward)"| UPD
    SC --> DYN["flip rates · mode collapse<br/>adapted-vs-base @ equal K"]
```

**The oracle is the bottleneck, not the GPU** — Z3 can hold a step for minutes, which is why the
verdict cache and a bounded worker pool are prerequisites rather than optimizations.
