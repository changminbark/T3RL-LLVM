"""Render Phase 1/2 result plots from saved run artifacts into docs/phase{1,2}/figures/.

Run (matplotlib/numpy pulled in ephemerally, not project deps):
    uv run --with matplotlib --with numpy python scripts/make_plots.py

Inputs (produced by earlier runs; paths overridable via env):
    PHASE2_JSON   results/phase2_k16/phase2_baseline.json   (best-of-K curve + buckets)
    SLM_GLOB      results/probe_slm_qwen1p5b/*.rewrites.jsonl
    LLM_GLOB      results/probe_llm_deepseek_k16/*.rewrites.jsonl
"""
import collections
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Figures live under each phase's folder; plot2 is a Phase 1 result, the rest are Phase 2.
P1 = ROOT / "docs" / "phase1" / "figures"
P2 = ROOT / "docs" / "phase2" / "figures"
P1.mkdir(parents=True, exist_ok=True)
P2.mkdir(parents=True, exist_ok=True)

PHASE2_JSON = os.environ.get("PHASE2_JSON", str(ROOT / "results/phase2_k16/phase2_baseline.json"))
SLM_GLOB = os.environ.get("SLM_GLOB", str(ROOT / "results/probe_slm_qwen1p5b/*.rewrites.jsonl"))
LLM_GLOB = os.environ.get("LLM_GLOB", str(ROOT / "results/probe_llm_deepseek_k16/*.rewrites.jsonl"))


def load_rewrites(pattern):
    rows = []
    for f in glob.glob(pattern):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def outcome_fracs(rows):
    c = collections.Counter(r["outcome"] for r in rows)
    n = len(rows) or 1
    return {k: v / n for k, v in c.items()}


def solve_at_k(rows, k):
    by_fn = collections.defaultdict(list)
    for r in rows:
        by_fn[r["function_id"]].append(r)
    solved = sum(
        any(x["outcome"] == "verified_faster" for x in rs[:k]) for rs in by_fn.values()
    )
    return solved / (len(by_fn) or 1)


# ---------- Plot 1: best-of-K curve (Phase 2 headline) ----------
curve = json.load(open(PHASE2_JSON))["curve"]["overall"]
ks = sorted((int(k) for k in curve), key=int)
cov = [curve[str(k)]["coverage"] * 100 for k in ks]
spd = [curve[str(k)]["mean_speedup"] for k in ks]

fig, ax1 = plt.subplots(figsize=(7, 4.5))
c1 = "#1f77b4"
ax1.plot(ks, cov, "o-", color=c1, lw=2)
ax1.set_xlabel("K (selection budget; N=16 samples per function)")
ax1.set_ylabel("Coverage: % of functions beating -O3", color=c1)
ax1.tick_params(axis="y", labelcolor=c1)
ax1.set_xticks(ks)
ax1.set_ylim(0, 35)
for x, y in zip(ks, cov):
    ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=8, color=c1)
ax2 = ax1.twinx()
c2 = "#d62728"
ax2.plot(ks, spd, "s--", color=c2, lw=2)
ax2.set_ylabel("Mean speedup vs -O3 (x)", color=c2)
ax2.tick_params(axis="y", labelcolor=c2)
ax2.set_ylim(1.0, 1.5)
ax1.set_title("Phase 2 - best-of-K baseline (deepseek-v4-pro, 64 fns)\n"
              "LLVM as a tool: best-of-1 -> best-of-16 ~doubles coverage; curve saturates")
ax1.axvspan(8, 16, alpha=0.06, color="gray")
fig.tight_layout()
fig.savefig(P2 / "plot1_bestofk_curve.png", dpi=150)
print("wrote", P2 / "plot1_bestofk_curve.png")

# ---------- Plot 2: SLM vs LLM prior ----------
slm = load_rewrites(SLM_GLOB)
llm = load_rewrites(LLM_GLOB)
slm_o, llm_o = outcome_fracs(slm), outcome_fracs(llm)
metrics = ["solve@8", "verified_faster", "invalid_syntax", "counterexample"]
slm_vals = [solve_at_k(slm, 8), slm_o.get("verified_faster", 0),
            slm_o.get("invalid_syntax", 0), slm_o.get("counterexample", 0)]
llm_vals = [solve_at_k(llm, 8), llm_o.get("verified_faster", 0),
            llm_o.get("invalid_syntax", 0), llm_o.get("counterexample", 0)]
x = np.arange(len(metrics)); w = 0.38
fig, ax = plt.subplots(figsize=(7.5, 4.5))
b1 = ax.bar(x - w / 2, [v * 100 for v in slm_vals], w, label="SLM  qwen2.5-coder:1.5b", color="#ff7f0e")
b2 = ax.bar(x + w / 2, [v * 100 for v in llm_vals], w, label="LLM  deepseek-v4-pro", color="#2ca02c")
ax.set_ylabel("percent")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_title("Does the prior resonate with capability? (same 64 fns, K=8)\n"
             "SLM fails on IR *syntax* (95% invalid); LLM has a real, un-gamed prior")
ax.legend()
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(P1 / "plot2_slm_vs_llm.png", dpi=150)
print("wrote", P1 / "plot2_slm_vs_llm.png")

# ---------- Plot 3: coverage by bucket ----------
bybk = json.load(open(PHASE2_JSON))["curve"]["by_bucket"]
labels, cov8, n = [], [], []
for key in sorted(bybk):
    labels.append(key.replace("loops=", ""))
    cov8.append(bybk[key]["8"]["coverage"] * 100)
    n.append(bybk[key]["8"]["n_functions"])
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.barh(range(len(labels)), cov8, color="#1f77b4")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels([f"{l}  (n={c})" for l, c in zip(labels, n)])
ax.set_xlabel("best-of-8 coverage: % beating -O3")
ax.set_title("Where the wins are (best-of-8, by size|loops)\n"
             "Headroom concentrates in loops; small loop-free is mostly already optimal")
for i, v in enumerate(cov8):
    ax.annotate(f"{v:.0f}%", (v, i), textcoords="offset points", xytext=(4, 0), va="center", fontsize=8)
fig.tight_layout()
fig.savefig(P2 / "plot3_coverage_by_bucket.png", dpi=150)
print("wrote", P2 / "plot3_coverage_by_bucket.png")

# ---------- Plot 4: base model vs base + LLVM (the direct comparison) ----------
cov1 = curve["1"]["coverage"] * 100      # % of single samples that are verified-faster
cov16 = curve["16"]["coverage"] * 100    # best-of-16 coverage
conditions = ["Base model\n(no oracle)", "Base + LLVM\nbest-of-1", "Base + LLVM\nbest-of-16"]
shippable = [0.0, cov1, cov16]           # provably-correct faster rewrites you can actually use
colors = ["#9e9e9e", "#2ca02c", "#1f77b4"]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
bars = ax.bar(conditions, shippable, color=colors, width=0.6)
# Ghost bar: base model DOES emit ~cov1% correct-faster code, but unverifiable -> not shippable.
ax.bar([0], [cov1], width=0.6, facecolor="none", edgecolor="#9e9e9e", hatch="///", linewidth=1.2)
ax.set_ylabel("% functions with a PROVABLY correct, faster-than-O3 rewrite")
ax.set_ylim(0, 35)
ax.set_title("Base model vs base + LLVM tool (deepseek-v4-pro, 64 fns)\n"
             "Same weights - the LLVM oracle adds trust (k=1) then selection (k->16)")
for b, v in zip(bars, shippable):
    ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v),
                textcoords="offset points", xytext=(0, 4), ha="center", fontweight="bold")
ax.annotate(f"~{cov1:.0f}% correct but\nUNVERIFIABLE\n(can't tell which)", (0, cov1),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="#616161")
ax.annotate("verification\nvalue", xy=(1, cov1), xytext=(0.5, cov1 + 6),
            fontsize=8, ha="center", color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c"))
ax.annotate("selection\nvalue", xy=(2, cov16), xytext=(1.5, cov16 + 4),
            fontsize=8, ha="center", color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"))
fig.tight_layout()
fig.savefig(P2 / "plot4_base_vs_llm_tool.png", dpi=150)
print("wrote", P2 / "plot4_base_vs_llm_tool.png")
