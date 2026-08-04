"""Derive the *train* and *report* corpora from one full corpus build.

These are different objectives and must not be the same file:

- **report** — balanced across `(size_bucket, has_loops)` so per-bucket numbers are stable.
  Caps the dominant tiny/loop-free bucket, keeps every scarce one.
- **train** — the RL stream. Balance is the *wrong* objective here: a function Alive2 can never
  verify returns reward 0 on every rollout forever, and a uniformly-zero GRPO group has no
  gradient. So filter to functions where the oracle demonstrably works and the measurement is
  meaningful, then let the natural distribution stand.

Both drop the two artifact families the 2026-08-02 perf-sanity run exposed (see
docs/phase3/README.md): driver `main`s, and libc reimplementations where `-O3`'s loop-idiom
recognition rewrites a hand-written `strlen` into a *call to* `strlen` — a legitimate
optimization whose mca comparison is meaningless.

    uv run python -m probe.make_corpora --corpus data/corpus/testsuite-full.jsonl \
        --verdicts results/verdicts.jsonl \
        --out-train data/corpus/train.jsonl --out-report data/corpus/report.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from .schema import CorpusRecord

# `call ... @name(` — the callee of a direct call. Indirect calls have no @name and are ignored.
_CALL_RE = re.compile(r"\bcall\b[^@\n]*@([A-Za-z0-9_.$]+)")

# Buckets where the oracle is effectively unusable (1–9% verified) and p90 latency blows out to
# ~22 s, which would dominate rollout time for almost no reward signal.
_EXCLUDED_TRAIN_BUCKETS = (">150",)

# Intrinsics that are pure bookkeeping and say nothing about what the code does. Everything else —
# crucially `llvm.memcpy`/`llvm.memset`/`llvm.memmove` — counts as a call, because -O3 emitting one
# that -O0 lacks IS loop-idiom recognition, the same artifact as rewriting a loop into `@strlen`.
_IGNORED_CALLEES = ("llvm.lifetime", "llvm.dbg", "llvm.assume", "llvm.expect")


def called_functions(ir: str) -> set[str]:
    """Direct callees in a module. Indirect calls have no `@name` and are invisible here."""
    return {m for m in _CALL_RE.findall(ir) if not m.startswith(_IGNORED_CALLEES)}


def has_call(rec: CorpusRecord) -> bool:
    """Does the -O3 baseline contain a call at all?

    Matters because llvm-mca prices a single call at ~104 cycles (measured, aarch64/cortex-a72)
    while a whole arithmetic function costs ~4. Any function containing a call therefore has its
    cycle count swamped by a constant, and a rewrite that shaves one real cycle registers as a
    ~1% "speedup" that is pure model artifact. Reported, not filtered — see docs/phase3.
    """
    return bool(called_functions(rec.o3_baseline_ir or ""))


def introduces_calls(rec: CorpusRecord) -> bool:
    """True if -O3 calls something -O0 does not.

    This is the libc-idiom signature: at -O3 LLVM replaces a hand-written loop with a call to the
    library function it recognises (often the very function being defined). The rewrite is correct,
    but comparing mca cycles across it compares a loop against a call and tells us nothing.
    """
    if not rec.o3_baseline_ir:
        return False
    return bool(called_functions(rec.o3_baseline_ir) - called_functions(rec.src_ir))


def is_driver_main(rec: CorpusRecord) -> bool:
    """Test-suite `main` — harness scaffolding, not an optimizable kernel."""
    return rec.function_id.rsplit("::", 1)[-1] == "main"


def load_verified(path: Path) -> set[str]:
    """function_ids whose (-O0, -O3) pair Alive2 proved. Written by verify_corpus."""
    out = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "verified":
            out.add(row["function_id"])
    return out


def select_train(
    records: list[CorpusRecord],
    verified: set[str] | None,
    cap: int = 0,
    seed: int = 0,
) -> tuple[list[CorpusRecord], Counter]:
    """Filter to functions that can actually produce reward signal. Returns (kept, drop reasons).

    `cap` (0 = off) limits each (bucket, loops) stratum. Without it the oracle filter *worsens* the
    skew — Alive2 verifies tiny loop-free functions most easily, and those are exactly the ones
    Phase 2 found are already optimal at -O3, i.e. reward 0 on every rollout.
    """
    kept: list[CorpusRecord] = []
    dropped: Counter = Counter()
    for rec in records:
        if rec.mca_cycles_o3 is None:
            dropped["no mca baseline (unscoreable)"] += 1
        elif is_driver_main(rec):
            dropped["driver main"] += 1
        elif introduces_calls(rec):
            dropped["-O3 introduces a call (libc idiom)"] += 1
        elif rec.size_bucket() in _EXCLUDED_TRAIN_BUCKETS:
            dropped[">150 instrs (oracle ~1-9%, p90 22s)"] += 1
        elif verified is not None and rec.function_id not in verified:
            dropped["oracle could not verify -O0 == -O3"] += 1
        else:
            kept.append(rec)

    if cap:
        rng = random.Random(seed)
        strata: dict[tuple[str, bool], list[CorpusRecord]] = {}
        for rec in kept:
            strata.setdefault((rec.size_bucket(), rec.has_loops), []).append(rec)
        capped: list[CorpusRecord] = []
        for group in strata.values():
            if len(group) > cap:
                dropped[f"over the {cap}/stratum cap"] += len(group) - cap
                group = rng.sample(group, cap)
            capped.extend(group)
        capped.sort(key=lambda r: r.function_id)
        kept = capped
    return kept, dropped


def select_report(
    records: list[CorpusRecord], cap: int, seed: int
) -> tuple[list[CorpusRecord], Counter]:
    """Cap each (bucket, loops) stratum at `cap`, sampling without replacement."""
    rng = random.Random(seed)
    strata: dict[tuple[str, bool], list[CorpusRecord]] = {}
    for rec in records:
        if is_driver_main(rec) or introduces_calls(rec):
            continue
        strata.setdefault((rec.size_bucket(), rec.has_loops), []).append(rec)

    kept: list[CorpusRecord] = []
    sizes: Counter = Counter()
    for key, group in sorted(strata.items()):
        chosen = group if len(group) <= cap else rng.sample(group, cap)
        # Stable order regardless of sampling, so reruns with the same seed diff cleanly.
        chosen.sort(key=lambda r: r.function_id)
        kept.extend(chosen)
        sizes[f"{key[0]}|loops={key[1]}"] = len(chosen)
    return kept, sizes


def _write(path: Path, records: list[CorpusRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records))


def _histogram(records: list[CorpusRecord]) -> str:
    c = Counter((r.size_bucket(), r.has_loops) for r in records)
    return "\n".join(
        f"    {b:>7}  loops={str(loop):5}  {n}" for (b, loop), n in sorted(c.items())
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Derive train + report corpora from a full build"
    )
    p.add_argument("--corpus", required=True, help="full corpus JSONL")
    p.add_argument(
        "--verdicts", help="verdicts.jsonl from verify_corpus (filters train to proven)"
    )
    p.add_argument("--out-train", required=True)
    p.add_argument("--out-report", required=True)
    p.add_argument(
        "--cap-per-bucket", type=int, default=150, help="report corpus stratum cap"
    )
    p.add_argument(
        "--train-cap-per-bucket",
        type=int,
        default=300,
        help="train corpus stratum cap (0 = off). Default caps the tiny/loop-free flood.",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="sampling seed (reproducibility)"
    )
    args = p.parse_args(argv)

    records = [
        CorpusRecord(**json.loads(line))
        for line in Path(args.corpus).read_text().splitlines()
        if line.strip()
    ]
    verified = load_verified(Path(args.verdicts)) if args.verdicts else None
    if verified is None:
        print(
            "!! no --verdicts: train corpus is NOT filtered to oracle-verifiable functions"
        )

    train, dropped = select_train(
        records, verified, args.train_cap_per_bucket, args.seed
    )
    report, _ = select_report(records, args.cap_per_bucket, args.seed)

    _write(Path(args.out_train), train)
    _write(Path(args.out_report), report)

    print(f"\nread {len(records)} records")
    print(f"\ntrain -> {args.out_train}  ({len(train)} functions)")
    for reason, n in dropped.most_common():
        print(f"    dropped {n:>5}  {reason}")
    print(_histogram(train))
    n_call = sum(1 for r in train if has_call(r))
    if n_call:
        print(
            f"    note: {n_call} ({n_call / len(train):.0%}) contain a call — llvm-mca prices one"
            " call at ~104 cycles vs ~4 for plain arithmetic, so their cycle counts are"
            " swamped by a constant. Prefer --perf timing for these."
        )
    print(
        f"\nreport -> {args.out_report}  ({len(report)} functions, cap {args.cap_per_bucket})"
    )
    print(_histogram(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
