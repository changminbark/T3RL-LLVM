"""Build corpus JSONL from C source files using the local clang.

Each *.c file under the input dir should contain a single function. We compile it at -O0
(src_ir, the record's source) and -O3 (o3_baseline_ir, the "beat the compiler" baseline),
then compute n_instructions and has_loops from the -O0 IR.

    uv run python -m probe.build_corpus --src data/c_sources --out data/bootstrap/seed.jsonl

If llvm-mca is installed we also fill mca_cycles_o3; otherwise it stays null and O3 comparisons
fall back to the source function (see outcome._baseline_cycles).

NOTE: n_instructions / has_loops here are a reasonable local definition. Person A's corpus is the
source of truth for the real runs — confirm both sides compute these the same way (open item).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from .perf import McaPerf
from .schema import CorpusRecord

_CLANG = shutil.which("clang")
_LLVM_EXTRACT = shutil.which("llvm-extract")
_DEFINE_RE = re.compile(r"define[^@]*@([A-Za-z0-9_.$]+)\s*\(")
_LABEL_DEF_RE = re.compile(r"^([A-Za-z0-9_.$]+):")
_BR_TARGET_RE = re.compile(r"\blabel %([A-Za-z0-9_.$]+)")


def _clang_ir(c_path: Path, opt: str) -> str | None:
    if _CLANG is None:
        return None
    try:
        proc = subprocess.run(
            [_CLANG, opt, "-emit-llvm", "-S", "-o", "-", str(c_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _function_name(ir: str) -> str | None:
    m = _DEFINE_RE.search(ir)
    return m.group(1) if m else None


def count_instructions(ir: str) -> int:
    """Count instruction-ish lines inside function bodies (non-label, non-brace, non-comment)."""
    n = 0
    in_body = False
    for line in ir.splitlines():
        s = line.strip()
        if s.startswith("define ") and s.endswith("{"):
            in_body = True
            continue
        if s == "}":
            in_body = False
            continue
        if in_body and s and not s.startswith(";") and not _LABEL_DEF_RE.match(s):
            n += 1
    return n


def has_loops(ir: str) -> bool:
    """Detect a natural loop via a back-edge: a br to a label defined earlier in the function."""
    seen: set[str] = set()
    for line in ir.splitlines():
        s = line.strip()
        m = _LABEL_DEF_RE.match(s)
        if m:
            seen.add(m.group(1))
        for target in _BR_TARGET_RE.findall(s):
            if target in seen:  # jumping back to an already-defined block
                return True
    return False


def list_defined_functions(ir: str) -> list[str]:
    """Names of all functions with a body (`define`) in a module, in source order."""
    return _DEFINE_RE.findall(ir)


def extract_function(module_ir: str, func_name: str) -> str | None:
    """Pull one function into its own module via llvm-extract. None if tool absent/fails."""
    if _LLVM_EXTRACT is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        mod = Path(td) / "mod.ll"
        mod.write_text(module_ir)
        try:
            proc = subprocess.run(
                [_LLVM_EXTRACT, f"-func={func_name}", "-S", "-o", "-", str(mod)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
    return proc.stdout if proc.returncode == 0 else None


def _norm_hash(ir: str) -> str:
    """Dedup key: strip comments, collapse whitespace, hash. Same normalization idea as the stub verifier."""
    lines = []
    for line in ir.splitlines():
        line = re.sub(r";.*$", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()


def bucket_histogram(records: list[CorpusRecord]) -> dict[tuple[str, bool], int]:
    """Counts per (size_bucket, has_loops) for the end-of-run summary."""
    return dict(Counter((r.size_bucket(), r.has_loops) for r in records))


def build_records(
    src_dir: Path, mca: "McaPerf | None", max_functions: int | None = None
) -> list[CorpusRecord]:
    """Walk .c files recursively; compile each -O0/-O3; split every defined function.

    function_id is `<file-stem>.<func>` (unique across files). Deduped by normalized -O0 IR.
    Stops once `max_functions` records are collected.
    """
    records: list[CorpusRecord] = []
    seen: set[str] = set()
    for c_path in sorted(src_dir.rglob("*.c")):
        o0_mod = _clang_ir(c_path, "-O0")
        if o0_mod is None:
            continue
        o3_mod = _clang_ir(c_path, "-O3")
        for name in list_defined_functions(o0_mod):
            src_ir = extract_function(o0_mod, name)
            if not src_ir:  # llvm-extract absent or function empty
                continue
            h = _norm_hash(src_ir)
            if h in seen:
                continue
            seen.add(h)
            o3_ir = extract_function(o3_mod, name) if o3_mod else None
            mca_cycles = None
            if mca is not None and o3_ir:
                score = mca.score(o3_ir)
                mca_cycles = score.mca_cycles if score else None
            records.append(
                CorpusRecord(
                    function_id=f"{c_path.stem}.{name}",
                    src_ir=src_ir.strip(),
                    source_lang="c",
                    n_instructions=count_instructions(src_ir),
                    has_loops=has_loops(src_ir),
                    o3_baseline_ir=o3_ir.strip() if o3_ir else None,
                    mca_cycles_o3=mca_cycles,
                )
            )
            if max_functions is not None and len(records) >= max_functions:
                return records
    return records


def build_record(c_path: Path, mca: McaPerf | None) -> CorpusRecord | None:
    src_ir = _clang_ir(c_path, "-O0")
    if src_ir is None:
        return None
    o3_ir = _clang_ir(c_path, "-O3")
    name = _function_name(src_ir) or c_path.stem
    mca_cycles = None
    if mca is not None and o3_ir is not None:
        score = mca.score(o3_ir)
        mca_cycles = score.mca_cycles if score else None
    return CorpusRecord(
        function_id=name,
        src_ir=src_ir.strip(),
        source_lang="c",
        n_instructions=count_instructions(src_ir),
        has_loops=has_loops(src_ir),
        o3_baseline_ir=o3_ir.strip() if o3_ir else None,
        mca_cycles_o3=mca_cycles,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build corpus JSONL from C sources via clang")
    p.add_argument("--src", required=True, help="dir of single-function *.c files")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--with-mca", action="store_true", help="fill mca_cycles_o3 via llvm-mca")
    p.add_argument("--max-functions", type=int, default=None, help="cap total records")
    args = p.parse_args()

    if _CLANG is None:
        raise SystemExit("clang not found on PATH")

    mca = McaPerf() if args.with_mca else None
    src_dir = Path(args.src)

    if _LLVM_EXTRACT is not None:
        records = build_records(src_dir, mca, args.max_functions)
    else:
        # Fallback: no llvm-extract -> one record per whole file (original behavior).
        print("llvm-extract not found; falling back to whole-file records")
        records = []
        for c_path in sorted(src_dir.rglob("*.c")):
            rec = build_record(c_path, mca)
            if rec:
                records.append(rec)
            if args.max_functions and len(records) >= args.max_functions:
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(r.model_dump_json() for r in records) + "\n")
    print(f"wrote {len(records)} records to {out}")
    for (bucket, loops), n in sorted(bucket_histogram(records).items()):
        print(f"  {bucket:>7}  loops={str(loops):5}  {n}")


if __name__ == "__main__":
    main()
