"""Real wall-clock timing scorer (TimingPerf).

Compiles a function's IR + a synthesized C benchmark driver into a **native host** executable and
measures median per-call nanoseconds. Runs on whichever host: we strip the IR's linux
`target triple`/`datalayout` and let the resolved clang retarget to the native machine (host default
target), so the same code times natively on the Linux run box *and* the macOS dev machine.

Scope: integer returns/params and pointer params (opaque `ptr` and `iN*`). Anything else
(float/double/struct/vector/vararg) → not timeable (`None`), reported by the audit. Inputs are a
deterministic function of the signature + seed, so the rewrite, the -O3 baseline, and the source are
all timed with identical inputs — a fair speedup.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .perf import PerfScorer
from .schema import PerfScore
from .tools import find_tool

# Match up to the `(` that opens the parameter list: linkage/return-attrs+type, then @name.
# The return type is the last whitespace token before `@name`. The param list itself is scanned with
# balanced parens (attributes like `captures(none)` / `range(i32 0, 65)` contain parens and commas).
_DEFINE_RE = re.compile(r"define\s+(.+?)@([A-Za-z0-9_.$]+)\s*\(")


def _balanced_params(ir: str, open_paren_idx: int) -> str | None:
    """Return the text inside the param parens starting at `open_paren_idx` ('('), tracking depth."""
    depth = 0
    for i in range(open_paren_idx, len(ir)):
        c = ir[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return ir[open_paren_idx + 1 : i]
    return None


def _split_top_level(params: str) -> list[str]:
    """Split a param list on top-level commas (ignoring commas inside attribute parens)."""
    out, depth, start = [], 0, 0
    for i, c in enumerate(params):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(params[start:i])
            start = i + 1
    tail = params[start:]
    if tail.strip():
        out.append(tail)
    return out


# IR integer type -> C type (via <stdint.h>). i1 is a bool in the C ABI.
_INT_C = {
    "i1": "_Bool",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
}

# Benchmark shape (bounds chosen so 2D index patterns like m[i*n+i] stay in-bounds: LEN*LEN < ARR_N).
_ARR_N = 8192
_LEN = 64


@dataclass
class Signature:
    name: str
    ret: str  # C type or "void"
    params: list[str]  # each: "int" | "ptr" (abstract kind)
    param_c: list[str]  # C type per param


class _Unsupported(Exception):
    pass


def _c_type(ir_ty: str) -> str:
    ir_ty = ir_ty.strip()
    if ir_ty == "ptr" or ir_ty.endswith("*"):
        return "PTR"
    if ir_ty in _INT_C:
        return _INT_C[ir_ty]
    raise _Unsupported(ir_ty)


def parse_signature(ir: str) -> Signature | None:
    """Parse `define <ret> @name(<params>)`. None if absent or an unsupported type appears."""
    m = _DEFINE_RE.search(ir)
    if not m:
        return None
    pre, name = m.group(1), m.group(2)
    raw_params = _balanced_params(ir, m.end() - 1)  # m.end()-1 points at the '('
    if raw_params is None:
        return None
    ret_tokens = pre.split()
    if not ret_tokens:
        return None
    try:
        ret_ir = ret_tokens[-1]  # type is the last token before @name
        ret = "void" if ret_ir == "void" else _c_type(ret_ir)
        if ret == "PTR":  # pointer return: can't fold/measure cleanly
            return None
        kinds, cparams = [], []
        if raw_params.strip() and raw_params.strip() != "void":
            for p in _split_top_level(raw_params):
                tok = p.strip().split()[0]  # leading type token, ignoring attrs/name
                c = _c_type(tok)
                if c == "PTR":
                    kinds.append("ptr")
                    cparams.append("int32_t*")
                else:
                    kinds.append("int")
                    cparams.append(c)
    except (_Unsupported, IndexError):
        return None
    return Signature(name=name, ret=ret, params=kinds, param_c=cparams)


def normalize_ir(ir: str) -> str:
    """Drop the module's target lines so host clang retargets to the native machine."""
    keep = [
        ln
        for ln in ir.splitlines()
        if not ln.lstrip().startswith(("target datalayout", "target triple"))
    ]
    return "\n".join(keep)


def render_driver(sig: Signature, seed: int) -> str:
    """Emit a C benchmark: seeded inputs, warmup, median-of-trials per-call ns minus loop overhead."""
    # Build argument expressions. A shared int32 array backs every pointer; an integer param that
    # follows a pointer is treated as that array's length (=LEN). Standalone ints get small seeded
    # positive values (keeps loop trip counts bounded).
    decls, args = [], []
    seen_ptr = False
    scalar_i = 0
    for kind in sig.params:
        if kind == "ptr":
            seen_ptr = True
            args.append("arr")
        elif seen_ptr:
            args.append("LEN")  # length of the array we just passed
        else:
            v = f"a{scalar_i}"
            decls.append(f"  int {v} = 1 + (int)(lcg() % 100);")
            args.append(v)
            scalar_i += 1
    call = f"{sig.name}({', '.join(args)})"
    fold = (
        f"sink += (int64_t){call};" if sig.ret != "void" else f"{call}; sink += arr[0];"
    )
    extern_params = ", ".join(sig.param_c) if sig.param_c else "void"

    return f"""#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARR_N {_ARR_N}
#define LEN {_LEN}
#define WARMUP 50
#define TRIALS 11
#define CALLS 200

extern {sig.ret} {sig.name}({extern_params});
static volatile int64_t sink;
static uint64_t st = {seed}ULL;
static uint32_t lcg(void) {{ st = st*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t)(st>>33); }}
static double now_ns(void) {{ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9 + t.tv_nsec; }}
static int cmpd(const void*a,const void*b) {{ double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y); }}

int main(void) {{
  static int32_t arr[ARR_N];
  for (int i=0;i<ARR_N;i++) arr[i] = (int32_t)(lcg()%1000) - 500;
{chr(10).join(decls)}
  for (int w=0; w<WARMUP; w++) {{ {fold} }}
  double trials[TRIALS], empty[TRIALS];
  for (int t=0;t<TRIALS;t++) {{
    double s=now_ns();
    for (int c=0;c<CALLS;c++) {{ {fold} }}
    trials[t]=(now_ns()-s)/CALLS;
  }}
  for (int t=0;t<TRIALS;t++) {{
    double s=now_ns();
    for (int c=0;c<CALLS;c++) {{ sink += arr[c & (ARR_N-1)]; }}
    empty[t]=(now_ns()-s)/CALLS;
  }}
  qsort(trials,TRIALS,sizeof(double),cmpd);
  qsort(empty,TRIALS,sizeof(double),cmpd);
  double ns = trials[TRIALS/2] - empty[TRIALS/2];
  if (ns <= 0) ns = trials[TRIALS/2];
  printf("%.4f\\n", ns);
  return 0;
}}
"""


class TimingPerf(PerfScorer):
    """Native-host wall-clock scorer. cost = measured per-call nanoseconds."""

    def __init__(self, seed: int = 1234567, run_timeout_s: int = 15):
        self.clang = find_tool("clang")
        self.seed = seed
        self.run_timeout_s = run_timeout_s

    def available(self) -> bool:
        return self.clang is not None

    def score(self, ir: str) -> PerfScore | None:
        if self.clang is None:
            return None
        sig = parse_signature(ir)
        if sig is None:
            return None
        driver = render_driver(sig, self.seed)
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "mod.ll").write_text(normalize_ir(ir))
            (d / "drv.c").write_text(driver)
            obj, exe = d / "mod.o", d / "bench"
            try:
                # Compile IR -> host object (default target), then link with the driver.
                if subprocess.run(
                    [
                        self.clang,
                        "-O2",
                        "-x",
                        "ir",
                        "-c",
                        str(d / "mod.ll"),
                        "-o",
                        str(obj),
                    ],
                    capture_output=True,
                    timeout=60,
                ).returncode:
                    return None
                if subprocess.run(
                    [self.clang, "-O2", str(d / "drv.c"), str(obj), "-o", str(exe)],
                    capture_output=True,
                    timeout=60,
                ).returncode:
                    return None
                proc = subprocess.run(
                    [str(exe)],
                    capture_output=True,
                    text=True,
                    timeout=self.run_timeout_s,
                )
            except (subprocess.TimeoutExpired, OSError):
                return None
            if proc.returncode != 0:
                return None
            try:
                ns = float(proc.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                return None
            if ns <= 0:
                ns = 1e-3
            return PerfScore(
                mca_cycles=0.0, code_size_bytes=obj.stat().st_size, wall_ns=ns
            )

    def cost(self, score: PerfScore) -> float:
        return score.wall_ns if score.wall_ns is not None else score.mca_cycles
