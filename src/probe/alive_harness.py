"""`alive-harness` CLI — the Alive2 wrapper Part B's AliveCliVerifier shells out to.

Invoked as `alive-harness <src.ll> <tgt.ll> --timeout <s>`; prints one line of `Verdict`
JSON (schema.Verdict) to stdout. `classify_alive_output` is a pure function over alive-tv's
text output so parsing is unit-tested without Alive2 installed. A missing/failed alive-tv
yields a `Verdict(error)`, never a crash.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .schema import Verdict, VerdictStatus

_UNSUPPORTED_MARKERS = ("unsupported",)
_TIMEOUT_MARKERS = ("smt error: timeout", "timed out", "timeout")
_INCORRECT_MARKERS = ("doesn't verify", "value mismatch")
_CORRECT_MARKERS = ("seems to be correct",)


def _capture_counterexample(text: str, limit: int = 2000) -> str:
    """Grab the informative slice of a failing run (the Example / mismatch block)."""
    lowered = text.lower()
    for anchor in ("example:", "error:"):
        idx = lowered.find(anchor)
        if idx != -1:
            return text[idx : idx + limit].strip()
    return text[:limit].strip()


def classify_alive_output(
    stdout: str, stderr: str, returncode: int, timed_out: bool
) -> Verdict:
    """Map alive-tv's text output to a Verdict. Precedence is deliberate and ordered."""
    if timed_out:
        return Verdict(status=VerdictStatus.timeout)

    text = f"{stdout}\n{stderr}"
    low = text.lower()

    if any(m in low for m in _UNSUPPORTED_MARKERS):
        return Verdict(status=VerdictStatus.unsupported)
    if any(m in low for m in _TIMEOUT_MARKERS):
        return Verdict(status=VerdictStatus.timeout)
    if any(m in low for m in _INCORRECT_MARKERS):
        return Verdict(
            status=VerdictStatus.counterexample,
            counterexample=_capture_counterexample(text),
        )
    if any(m in low for m in _CORRECT_MARKERS):
        return Verdict(status=VerdictStatus.verified)
    return Verdict(
        status=VerdictStatus.error,
        counterexample=(stderr or stdout)[:2000].strip() or None,
    )


def _run_alive(alive_tv: str, src: Path, tgt: Path, timeout_s: int) -> Verdict:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [alive_tv, str(src), str(tgt)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        v = classify_alive_output("", "", -1, True)
        v.wall_time_s = time.perf_counter() - start
        return v
    except OSError as e:
        return Verdict(
            status=VerdictStatus.error,
            counterexample=str(e),
            wall_time_s=time.perf_counter() - start,
        )
    v = classify_alive_output(proc.stdout, proc.stderr, proc.returncode, False)
    v.wall_time_s = time.perf_counter() - start
    return v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Alive2 equivalence-check wrapper -> Verdict JSON")
    p.add_argument("src", help="source .ll file")
    p.add_argument("tgt", help="target .ll file")
    p.add_argument("--timeout", type=int, default=30, help="hard wall timeout (s)")
    p.add_argument(
        "--alive-tv",
        default=os.environ.get("ALIVE_TV") or shutil.which("alive-tv"),
        help="path to alive-tv (default: $ALIVE_TV or PATH)",
    )
    args = p.parse_args(argv)

    if not args.alive_tv:
        print(
            Verdict(
                status=VerdictStatus.error, counterexample="alive-tv not found"
            ).model_dump_json()
        )
        return 0

    verdict = _run_alive(args.alive_tv, Path(args.src), Path(args.tgt), args.timeout)
    print(verdict.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
