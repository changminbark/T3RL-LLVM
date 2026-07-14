from probe.alive_harness import classify_alive_output
from probe.schema import VerdictStatus

VERIFIED = """
----------------------------------------
define i32 @f(i32 %x) {
  ret i32 %x
}
=>
define i32 @f(i32 %x) {
  ret i32 %x
}

Transformation seems to be correct!

Summary:
  1 correct transformations
  0 incorrect transformations
  0 failed-to-prove transformations
  0 Alive2 errors
"""

COUNTEREXAMPLE = """
Transformation doesn't verify!
ERROR: Value mismatch

Example:
i32 %x = #x00000001 (1)

Source value: #x00000001 (1)
Target value: #x00000000 (0)

Summary:
  0 correct transformations
  1 incorrect transformations
"""

UNSUPPORTED = """
ERROR: Unsupported instruction: call
Summary:
  0 correct transformations
  0 incorrect transformations
  1 Alive2 errors
"""

ALIVE_TIMEOUT = """
ERROR: SMT Error: timeout
"""


def test_verified():
    v = classify_alive_output(VERIFIED, "", 0, False)
    assert v.status is VerdictStatus.verified


def test_counterexample_captures_example():
    v = classify_alive_output(COUNTEREXAMPLE, "", 0, False)
    assert v.status is VerdictStatus.counterexample
    assert "Target value" in (v.counterexample or "")


def test_unsupported():
    v = classify_alive_output(UNSUPPORTED, "", 0, False)
    assert v.status is VerdictStatus.unsupported


def test_wall_timeout_wins():
    # Our own wall-clock timeout tripped -> timeout regardless of stdout.
    v = classify_alive_output("", "", -9, True)
    assert v.status is VerdictStatus.timeout


def test_alive_reported_timeout():
    v = classify_alive_output(ALIVE_TIMEOUT, "", 1, False)
    assert v.status is VerdictStatus.timeout


def test_garbage_is_error():
    v = classify_alive_output("random noise", "boom", 2, False)
    assert v.status is VerdictStatus.error
    assert "boom" in (v.counterexample or "")


def test_mixed_output_is_conservative_counterexample():
    # A module output containing BOTH a correct and a failing transformation
    # must be classified counterexample, never verified.
    mixed = (
        "Transformation seems to be correct!\n"
        "Transformation doesn't verify!\n"
        "ERROR: Value mismatch\nExample:\n i32 %x = #x00000001\n"
    )
    from probe.alive_harness import classify_alive_output
    from probe.schema import VerdictStatus
    v = classify_alive_output(mixed, "", 0, False)
    assert v.status is VerdictStatus.counterexample


import os
import stat
import subprocess
import sys
from pathlib import Path

from probe.verifier import AliveCliVerifier
from probe.schema import VerdictStatus


def _write_fake_alive_tv(dir_path: Path, body: str) -> Path:
    """A fake alive-tv that ignores its args and prints `body` to stdout."""
    script = dir_path / "alive-tv"
    script.write_text(f"#!/usr/bin/env python3\nimport sys\nsys.stdout.write({body!r})\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def test_roundtrip_through_alivecliverifier(tmp_path, monkeypatch):
    # Fake alive-tv that always reports a correct transformation.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_fake_alive_tv(bindir, "Transformation seems to be correct!\n")
    monkeypatch.setenv("ALIVE_TV", str(bindir / "alive-tv"))

    # Invoke the real alive-harness CLI via a wrapper script AliveCliVerifier can exec.
    harness = tmp_path / "alive-harness"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        f'exec {sys.executable} -m probe.alive_harness "$@"\n'
    )
    harness.chmod(harness.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv(
        "PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src")
    )

    verifier = AliveCliVerifier(cli_cmd=str(harness))
    verdict = verifier.check("define i32 @f() {\n ret i32 0\n}", "define i32 @f() {\n ret i32 0\n}")
    assert verdict.status is VerdictStatus.verified
