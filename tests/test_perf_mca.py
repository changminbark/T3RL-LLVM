"""McaPerf integration tests. Skipped when the LLVM toolchain isn't installed."""

import pytest

from probe.perf import McaPerf

_MCA = McaPerf()
requires_mca = pytest.mark.skipif(not _MCA.available(), reason="llc/llvm-mca not installed")

# Same function at -O0 (stack-heavy) and -O3 (returns the argument): O0 must score more cycles.
O0_IR = """define i32 @f(i32 %x) {
entry:
  %p = alloca i32
  store i32 %x, ptr %p
  %a = load i32, ptr %p
  %b = mul i32 %a, 8
  ret i32 %b
}
"""
O3_IR = """define i32 @f(i32 %x) {
  %b = shl i32 %x, 3
  ret i32 %b
}
"""


@requires_mca
def test_mca_scores_are_positive():
    s = _MCA.score(O3_IR)
    assert s is not None and s.mca_cycles > 0 and s.code_size_bytes > 0


@requires_mca
def test_mca_ranks_o0_slower_than_o3():
    assert _MCA.score(O0_IR).mca_cycles > _MCA.score(O3_IR).mca_cycles


def test_mca_returns_none_on_garbage_when_available():
    # invalid IR should fail gracefully (None), not raise
    assert _MCA.score("this is not IR") is None
