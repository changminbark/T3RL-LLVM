from probe.verifier import StubVerifier
from probe.schema import VerdictStatus


def test_identical_ir_is_verified():
    ir = "define i32 @f() {\n  ret i32 0\n}"
    assert StubVerifier().check(ir, ir).status is VerdictStatus.verified


def test_formatting_differences_still_verified():
    src = "define i32 @f() {\n  ret i32 0  ; comment\n}"
    tgt = "define i32 @f() {\n\n    ret i32 0\n}"
    assert StubVerifier().check(src, tgt).status is VerdictStatus.verified


def test_different_ir_is_unsupported_not_verified():
    src = "define i32 @f() {\n  ret i32 0\n}"
    tgt = "define i32 @f() {\n  ret i32 1\n}"
    assert StubVerifier().check(src, tgt).status is VerdictStatus.unsupported
