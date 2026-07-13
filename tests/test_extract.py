from probe.extract import extract_code
from probe.schema import GenFormat


def test_extract_tagged_llvm_block():
    c = "Sure, here it is:\n```llvm\ndefine i32 @f() {\n  ret i32 0\n}\n```\nDone."
    out = extract_code(c, GenFormat.ir)
    assert out is not None and "define i32 @f()" in out


def test_prefers_language_match_over_other_block():
    c = "```c\nint f(){return 0;}\n```\nand IR:\n```llvm\ndefine i32 @f() { ret i32 0 }\n```"
    out = extract_code(c, GenFormat.ir)
    assert "define" in out


def test_untagged_block_falls_back_to_largest():
    c = "```\ndefine i32 @f() {\n  ret i32 0\n}\n```"
    assert extract_code(c, GenFormat.ir) is not None


def test_no_code_returns_none():
    assert extract_code("I cannot help with that.", GenFormat.ir) is None


def test_c_block_extraction():
    c = "```c\nint f(int x){ return x + 1; }\n```"
    out = extract_code(c, GenFormat.c)
    assert out is not None and "return x + 1" in out
