from probe.ir_utils import sanitize_module

SRC = (
    'target datalayout = "e-m:e-p270:32:32"\n'
    'target triple = "aarch64-linux-gnu"\n'
    "define dso_local i32 @f(i32 noundef %0) {\n  ret i32 %0\n}\n"
)


def test_placeholder_datalayout_is_replaced_from_src():
    model = (
        '; ModuleID = \'x\'\ntarget datalayout = "..."\n'
        'target triple = "aarch64-unknown-linux-gnu"\n'
        "declare i32 @llvm.abs.i32(i32, i1) #0\n"
        "define dso_local i32 @f(i32 noundef %0) #1 {\n"
        "  %2 = call i32 @llvm.abs.i32(i32 %0, i1 true)\n  ret i32 %2\n}\n"
        "attributes #0 = { nounwind }\nattributes #1 = { nounwind }\n"
    )
    out = sanitize_module(model, SRC)
    assert out is not None
    assert '"..."' not in out                       # placeholder replaced
    assert "aarch64-linux-gnu" in out               # src triple injected
    assert "#0" not in out and "#1" not in out      # attr-group refs stripped
    assert "attributes #" not in out                # attr-group defs dropped
    assert "declare i32 @llvm.abs.i32(i32, i1)" in out  # declare kept (ref stripped)
    assert "define dso_local i32 @f(i32 noundef %0) {" in out


def test_bare_function_gets_target_lines_injected():
    model = "define i32 @f(i32 %0) {\n  ret i32 %0\n}\n"
    out = sanitize_module(model, SRC)
    assert out is not None
    assert "target datalayout" in out and "target triple" in out


def test_no_define_returns_none():
    assert sanitize_module("just prose, no IR", SRC) is None


def test_keeps_globals_and_metadata():
    model = (
        "@.g = private constant [2 x i32] [i32 1, i32 2]\n"
        "define i32 @f(i32 %0) {\n  ret i32 %0\n}\n"
        "!0 = !{i32 1}\n"
    )
    out = sanitize_module(model, SRC)
    assert "@.g = private constant" in out and "!0 = !{i32 1}" in out
