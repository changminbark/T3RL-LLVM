from pathlib import Path

from probe.build_corpus import count_instructions, has_loops, _function_name, list_defined_functions, _norm_hash, bucket_histogram, _function_id
from probe.schema import CorpusRecord

LOOP_IR = """define i32 @sum(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %done
done:
  ret i32 %i.next
}
"""

STRAIGHT_IR = """define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 1
  ret i32 %a
}
"""


def test_function_name():
    assert _function_name(LOOP_IR) == "sum"
    assert _function_name(STRAIGHT_IR) == "f"


def test_has_loops_detects_back_edge():
    assert has_loops(LOOP_IR) is True


def test_no_loops_on_straightline():
    assert has_loops(STRAIGHT_IR) is False


def test_count_instructions_ignores_labels_and_braces():
    # entry(br) + phi + add + icmp + br + ret = 6 instruction lines
    assert count_instructions(LOOP_IR) == 6


TWO_FUNCS = """define i32 @a(i32 %x) {
  ret i32 %x
}

define i32 @b(i32 %y) {
  %r = add i32 %y, 1
  ret i32 %r
}

declare i32 @ext(i32)
"""


def test_list_defined_functions_ignores_declares():
    assert list_defined_functions(TWO_FUNCS) == ["a", "b"]


def test_norm_hash_ignores_comments_and_whitespace():
    a = "define i32 @f() {\n  ret i32 0  ; note\n}"
    b = "define i32 @f() {\n\n    ret i32 0\n}"
    assert _norm_hash(a) == _norm_hash(b)


def test_norm_hash_distinguishes_bodies():
    a = "define i32 @f() {\n  ret i32 0\n}"
    b = "define i32 @f() {\n  ret i32 1\n}"
    assert _norm_hash(a) != _norm_hash(b)


def test_function_id_disambiguates_same_stem_across_dirs():
    root = Path("/corpus")
    a = _function_id(Path("/corpus/a/foo.c"), root, "bar")
    b = _function_id(Path("/corpus/b/foo.c"), root, "bar")
    assert a != b
    assert a == "a/foo::bar"
    assert b == "b/foo::bar"


def test_bucket_histogram_counts_by_size_and_loop():
    recs = [
        CorpusRecord(function_id="x", src_ir="", n_instructions=5, has_loops=False),
        CorpusRecord(function_id="y", src_ir="", n_instructions=10, has_loops=False),
        CorpusRecord(function_id="z", src_ir="", n_instructions=30, has_loops=True),
    ]
    hist = bucket_histogram(recs)
    assert hist[("<=20", False)] == 2
    assert hist[("20-50", True)] == 1
