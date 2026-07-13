from probe.build_corpus import count_instructions, has_loops, _function_name

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
