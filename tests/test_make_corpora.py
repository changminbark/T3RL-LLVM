import json

from probe.make_corpora import (
    called_functions,
    introduces_calls,
    is_driver_main,
    load_verified,
    select_report,
    select_train,
)
from probe.schema import CorpusRecord


def _rec(fid, *, n=10, loops=False, o3="", mca=1.0, src="") -> CorpusRecord:
    return CorpusRecord(
        function_id=fid,
        src_ir=src or "define i32 @f() {\n ret i32 0\n}",
        n_instructions=n,
        has_loops=loops,
        o3_baseline_ir=o3 or "define i32 @f() {\n ret i32 0\n}",
        mca_cycles_o3=mca,
    )


def test_called_functions_ignores_intrinsics_and_indirect():
    ir = """
      %r = call i32 @foo(i32 1)
      call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 8, i1 false)
      %s = call i32 %fnptr(i32 2)
    """
    assert called_functions(ir) == {"foo"}


def test_introduces_calls_catches_the_libc_idiom():
    # The strlen case: -O0 is a hand-written loop, -O3 calls strlen.
    rec = _rec(
        "builtins/lib/strlen::strlen",
        src="define i64 @strlen(ptr %s) {\n br label %loop\n}",
        o3="define i64 @strlen(ptr %s) {\n %n = call i64 @strlen(ptr %s)\n ret i64 %n\n}",
    )
    assert introduces_calls(rec)


def test_introduces_calls_allows_calls_present_in_both():
    rec = _rec(
        "a::f",
        src="define void @f() {\n call void @g()\n ret void\n}",
        o3="define void @f() {\n call void @g()\n ret void\n}",
    )
    assert not introduces_calls(rec)


def test_is_driver_main():
    assert is_driver_main(_rec("Regression/C/x::main"))
    assert not is_driver_main(_rec("Regression/C/x::main_test"))


def test_select_train_drop_reasons_are_exclusive_and_counted():
    records = [
        _rec("ok::good"),
        _rec("x::main"),
        _rec("y::f", mca=None),
        _rec("big::f", n=400),
        _rec(
            "libc::strlen",
            src="define i64 @strlen(ptr %s) {\n ret i64 0\n}",
            o3="define i64 @strlen(ptr %s) {\n %n = call i64 @strlen(ptr %s)\n ret i64 %n\n}",
        ),
    ]
    kept, dropped = select_train(records, verified=None)
    assert [r.function_id for r in kept] == ["ok::good"]
    assert sum(dropped.values()) == 4
    assert dropped["driver main"] == 1
    assert dropped[">150 instrs (oracle ~1-9%, p90 22s)"] == 1


def test_select_train_filters_to_verified_when_given():
    records = [_rec("a::f"), _rec("b::f")]
    kept, dropped = select_train(records, verified={"a::f"})
    assert [r.function_id for r in kept] == ["a::f"]
    assert dropped["oracle could not verify -O0 == -O3"] == 1


def test_select_report_caps_dominant_stratum_and_keeps_scarce_ones():
    records = [_rec(f"tiny{i}::f", n=5) for i in range(100)]
    records += [_rec(f"loopy{i}::f", n=30, loops=True) for i in range(3)]
    kept, sizes = select_report(records, cap=10, seed=0)
    assert sizes["<=20|loops=False"] == 10
    assert sizes["20-50|loops=True"] == 3
    assert len(kept) == 13


def test_select_report_is_deterministic_for_a_seed():
    records = [_rec(f"t{i}::f", n=5) for i in range(50)]
    a, _ = select_report(records, cap=7, seed=42)
    b, _ = select_report(records, cap=7, seed=42)
    c, _ = select_report(records, cap=7, seed=1)
    assert [r.function_id for r in a] == [r.function_id for r in b]
    assert [r.function_id for r in a] != [r.function_id for r in c]


def test_load_verified_selects_only_verified(tmp_path):
    p = tmp_path / "verdicts.jsonl"
    p.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [
                {"function_id": "a", "status": "verified"},
                {"function_id": "b", "status": "counterexample"},
                {"function_id": "c", "status": "failed_to_prove"},
            ]
        )
    )
    assert load_verified(p) == {"a"}
