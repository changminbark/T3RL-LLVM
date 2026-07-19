import math
import pytest

from probe.bestofk import (
    passk_estimator,
    expected_best_speedup_at_k,
    per_function_speedups,
    curve,
)
from probe.schema import RewriteResult, RewriteOutcome


def test_passk_known_values():
    assert passk_estimator(8, 0, 1) == 0.0          # no successes -> never covered
    assert passk_estimator(8, 8, 1) == 1.0          # all successes -> always covered
    assert passk_estimator(4, 1, 2) == pytest.approx(0.5)      # 1 - C(3,2)/C(4,2) = 1 - 3/6
    assert passk_estimator(4, 2, 2) == pytest.approx(5 / 6)    # 1 - C(2,2)/C(4,2) = 1 - 1/6


def test_passk_k_ge_available_nonsuccess():
    # If k covers more than the non-successes, coverage is certain.
    assert passk_estimator(4, 2, 3) == 1.0          # any 3 of 4 must include a success


def test_expected_best_speedup_two_samples():
    # speedups [2.0, 1.0]: k=1 -> mean 1.5 ; k=2 -> max 2.0
    assert expected_best_speedup_at_k([2.0, 1.0], 1) == pytest.approx(1.5)
    assert expected_best_speedup_at_k([2.0, 1.0], 2) == pytest.approx(2.0)


def test_expected_best_speedup_k_ge_n():
    assert expected_best_speedup_at_k([1.3, 1.1, 1.0], 5) == pytest.approx(1.3)


def _rec(idx, outcome, speedup):
    return RewriteResult(
        function_id="f", sample_index=idx, outcome=outcome, speedup_vs_o3=speedup
    )


def test_per_function_speedups_mapping():
    recs = [
        _rec(0, RewriteOutcome.verified_faster, 1.5),
        _rec(1, RewriteOutcome.verified_no_gain, None),
        _rec(2, RewriteOutcome.invalid_syntax, None),
        _rec(3, RewriteOutcome.verified_faster, 0.9),  # capped up to 1.0 by max()
    ]
    assert per_function_speedups(recs) == [1.5, 1.0, 1.0, 1.0]


def _recs_for(fid, outcomes_speedups):
    return [
        _rec_named(fid, i, o, s) for i, (o, s) in enumerate(outcomes_speedups)
    ]


def _rec_named(fid, idx, outcome, speedup):
    return RewriteResult(
        function_id=fid, sample_index=idx, outcome=outcome, speedup_vs_o3=speedup
    )


def test_curve_overall_and_buckets():
    VF = RewriteOutcome.verified_faster
    NG = RewriteOutcome.verified_no_gain
    # f1: 2 samples, one is 2.0x faster ; f2: 2 samples, none beat O3
    by_fn = {
        "f1": _recs_for("f1", [(VF, 2.0), (NG, None)]),
        "f2": _recs_for("f2", [(NG, None), (NG, None)]),
    }
    buckets = {"f1": ("<=20", False), "f2": ("<=20", False)}
    out = curve(by_fn, buckets, ks=[1, 2])

    # Coverage@1: f1 has c=1 of n=2 -> passk(2,1,1)=0.5 ; f2 -> 0. Mean over 2 fns = 0.25
    assert out["overall"][1]["coverage"] == pytest.approx(0.25)
    # Coverage@2: f1 -> 1.0 ; f2 -> 0 ; mean = 0.5
    assert out["overall"][2]["coverage"] == pytest.approx(0.5)
    # MeanSpeedup@1: f1 expected best of 1 over [2.0,1.0] = 1.5 ; f2 = 1.0 ; mean = 1.25
    assert out["overall"][1]["mean_speedup"] == pytest.approx(1.25)
    # MeanSpeedup@2: f1 = 2.0 ; f2 = 1.0 ; mean = 1.5
    assert out["overall"][2]["mean_speedup"] == pytest.approx(1.5)
    assert out["overall"][1]["n_functions"] == 2
    key = "<=20|loops=False"
    assert out["by_bucket"][key][2]["coverage"] == pytest.approx(0.5)
