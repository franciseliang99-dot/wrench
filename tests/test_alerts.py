import numpy as np
import pytest

from wrench.alerts import Thresholds, choose_thresholds, drivers, levels

T = Thresholds(score_watch=1.0, score_act=2.0, z_watch=5.0, z_act=8.0, persist=3)
NAN = np.nan


def test_exactly_at_threshold_counts():
    assert levels([1.0], [0.0], T) == ["watch"]
    assert levels([2.0, 2.0, 2.0], [0, 0, 0], T)[-1] == "act"


def test_just_below_threshold_does_not():
    assert levels([0.999], [4.999], T) == ["ok"]


def test_act_needs_persistence():
    assert levels([2.5, 2.5, 0.0, 2.5, 2.5], [0] * 5, T) == ["watch", "watch", "ok", "watch", "watch"]
    assert levels([2.5] * 4, [0] * 4, T) == ["watch", "watch", "act", "act"]


def test_either_signal_alone_escalates():
    assert levels([0, 0, 0], [9, 9, 9], T)[-1] == "act"      # deviation only
    assert levels([3, 3, 3], [0, 0, 0], T)[-1] == "act"      # score only


def test_signals_do_not_combine_across_steps():
    # score high, then deviation high: neither persists 3 steps on its own
    assert levels([3, 3, 0], [0, 0, 9], T)[-1] == "watch"


def test_missing_data_is_ok_and_breaks_persistence():
    assert levels([2.5, NAN, 2.5, 2.5], [NAN] * 4, T) == ["watch", "ok", "watch", "watch"]


def test_inconsistent_thresholds_rejected():
    with pytest.raises(ValueError):
        Thresholds(3.0, 2.0, 5.0, 8.0)


def test_drivers_are_top3_by_magnitude_with_direction():
    d = drivers(np.array([0.5, -4.0, 2.0, 3.0, NAN]), ["a", "b", "c", "d", "e"])
    assert [x["sensor"] for x in d] == ["b", "d", "c"]
    assert d[0]["direction"] == "low" and d[1]["direction"] == "high"
    assert drivers(np.array([NAN, NAN]), ["a", "b"]) == []


def test_threshold_choice_respects_budget_and_ignores_excluded_steps():
    rng = np.random.default_rng(0)
    s = rng.random(2016) * 0.1            # two weeks of quiet 10-minute steps
    s[100:110] = 5.0                       # one sustained bump in normal time
    s[1500:1600] = 50.0                    # a failure, excluded from normal time
    normal = np.ones(len(s), bool)
    normal[1400:1700] = False
    z = np.zeros(len(s))
    t = choose_thresholds(s, z, normal, weeks=2.0, act_per_week=1.0, watch_per_week=7.0, persist=6)
    lv = levels(s, z, t)
    assert "act" in lv[1500:1600]          # the failure still escalates
    # one bump in two weeks fits the budget, so the threshold stays near the quiet level;
    # counting the excluded failure as a false alarm would push it above the bump (5.0)
    assert t.score_act < 1.0
