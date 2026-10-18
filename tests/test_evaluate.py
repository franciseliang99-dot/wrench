import pandas as pd

from wrench.evaluate import false_alarms, lead_times


def timeline(levels_by_time):
    ts = pd.date_range("2020-06-01", periods=len(levels_by_time), freq="10min")
    return pd.DataFrame({"ts": ts, "alert_level": levels_by_time})


def failures(start, end):
    return pd.DataFrame([{"id": 1, "start": pd.Timestamp(start), "end": pd.Timestamp(end)}])


def test_lead_time_from_first_act_in_lookback():
    tl = timeline(["ok"] * 6 + ["act"] * 3 + ["ok"] * 3)       # act from 01:00
    r = lead_times(tl, failures("2020-06-01 02:00", "2020-06-01 03:00"))[0]
    assert r["result"] == "early" and r["lead_hours"] == 1.0


def test_act_after_start_is_during_with_zero_lead():
    tl = timeline(["ok"] * 13 + ["act"] * 2)
    r = lead_times(tl, failures("2020-06-01 02:00", "2020-06-01 03:00"))[0]
    assert r["result"] == "during" and r["lead_hours"] == 0.0


def test_no_act_is_reported_as_missed():
    r = lead_times(timeline(["watch"] * 20), failures("2020-06-01 02:00", "2020-06-01 03:00"))[0]
    assert r["result"] == "missed" and r["lead_hours"] is None


def test_false_alarms_count_episodes_outside_zones_only():
    lv = ["ok"] * 1008
    lv[10:20] = ["act"] * 10          # one episode in normal time
    lv[30:32] = ["act"] * 2           # another one
    lv[900:905] = ["act"] * 5         # inside the zone around a failure
    tl = timeline(lv)
    f = failures("2020-06-07 06:00", "2020-06-07 07:00")
    out = false_alarms(tl, f, tl.ts.min(), tl.ts.max() + pd.Timedelta("10min"))
    assert out["episodes"] == 2
