"""Per-failure alert lead time and false alarms per week (the only numbers allowed on the slides)."""
from __future__ import annotations

import pandas as pd

LOOKBACK = pd.Timedelta(hours=72)   # an act this long before a failure counts as an early warning
AFTER = pd.Timedelta(hours=24)      # repair settling time after a failure ends


def exclusion_zones(failures: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [(r.start - LOOKBACK, r.end + AFTER) for r in failures.itertuples()]


def in_zones(ts: pd.Series, zones) -> pd.Series:
    hit = pd.Series(False, index=ts.index)
    for a, b in zones:
        hit |= (ts >= a) & (ts < b)
    return hit


def lead_times(tl: pd.DataFrame, failures: pd.DataFrame) -> list[dict]:
    """tl: columns ts, alert_level. Lead = hours from first act in [start-LOOKBACK, end] to failure start."""
    out = []
    for r in failures.itertuples():
        w = tl[(tl.ts >= r.start - LOOKBACK) & (tl.ts <= r.end) & (tl.alert_level == "act")]
        if w.empty:
            out.append({"failure_id": r.id, "start": str(r.start), "first_act": None, "lead_hours": None, "result": "missed"})
            continue
        first = w.ts.iloc[0]
        lead = max(0.0, (r.start - first).total_seconds() / 3600)
        out.append({"failure_id": r.id, "start": str(r.start), "first_act": str(first), "lead_hours": round(lead, 1),
                    "result": "early" if lead > 0 else "during"})
    return out


def false_alarms(tl: pd.DataFrame, failures: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Act episodes (rising edges) that start outside every exclusion zone, per week of normal time."""
    seg = tl[(tl.ts >= start) & (tl.ts < end)].reset_index(drop=True)
    is_act = seg.alert_level.eq("act")
    rising = is_act & ~is_act.shift(1, fill_value=False)
    zones = exclusion_zones(failures)
    normal = ~in_zones(seg.ts, zones)
    episodes = int((rising & normal).sum())
    step = pd.Timedelta("10min")
    weeks = normal.sum() * step / pd.Timedelta(days=7)
    return {"episodes": episodes, "normal_weeks": round(float(weeks), 2),
            "per_week": round(episodes / weeks, 2) if weeks > 0 else None,
            "episode_starts": [str(t) for t in seg.ts[rising & normal]]}
