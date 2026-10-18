"""Fixed time split and the guard that keeps evaluation data out of training (invariant S5)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import DATA_SPEC

SEGMENTS = ("train", "validation", "evaluation")


class SplitLeak(RuntimeError):
    """A step read samples from a segment it is not allowed to see."""


@dataclass(frozen=True)
class Segment:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive

    def mask(self, ts: pd.Series) -> pd.Series:
        return (ts >= self.start) & (ts < self.end)


def load_splits(path: Path | None = None) -> dict[str, Segment]:
    raw = json.loads((path or DATA_SPEC / "splits.json").read_text())
    segs = {n: Segment(n, pd.Timestamp(raw[n]["start"]), pd.Timestamp(raw[n]["end"])) for n in SEGMENTS}
    ordered = [segs[n] for n in SEGMENTS]
    for a, b in zip(ordered, ordered[1:]):
        if a.end != b.start or a.start >= a.end:
            raise ValueError(f"splits must be contiguous and ordered: {a.name} -> {b.name}")
    return segs


def load_failures(path: Path | None = None) -> pd.DataFrame:
    raw = json.loads((path or DATA_SPEC / "failures.json").read_text())
    df = pd.DataFrame(raw["failures"])
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def take(df: pd.DataFrame, allowed: set[str], segs: dict[str, Segment], ts_col: str = "timestamp") -> pd.DataFrame:
    """Return the rows of df inside the allowed segments.

    Raises SplitLeak if df contains any row from a segment that is not allowed:
    callers must slice before they hand data over, so a leak is a bug, not a filter.
    """
    unknown = set(allowed) - set(SEGMENTS)
    if unknown:
        raise ValueError(f"unknown segments: {sorted(unknown)}")
    ts = df[ts_col]
    for name in SEGMENTS:
        if name not in allowed and segs[name].mask(ts).any():
            raise SplitLeak(f"{int(segs[name].mask(ts).sum())} rows from the {name} segment reached a step that may only see {sorted(allowed)}")
    keep = pd.Series(False, index=df.index)
    for name in allowed:
        keep |= segs[name].mask(ts)
    return df[keep]


def slice_segment(df: pd.DataFrame, name: str, segs: dict[str, Segment], ts_col: str = "timestamp") -> pd.DataFrame:
    return df[segs[name].mask(df[ts_col])]
