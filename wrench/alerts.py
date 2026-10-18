"""Deterministic alert rule (invariant S2: no language model takes part in deciding an alert).

Per 10-minute step:
  act   if the anomaly score stays >= score_act for `persist` consecutive steps,
        or the largest sensor deviation |z| stays >= z_act for `persist` consecutive steps;
  watch if the score >= score_watch or the largest |z| >= z_watch at this step;
  ok    otherwise. Steps with no data are "ok" and carry no score.
Thresholds are chosen on the validation segment only (see choose_thresholds) and then frozen:
we fix a false-alarm budget per week of normal operation and take the lowest thresholds that meet it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

LEVELS = ("ok", "watch", "act")


@dataclass(frozen=True)
class Thresholds:
    score_watch: float
    score_act: float
    z_watch: float
    z_act: float
    persist: int = 6

    def __post_init__(self):
        if not (self.score_watch <= self.score_act and self.z_watch <= self.z_act and self.persist >= 1):
            raise ValueError(f"inconsistent thresholds: {self}")

    def save(self, path: Path, **meta) -> None:
        path.write_text(json.dumps({**asdict(self), **meta}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Thresholds":
        d = json.loads(path.read_text())
        return cls(**{k: d[k] for k in ("score_watch", "score_act", "z_watch", "z_act", "persist")})


def _sustained(hit: np.ndarray, n: int) -> np.ndarray:
    """True at i when hit[i-n+1..i] are all True."""
    out = np.zeros(len(hit), dtype=bool)
    run = 0
    for i, h in enumerate(hit):
        run = run + 1 if h else 0
        out[i] = run >= n
    return out


def levels(score: np.ndarray, max_abs_z: np.ndarray, t: Thresholds) -> list[str]:
    score = np.asarray(score, dtype=float)
    max_abs_z = np.asarray(max_abs_z, dtype=float)
    s = np.nan_to_num(score, nan=-np.inf)
    z = np.nan_to_num(max_abs_z, nan=-np.inf)
    act = _sustained(s >= t.score_act, t.persist) | _sustained(z >= t.z_act, t.persist)
    watch = (s >= t.score_watch) | (z >= t.z_watch)
    return ["act" if a else "watch" if w else "ok" for a, w in zip(act, watch)]


def _episodes(level_list: list[str], normal: np.ndarray, kind: str = "act") -> int:
    """Rising edges into `kind` (or above) that start in normal time."""
    hit = np.array([LEVELS.index(l) >= LEVELS.index(kind) for l in level_list])
    rising = hit & ~np.concatenate([[False], hit[:-1]])
    return int((rising & normal).sum())


def choose_thresholds(score: np.ndarray, max_abs_z: np.ndarray, normal: np.ndarray, weeks: float,
                      act_per_week: float = 1.0, watch_per_week: float = 7.0, persist: int = 6) -> Thresholds:
    """Pick thresholds on the validation segment only.

    score / max_abs_z: the contiguous validation series; normal: True where a step lies outside every
    failure exclusion zone; weeks: amount of normal time. For each level and each signal we take the
    lowest value on a quantile grid of normal steps whose episodes starting in normal time stay within
    that level's budget per week; half of each budget goes to each signal.
    """
    s = np.asarray(score, dtype=float)
    z = np.asarray(max_abs_z, dtype=float)
    normal = np.asarray(normal, dtype=bool)
    sn, zn = s[normal & ~np.isnan(s)], z[normal & ~np.isnan(z)]
    if len(sn) == 0 or len(zn) == 0 or weeks <= 0:
        raise ValueError("no validation steps to choose thresholds from")
    grid = np.concatenate([np.linspace(0.90, 0.999, 100), [1.0]])
    off = np.full_like(s, np.nan)

    def lowest(values: np.ndarray, as_score: bool, kind: str, per_week: float) -> float:
        allowed = per_week * weeks / 2
        for c in np.quantile(values, grid):
            t = Thresholds(c, c, np.inf, np.inf, persist) if as_score else Thresholds(np.inf, np.inf, c, c, persist)
            lv = levels(s, off, t) if as_score else levels(off, z, t)
            if _episodes(lv, normal, kind) <= allowed:
                return float(c)
        return float(np.nanmax(values)) * 1.01

    s_act, z_act = lowest(sn, True, "act", act_per_week), lowest(zn, False, "act", act_per_week)
    s_watch = min(lowest(sn, True, "watch", watch_per_week), s_act)
    z_watch = min(lowest(zn, False, "watch", watch_per_week), z_act)
    return Thresholds(s_watch, s_act, z_watch, z_act, persist)


def drivers(z_row: np.ndarray, sensors: list[str], k: int = 3) -> list[dict]:
    """The k largest standardised deviations from the healthy baseline."""
    z_row = np.asarray(z_row, dtype=float)
    if np.isnan(z_row).all():
        return []
    order = np.argsort(-np.nan_to_num(np.abs(z_row), nan=-1.0))[:k]
    return [{"sensor": sensors[i], "z": round(float(z_row[i]), 2), "direction": "high" if z_row[i] >= 0 else "low"}
            for i in order if not np.isnan(z_row[i])]
