"""Sensor frames: 1-minute means for the autoencoder, 10-minute means for the timeline and drivers."""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current",
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses",
]
UNIT_ID = "C-01"  # the dataset has one compressor; this label is ours
STEP = "10min"
STD_FLOOR = 0.1  # fraction of a sensor's training range


def minute_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """1-minute means on a complete grid; minutes with no reading are NaN (gaps stay visible)."""
    m = raw.set_index("timestamp")[SENSORS].resample("1min").mean()
    return m


def step_frame(minutes: pd.DataFrame) -> pd.DataFrame:
    """10-minute means; a step needs at least half its minutes, otherwise NaN."""
    agg = minutes.resample(STEP)
    means = agg.mean()
    counts = agg.count().min(axis=1)
    means[counts < 5] = np.nan
    return means


class Scaler:
    """Per-sensor standardisation fitted on training data only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.where(np.asarray(std, dtype=np.float64) < 1e-6, 1.0, std)

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Scaler":
        """Mean and std per sensor, with std floored at STD_FLOOR of the sensor's training range.

        Without the floor a channel that is almost constant in training (a switch that is nearly
        always on, a valve pressure that is nearly always zero) gets a tiny std, and any small
        change on it produces a huge z that drowns every other signal.
        """
        x = frame[SENSORS].dropna().to_numpy()
        lo, hi = np.quantile(x, 0.001, axis=0), np.quantile(x, 0.999, axis=0)
        span = np.maximum(hi - lo, x.max(axis=0) - x.min(axis=0))
        return cls(x.mean(axis=0), np.maximum(x.std(axis=0), STD_FLOOR * span))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return (frame[SENSORS].to_numpy() - self.mean) / self.std

    def to_json(self) -> dict:
        return {"sensors": SENSORS, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "Scaler":
        if d["sensors"] != SENSORS:
            raise ValueError("scaler was fitted on a different sensor list")
        return cls(d["mean"], d["std"])
