"""Train the autoencoder on the training segment, pick thresholds on validation, write the timeline and the evaluation.

Runs on CPU or on an AMD Instinct GPU (PyTorch ROCm exposes it as the "cuda" device):
    python scripts/build_timeline.py --device auto --final
Outputs (in artifacts/): timeline.json, thresholds.json, and validation.json (or evaluation.json with --final). Model weights go to WRENCH_DATA.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wrench import alerts, evaluate, features, model  # noqa: E402
from wrench.paths import REPO, data_dir  # noqa: E402
from wrench.splits import load_failures, load_splits, slice_segment, take  # noqa: E402

OUT = REPO / "artifacts"
ACT_PER_WEEK = 1.0    # act episodes allowed per week of normal operation, on validation
WATCH_PER_WEEK = 7.0  # watch episodes allowed per week (about one a day)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true",
                    help="also score the evaluation segment; without it only validation numbers are reported")
    args = ap.parse_args(argv)

    segs, failures = load_splits(), load_failures()
    raw = pd.read_parquet(data_dir() / "metropt3.parquet")
    minutes = features.minute_frame(raw)
    steps = features.step_frame(minutes)

    # S5: the model and the scaler only ever see the training segment
    m_df = minutes.reset_index()
    train_min = take(slice_segment(m_df, "train", segs), {"train"}, segs)
    scaler = features.Scaler.fit(train_min)
    z_min = scaler.transform(m_df)
    train_idx = np.flatnonzero(m_df.timestamp.isin(train_min.timestamp).to_numpy())
    x_train, _ = model.windows(z_min, train_idx)

    device = model.pick_device(args.device)
    net = model.build(len(features.SENSORS), seed=args.seed)
    t0 = time.time()
    losses = model.train(net, x_train, epochs=args.epochs, device=device, seed=args.seed)
    train_s = time.time() - t0
    torch.save(net.state_dict(), data_dir() / "autoencoder.pt")

    # score: the window ending at the last minute of each 10-minute step
    s_df = steps.reset_index()
    step_last_min = s_df.timestamp + pd.Timedelta(minutes=9)
    pos = pd.Index(m_df.timestamp).get_indexer(step_last_min)
    x_all, kept = model.windows(z_min, pos[pos >= 0])
    t0 = time.time()
    sc = model.score(net, x_all, device)
    score_s = time.time() - t0
    score = np.full(len(s_df), np.nan)
    where = {p: i for i, p in enumerate(pos)}
    for p, v in zip(kept, sc):
        score[where[p]] = v

    # deviation from the healthy baseline (training-segment step means)
    base = features.Scaler.fit(take(slice_segment(s_df, "train", segs), {"train"}, segs).dropna())
    z_step = base.transform(s_df)
    max_abs_z = np.nanmax(np.abs(z_step), axis=1, initial=-np.inf)
    max_abs_z[np.isinf(max_abs_z)] = np.nan

    # thresholds: validation segment only, outside failure exclusion zones
    val = take(slice_segment(s_df.assign(score=score, mz=max_abs_z), "validation", segs), {"validation"}, segs)
    val_normal = ~evaluate.in_zones(val.timestamp, evaluate.exclusion_zones(failures)).to_numpy()
    val_weeks = (val_normal & (~val.score.isna() | ~val.mz.isna()).to_numpy()).sum() * pd.Timedelta(features.STEP) / pd.Timedelta(days=7)
    thr = alerts.choose_thresholds(val.score.to_numpy(), val.mz.to_numpy(), val_normal, val_weeks,
                                   act_per_week=ACT_PER_WEEK, watch_per_week=WATCH_PER_WEEK)
    lv = alerts.levels(score, max_abs_z, thr)

    has_data = ~np.isnan(score) | ~np.isnan(max_abs_z)
    timeline = [
        {"unit_id": features.UNIT_ID, "ts": ts.isoformat(), "anomaly_score": None if np.isnan(s) else round(float(s), 4),
         "alert_level": l, "drivers": alerts.drivers(zr, features.SENSORS)}
        for ts, s, l, zr, ok in zip(s_df.timestamp, score, lv, z_step, has_data) if ok
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / "timeline.json").write_text(json.dumps(timeline, separators=(",", ":")))
    thr.save(OUT / "thresholds.json", chosen_on="validation segment; lowest thresholds within each level's episode budget",
             budget_per_week={"act": ACT_PER_WEEK, "watch": WATCH_PER_WEEK})

    tl = pd.DataFrame({"ts": [pd.Timestamp(r["ts"]) for r in timeline], "alert_level": [r["alert_level"] for r in timeline]})
    seg_of = {f.id: next(n for n, s in segs.items() if s.start <= f.start < s.end) for f in failures.itertuples()}
    leads = evaluate.lead_times(tl, failures)
    for row in leads:
        row["segment"] = seg_of[row["failure_id"]]
    shown = ("validation", "evaluation") if args.final else ("validation",)
    leads = [r for r in leads if r["segment"] in shown]
    report = {
        "mode": "final" if args.final else "development (evaluation segment not reported)",
        "lead_times": leads,
        "false_alarms": {n: evaluate.false_alarms(tl, failures, segs[n].start, segs[n].end) for n in shown},
        "definitions": {"lookback_hours": evaluate.LOOKBACK / pd.Timedelta(hours=1),
                        "after_failure_hours": evaluate.AFTER / pd.Timedelta(hours=1), "step": features.STEP},
        "training": {"device": str(device), "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
                     "epochs": args.epochs, "train_windows": int(len(x_train)), "final_loss": round(losses[-1], 5),
                     "train_seconds": round(train_s, 1), "scored_windows": int(len(x_all)), "score_seconds": round(score_s, 1),
                     "torch": torch.__version__},
    }
    (OUT / ("evaluation.json" if args.final else "validation.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("lead_times", "false_alarms")}, indent=1, default=str))
    print(f"thresholds: {thr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
