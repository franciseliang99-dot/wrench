"""Shrink artifacts/timeline.json (6 MB, not committed) into artifacts/timeline.min.json for the web app.

Columnar: one start time and a fixed step, then per-step score, level letter and drivers as [sensor index, z].
The source has gaps (no data); the output is padded to a complete grid and missing steps get level "-",
score null and no drivers. Direction is the sign of z. Nothing is recomputed here; values are copied and rounded.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrench.features import SENSORS, STEP  # noqa: E402
from wrench.paths import REPO  # noqa: E402

SRC = REPO / "artifacts" / "timeline.json"
OUT = REPO / "artifacts" / "timeline.min.json"
LETTER = {"ok": "o", "watch": "w", "act": "a"}


def main() -> int:
    rows = json.loads(SRC.read_text())
    units = {r["unit_id"] for r in rows}
    if len(units) != 1:
        raise SystemExit(f"expected one unit, got {sorted(units)}")
    idx = {s: i for i, s in enumerate(SENSORS)}
    step = timedelta(seconds=600)
    t0 = datetime.fromisoformat(rows[0]["ts"])
    n = int((datetime.fromisoformat(rows[-1]["ts"]) - t0) / step) + 1
    score, level, drivers = [None] * n, ["-"] * n, [[] for _ in range(n)]
    for r in rows:
        off = datetime.fromisoformat(r["ts"]) - t0
        if off % step:
            raise SystemExit(f"{r['ts']} is not on the {STEP} grid")
        i = off // step
        score[i] = None if r["anomaly_score"] is None else round(r["anomaly_score"], 3)
        level[i] = LETTER[r["alert_level"]]
        drivers[i] = [[idx[d["sensor"]], round(d["z"], 2)] for d in r["drivers"]]
    out = {
        "unit_id": rows[0]["unit_id"],
        "start": rows[0]["ts"],
        "step_seconds": 600,
        "sensors": SENSORS,
        "score": score,
        "level": "".join(level),
        "drivers": drivers,
        "segments": json.loads((REPO / "data" / "splits.json").read_text()),
        "failures": json.loads((REPO / "data" / "failures.json").read_text())["failures"],
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"{len(rows)} steps, {n} grid slots -> {OUT.name} {OUT.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
