"""Download MetroPT-3 from UCI, check its sha256, unpack to WRENCH_DATA, write a parquet copy.

Dataset: MetroPT-3, UCI Machine Learning Repository, DOI 10.24432/C5VW3R, licensed CC BY 4.0.
Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., Gama, J. (2021). Predictive maintenance
based on anomaly detection using deep learning for air production unit in the railway industry.
IEEE DSAA 2021. The data is not stored in this repository.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wrench.paths import data_dir  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
SHA256 = "aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"
CSV_NAME = "MetroPT3(AirCompressor).csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / "metropt3.zip"
    if not zpath.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
    got = sha256(zpath)
    if got != SHA256:
        print(f"sha256 mismatch: expected {SHA256}, got {got}", file=sys.stderr)
        return 1
    raw = root / "raw"
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            target = (raw / info.filename).resolve()
            if not target.is_relative_to(raw.resolve()):
                print(f"refusing unsafe path in zip: {info.filename}", file=sys.stderr)
                return 1
        z.extractall(raw)
    csv = next(raw.rglob(CSV_NAME), None)
    if csv is None:
        print(f"{CSV_NAME} not found in the archive", file=sys.stderr)
        return 1
    import pandas as pd

    df = pd.read_csv(csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.to_parquet(root / "metropt3.parquet")
    print(f"ok: {len(df)} rows, {df['timestamp'].min()} .. {df['timestamp'].max()} -> {root / 'metropt3.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
