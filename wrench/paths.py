"""Where things live. Raw data never lives in the repo (it is downloaded to WRENCH_DATA)."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_SPEC = REPO / "data"


def data_dir() -> Path:
    """Directory holding the downloaded dataset. Override with WRENCH_DATA."""
    return Path(os.environ.get("WRENCH_DATA", "~/.cache/wrench-data")).expanduser()
