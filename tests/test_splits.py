import json

import pandas as pd
import pytest

from wrench.splits import SplitLeak, load_failures, load_splits, take


def frame(*stamps):
    return pd.DataFrame({"timestamp": pd.to_datetime(list(stamps)), "x": range(len(stamps))})


def test_committed_splits_are_contiguous_and_failures_are_where_we_say():
    segs = load_splits()
    fails = load_failures()
    assert len(fails) == 4
    where = [next(n for n, s in segs.items() if s.start <= f < s.end) for f in fails["start"]]
    assert where == ["validation", "evaluation", "evaluation", "evaluation"]
    assert not any(segs["train"].start <= f < segs["train"].end for f in fails["end"])


def test_training_step_refuses_evaluation_rows():
    segs = load_splits()
    df = frame("2020-02-10", "2020-06-01")
    with pytest.raises(SplitLeak):
        take(df, {"train"}, segs)


def test_boundary_row_belongs_to_the_later_segment():
    segs = load_splits()
    with pytest.raises(SplitLeak):
        take(frame("2020-04-01T00:00:00"), {"train"}, segs)
    assert len(take(frame("2020-03-31T23:59:50"), {"train"}, segs)) == 1


def test_allowed_rows_pass_through():
    segs = load_splits()
    out = take(frame("2020-02-10", "2020-04-10"), {"train", "validation"}, segs)
    assert len(out) == 2


def test_gap_between_segments_is_rejected(tmp_path):
    bad = json.loads(load_splits.__globals__["DATA_SPEC"].joinpath("splits.json").read_text())
    bad["validation"]["start"] = "2020-04-02T00:00:00"
    p = tmp_path / "s.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_splits(p)
