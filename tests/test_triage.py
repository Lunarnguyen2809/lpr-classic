"""Test hàm phân loại triage (experiments/triage_failures.py) bằng dòng tổng hợp, không cần dữ liệu thật."""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "triage_failures", pathlib.Path(__file__).resolve().parents[1] / "experiments" / "triage_failures.py")
tri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tri)

THR = 5.0


def _stage(any_tp=True, top1_tp=True, ce=1.0, count_ok=True, all_correct=True) -> str:
    return tri.classify_stage(any_tp, top1_tp, ce, THR, count_ok, all_correct)


@pytest.mark.parametrize("kwargs,expected", [
    (dict(any_tp=False, top1_tp=False, ce=None, count_ok=False, all_correct=False), "loc_no_candidate"),
    (dict(top1_tp=False, ce=None, count_ok=False, all_correct=False), "loc_rank"),
    (dict(top1_tp=False, ce=None, count_ok=True, all_correct=True), "loc_rank"),   # không có ok do may mắn
    (dict(any_tp=False, top1_tp=False, ce=None, count_ok=True, all_correct=True), "loc_no_candidate"),
    (dict(ce=THR + 0.1, count_ok=False, all_correct=False), "rectify"),
    (dict(ce=THR + 0.1, count_ok=True, all_correct=True), "ok"),   # nắn lệch nhưng tách đúng: không phải lỗi
    (dict(ce=THR, count_ok=False, all_correct=False), "segment_count"),   # đúng ngưỡng thì chưa tính là lệch
    (dict(count_ok=True, all_correct=False), "segment_position"),
    (dict(), "ok"),
    (dict(ce=None, count_ok=False, all_correct=False), "segment_count"),
])
def test_classify_stage(kwargs, expected):
    assert _stage(**kwargs) == expected


def test_classify_segment():
    assert tri.classify_segment(True, True) == "ok"
    assert tri.classify_segment(False, False) == "segment_count"
    assert tri.classify_segment(True, False) == "segment_position"


@pytest.mark.parametrize("ratio,expected", [
    (0.0079, "gt_too_small"), (0.008, "other"), (0.05, "other"), (0.16, "other"), (0.1601, "gt_too_large")])
def test_loc_subtag(ratio, expected):
    assert tri.loc_subtag(ratio, 0.008, 0.16) == expected


def test_tables_are_exclusive_and_complete():
    df = pd.DataFrame({
        "image_id": list("abcd"), "subset": ["base", "base", "tilt", "tilt"],
        "stage": ["ok", "loc_rank", "loc_no_candidate", "segment_count"],
        "subtag": ["", "", "gt_too_small", ""], "gt_status": ["ok", "ok", "segment_count", "segment_count"]})
    c = tri.stage_counts(df)
    for sub, n in [("base", 2), ("tilt", 2), ("ALL", 4)]:
        g = c[c["subset"] == sub]
        assert g["n"].sum() == n and (g["n_subset"] == n).all()
    x = tri.cross_table(df)
    assert x[x["subset"] == "ALL"]["n"].sum() == 4
    assert x[(x["subset"] == "ALL") & (x["gt_status"] == "segment_count")
             & (x["pred_stage"] == "loc_no_candidate")]["n"].iloc[0] == 1
    s = tri.subtag_table(df)
    assert s[(s["subset"] == "ALL") & (s["subtag"] == "gt_too_small")]["n"].iloc[0] == 1
