"""Test phần logic của scripts/label_chars.py (không mở cửa sổ)."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from lpr.config import load_config
from lpr.types import CharBox

_spec = importlib.util.spec_from_file_location(
    "label_chars", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "label_chars.py")
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config("config/default.yaml")


def test_pick_plates_fixed_split_and_balanced(cfg):
    p1, p2 = lc.pick_plates(cfg, 1), lc.pick_plates(cfg, 2)
    assert len(p1) == len(p2) == 15
    assert not {r.image_id for r in p1} & {r.image_id for r in p2}
    assert all(r.split == "test" for r in p1 + p2)
    for part in (p1, p2):
        subs = pd.Series([r.subset for r in part]).value_counts().to_dict()
        assert subs == {"base": 11, "blur": 1, "challenge": 1, "db": 1, "tilt": 1}
    assert [r.image_id for r in lc.pick_plates(cfg, 1)] == [r.image_id for r in p1]   # cố định
    assert len(lc.pick_plates(cfg)) == 30


def test_pick_plates_bad_part(cfg):
    with pytest.raises(ValueError):
        lc.pick_plates(cfg, 3)


def test_replace_boxes_keeps_other_images():
    df = pd.DataFrame([{"image_id": "other", "char_idx": 0, "x1": 1, "y1": 1, "x2": 5, "y2": 5}])
    out = lc.replace_boxes(df, "me", [CharBox(50, 10, 60, 60), CharBox(10, 10, 20, 60)])
    assert list(out.columns) == lc.COLS
    mine = out[out.image_id == "me"]
    assert mine.char_idx.tolist() == [0, 1] and mine.x1.tolist() == [10, 50]    # sắp trái → phải
    assert (out.image_id == "other").sum() == 1
    out2 = lc.replace_boxes(out, "me", [CharBox(0, 0, 9, 9)])                   # gán lại thì thay hẳn
    assert (out2.image_id == "me").sum() == 1


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "sub" / "char_boxes.csv")
    assert lc.load_labels(path).empty
    df = lc.replace_boxes(lc.load_labels(path), "a1", [CharBox(3, 4, 13, 54)])
    lc.save_labels(df, path)
    back = lc.load_labels(path)
    assert lc.boxes_of(back, "a1") == [CharBox(3, 4, 13, 54)]
    assert lc.boxes_of(back, "none") == []


def test_box_from_drag_scales_orders_and_clips():
    b = lc.box_from_drag(80, 200, 40, 40, w=220, h=70, scale=4)      # kéo ngược chiều
    assert b == CharBox(10, 10, 21, 51)
    assert lc.box_from_drag(0, 0, 2000, 2000, w=220, h=70, scale=4) == CharBox(0, 0, 220, 70)
    assert lc.box_from_drag(10, 10, 12, 12, w=220, h=70, scale=4) is None   # bấm nhầm, không kéo


def test_remove_at_smallest_containing():
    big, small = CharBox(0, 0, 50, 50), CharBox(10, 10, 20, 20)
    assert lc.remove_at([big, small], 15, 15) == [big]
    assert lc.remove_at([big, small], 40, 40) == [small]
    assert lc.remove_at([big, small], 100, 100) == [big, small]


def test_suggest_boxes_on_synthetic_plate(cfg):
    plate = np.zeros((70, 220, 3), np.uint8)
    for cx in (19, 47, 87, 115, 144, 172, 201):
        plate[10:60, cx - 7:cx + 7] = 255
    boxes = lc.suggest_boxes(plate, cfg)
    assert len(boxes) == 7
