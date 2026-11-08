"""Test E0 (exp_predict): các hàm dựng dòng CSV theo schema SPEC 10.1 / 10.2, dùng ảnh tổng hợp."""
from __future__ import annotations

import importlib.util
import pathlib

import cv2
import numpy as np
import pytest

from lpr.config import load_config
from lpr.types import Candidate, PlateRecord

# experiments/ không phải package nên nạp run_all.py theo đường dẫn
_spec = importlib.util.spec_from_file_location(
    "run_all", pathlib.Path(__file__).resolve().parents[1] / "experiments" / "run_all.py")
run_all = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_all)

BOX = (260, 200, 480, 270)  # 220×70, đúng tỉ lệ biển


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config("config/default.yaml")


def _record(text: str = "皖A12345") -> PlateRecord:
    x1, y1, x2, y2 = BOX
    quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return PlateRecord(image_id="base_9999", path="", source="ccpd", subset="base", split="test",
                       bbox=BOX, quad=quad, text=text)


def _image(cfg: dict) -> np.ndarray:
    """Ảnh 400×720 nền xám, biển xanh chữ trắng đặt đúng khuôn 7 vị trí."""
    img = np.full((400, 720, 3), 120, np.uint8)
    x1, y1, x2, y2 = BOX
    img[y1:y2, x1:x2] = (160, 60, 20)                      # BGR xanh dương đậm
    scale = cfg["rectify"]["plate_size"][0] / 440
    for lo, hi in cfg["segment"]["char_template_mm"]:
        cx = x1 + (lo + hi) / 2 * scale
        cv2.rectangle(img, (int(cx - 7), y1 + 10), (int(cx + 7), y2 - 11), (255, 255, 255), -1)
    return img


def _cand(bbox, score=1.0) -> Candidate:
    x1, y1, x2, y2 = bbox
    quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return Candidate(bbox=bbox, quad=quad, score=score, method="edge")


# ---------- localization ----------

def test_loc_rows_schema_and_tp(cfg):
    cands = [_cand(BOX, 0.9), _cand((0, 0, 50, 20), 0.5)]
    rows = run_all.localization_rows(_record(), cands, 12.3, "edge", "bilateral", cfg)
    assert [list(r) for r in rows] == [run_all.LOC_COLS] * 2
    assert [r["rank"] for r in rows] == [1, 2]
    assert rows[0]["is_tp"] and rows[0]["iou"] == pytest.approx(1.0)
    assert not rows[1]["is_tp"] and rows[1]["iou"] == 0.0
    assert all(r["ms"] == 12.3 and r["denoise"] == "bilateral" for r in rows)


def test_loc_rows_no_candidate_keeps_image(cfg):
    rows = run_all.localization_rows(_record(), [], 5.0, "edge", "none", cfg)
    assert len(rows) == 1
    assert rows[0]["rank"] == 0 and rows[0]["x1"] is None and rows[0]["is_tp"] is False


def test_loc_rows_capped_at_top_k(cfg):
    cands = [_cand((i, 0, i + 10, 10)) for i in range(10)]
    rows = run_all.localization_rows(_record(), cands, 1.0, "edge", "none", cfg)
    assert len(rows) == cfg["localize"]["top_k"]


# ---------- segmentation ----------

def test_seg_rows_all_combos(cfg):
    rows = run_all.segmentation_rows(_record(), _image(cfg), _cand(BOX), cfg)
    assert len(rows) == 16                       # 2 nguồn × 2 crop × 2 binarize × 2 segment
    assert all(list(r) == run_all.SEG_COLS for r in rows)
    combos = {(r["crop_source"], r["crop_method"], r["binarize"], r["segmenter"]) for r in rows}
    assert len(combos) == 16
    gt_quad = [r for r in rows if r["crop_source"] == "gt" and r["crop_method"] == "quad"]
    assert all(r["n_expected"] == 7 for r in rows)
    assert all(r["all_correct"] for r in gt_quad), gt_quad   # biển sạch: mọi cách đều đúng


def test_seg_rows_pred_without_candidate(cfg):
    rows = run_all.segmentation_rows(_record(), _image(cfg), None, cfg)
    pred = [r for r in rows if r["crop_source"] == "pred"]
    assert len(pred) == 8
    assert all(r["n_boxes"] == 0 and not r["count_match"] and not r["all_correct"] for r in pred)
    assert any(r["all_correct"] for r in rows if r["crop_source"] == "gt")


def test_seg_rows_consistent_columns(cfg):
    for r in run_all.segmentation_rows(_record(), _image(cfg), _cand(BOX), cfg):
        assert 0 <= r["pos1_6_hits"] <= 6
        assert r["count_match"] == (r["n_boxes"] == r["n_expected"])
        if r["all_correct"]:
            assert r["count_match"] and r["pos0_hit"] and r["pos1_6_hits"] == 6
