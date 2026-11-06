"""Test tách ký tự CCA (SPEC mục 8.10), dùng ảnh nhị phân tổng hợp 220×70."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from lpr.config import load_config
from lpr.evaluate import position_hits, template_intervals
from lpr.segment import merge_split_boxes, segment, segment_cca
from lpr.types import CharBox

H, W = 70, 220
Y0, Y1 = 10, 60          # ký tự cao 50 px ≈ 0.71 chiều cao biển


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config("config/default.yaml")


def _centers(cfg: dict) -> list[int]:
    return [int(round((a + b) / 2)) for a, b in template_intervals(cfg)]


def _plate(cfg: dict, *, frame=False, specks=False, bolt=False,
           broken_lr=False, broken_tb=False) -> np.ndarray:
    """7 ký tự đặc rộng 16 px đặt đúng tâm khuôn; tùy chọn thêm viền, nhiễu, ký tự vỡ."""
    img = np.zeros((H, W), np.uint8)
    for i, cx in enumerate(_centers(cfg)):
        x1, x2 = cx - 8, cx + 8
        if i == 0 and broken_lr:                  # ký tự Hán vỡ trái / phải, khe 3 px
            img[Y0:Y1, x1:x1 + 7] = 255
            img[Y0:Y1, x1 + 10:x2] = 255
        elif i == 3 and broken_tb:                # nét đứt ngang giữa thân
            img[Y0:33, x1:x2] = 255
            img[37:Y1, x1:x2] = 255
        else:
            img[Y0:Y1, x1:x2] = 255
    if frame:
        cv2.rectangle(img, (1, 1), (W - 2, H - 2), 255, 2)
    if specks:
        for x, y in [(33, 35), (70, 20), (130, 64), (100, 5)]:
            img[y:y + 2, x:x + 2] = 255
    if bolt:
        cv2.circle(img, (60, 5), 3, 255, -1)
        cv2.circle(img, (160, 65), 3, 255, -1)
    return img


def test_clean_plate_seven_boxes_on_template(cfg):
    boxes = segment(_plate(cfg), "cca", cfg)
    assert len(boxes) == 7
    assert all(position_hits(boxes, cfg))
    assert [b.x1 for b in boxes] == sorted(b.x1 for b in boxes)


def test_frame_specks_bolts_removed(cfg):
    boxes = segment(_plate(cfg, frame=True, specks=True, bolt=True), "cca", cfg)
    assert len(boxes) == 7
    assert all(position_hits(boxes, cfg))


def test_broken_left_right_char_merged(cfg):
    boxes = segment(_plate(cfg, broken_lr=True), "cca", cfg)
    assert len(boxes) == 7
    assert position_hits(boxes, cfg)[0]


def test_broken_top_bottom_char_merged(cfg):
    boxes = segment_cca(_plate(cfg, broken_tb=True), cfg)
    assert len(boxes) == 7                        # hai nửa trên/dưới gộp ngay trong CCA
    assert boxes[3].y1 == Y0 and boxes[3].y2 == Y1


def test_all_noise_combined(cfg):
    img = _plate(cfg, frame=True, specks=True, bolt=True, broken_lr=True, broken_tb=True)
    boxes = segment(img, "cca", cfg)
    assert len(boxes) == 7
    assert all(position_hits(boxes, cfg))


def test_merge_split_cuts_double_width_box():
    boxes = [CharBox(10, 10, 26, 60), CharBox(40, 10, 56, 60), CharBox(70, 10, 104, 60),
             CharBox(120, 10, 136, 60)]
    out = merge_split_boxes(boxes, W, H)
    assert len(out) == 5
    assert (out[2].x1, out[3].x2) == (70, 104)
    assert out[2].x2 == out[3].x1


def test_merge_split_keeps_narrow_one_apart():
    """Số "1" hẹp nhưng cách xa ký tự bên cạnh: không được gộp."""
    boxes = [CharBox(10, 10, 26, 60), CharBox(40, 10, 46, 60), CharBox(60, 10, 76, 60),
             CharBox(90, 10, 106, 60)]
    assert len(merge_split_boxes(boxes, W, H)) == 4


def test_merge_split_few_boxes_copied():
    boxes = [CharBox(10, 10, 60, 60), CharBox(70, 10, 76, 60)]
    out = merge_split_boxes(boxes, W, H)
    assert out == boxes and out[0] is not boxes[0]


def test_empty_image(cfg):
    assert segment(np.zeros((H, W), np.uint8), "cca", cfg) == []


def test_input_not_modified(cfg):
    img = _plate(cfg, frame=True, broken_lr=True)
    img0 = img.copy()
    segment(img, "cca", cfg)
    assert np.array_equal(img, img0)


def test_projection_not_implemented(cfg):
    with pytest.raises(NotImplementedError):
        segment(_plate(cfg), "projection", cfg)


def test_bad_method(cfg):
    with pytest.raises(ValueError):
        segment(_plate(cfg), "xyz", cfg)  # type: ignore[arg-type]
