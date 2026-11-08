"""Test tách ký tự CCA (SPEC mục 8.10), dùng ảnh nhị phân tổng hợp 220×70."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from lpr.config import load_config
from lpr.evaluate import char_box_prf, count_match, position_hits, template_intervals
from lpr.segment import merge_split_boxes, segment, segment_cca, segment_projection, split_lines
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


# ---------- split_lines ----------

def test_split_lines_single_line_trims_frame(cfg):
    lines = split_lines(_plate(cfg, frame=True))
    assert len(lines) == 1
    y0, y1 = lines[0]
    assert y0 <= Y0 and y1 >= Y1 and y1 - y0 < H - 4    # bỏ hàng viền trên/dưới


def test_split_lines_two_lines():
    img = np.zeros((H, W), np.uint8)
    for x in range(20, 200, 30):
        img[5:30, x:x + 14] = 255      # dòng trên
        img[40:65, x:x + 14] = 255     # dòng dưới
    assert split_lines(img) == [(5, 30), (40, 65)]


def test_split_lines_empty_returns_whole():
    assert split_lines(np.zeros((H, W), np.uint8)) == [(0, H)]


# ---------- segment_projection ----------

@pytest.mark.parametrize("kw", [{}, {"frame": True, "specks": True, "bolt": True},
                                {"broken_lr": True}, {"broken_tb": True}])
def test_projection_seven_boxes(cfg, kw):
    boxes = segment(_plate(cfg, **kw), "projection", cfg)
    assert len(boxes) == 7
    assert all(position_hits(boxes, cfg))


def test_projection_cuts_touching_chars_at_valley(cfg):
    img = _plate(cfg)
    c1, c2 = _centers(cfg)[2], _centers(cfg)[3]
    img[30:33, c1 + 8:c2 - 8] = 255            # cầu nối mảnh giữa ký tự 2 và 3
    raw = segment_projection(img, cfg)
    assert len(raw) == 7
    cut = [b for b in raw if b.x1 >= c1 - 8][1].x1
    assert c1 + 8 <= cut <= c2 - 8             # đường cắt rơi vào khe, không cắt ngang ký tự


def test_projection_cuts_wide_blob_at_valley(cfg):
    """Hai ký tự dính bằng cầu nối dày (cột chữ liền), phải cắt tại cột ít trắng nhất."""
    img = _plate(cfg)
    c1, c2 = _centers(cfg)[2], _centers(cfg)[3]
    img[25:45, c1 + 8:c2 - 8] = 255
    boxes = segment(img, "projection", cfg)
    assert len(boxes) == 7
    assert all(position_hits(boxes, cfg))


def test_projection_input_not_modified(cfg):
    img = _plate(cfg, frame=True, bolt=True)
    img0 = img.copy()
    segment(img, "projection", cfg)
    assert np.array_equal(img, img0)


def test_projection_two_line_plate_line_index(cfg):
    img = np.zeros((H, W), np.uint8)
    for x in range(20, 200, 30):
        img[5:30, x:x + 14] = 255
        img[40:65, x:x + 14] = 255
    boxes = segment(img, "projection", cfg)
    assert {b.line for b in boxes} == {0, 1}
    assert [b.line for b in boxes] == sorted(b.line for b in boxes)


# ---------- nối với evaluate ----------

@pytest.mark.parametrize("method", ["cca", "projection"])
def test_metrics_on_segment_output(cfg, method):
    gt = [CharBox(int(c) - 8, Y0, int(c) + 8, Y1) for c in _centers(cfg)]
    pred = segment(_plate(cfg, frame=True, specks=True), method, cfg)
    assert count_match(pred, "皖A12345")
    assert position_hits(pred, cfg) == [True] * 7
    assert char_box_prf(pred, gt, 0.5)["f1"] == pytest.approx(1.0)
    assert char_box_prf(pred[:5], gt, 0.5)["recall"] == pytest.approx(5 / 7)


def test_template_intervals_pixels(cfg):
    iv = template_intervals(cfg)
    assert len(iv) == 7
    assert iv[0] == pytest.approx((7.75, 30.25))
    assert all(b - a == pytest.approx(22.5) for a, b in iv)


def test_bad_method(cfg):
    with pytest.raises(ValueError):
        segment(_plate(cfg), "xyz", cfg)  # type: ignore[arg-type]


# ---------- #14: ngưỡng gộp mảnh hẹp riêng, cắt theo khe, lọc mảnh dẹt sau khi gộp ----------

def test_narrow_merge_independent_of_aspect_min(cfg):
    """Hạ char_aspect_range min (0.10, để giữ số "1") không được làm mất luật gộp mảnh chữ Hán:
    hai mảnh rộng 7 và 6 px (rộng/cao 0.14, 0.12) vẫn phải gộp ngay trong CCA."""
    low = {**cfg, "segment": {**cfg["segment"], "char_aspect_range": [0.10, 1.0]}}
    boxes = segment_cca(_plate(cfg, broken_lr=True), low)
    assert len(boxes) == 7
    assert position_hits(boxes, low)[0]


def test_split_count_uses_gap():
    """Khối 2 ký tự rộng 16 + khe 10 + 16 = 42 px: w / med = 2.6 → 3 phần (sai);
    (w + khe) / (med + khe) = 2 phần (đúng)."""
    boxes = [CharBox(10, 10, 26, 60), CharBox(36, 10, 52, 60), CharBox(62, 10, 104, 60),
             CharBox(114, 10, 130, 60), CharBox(140, 10, 156, 60)]
    out = merge_split_boxes(boxes, W, H)
    assert len(out) == 6
    assert [(b.x1, b.x2) for b in out[2:4]] == [(62, 83), (83, 104)]


def test_flat_stroke_joins_char_pieces(cfg):
    """Ký tự vỡ thành nửa trên lệch trái + nét ngang + nửa dưới lệch phải: hai nửa không chồng
    nhau theo x, chỉ nối qua nét ngang dẹt. Lọc mảnh dẹt trước khi gộp thì mất ký tự."""
    img = _plate(cfg)
    cx = _centers(cfg)[4]
    x1, x2 = cx - 8, cx + 8
    img[Y0:Y1, x1:x2] = 0
    img[Y0:32, x1:x1 + 6] = 255        # nửa trên, cao 22 < 0.4 × 70
    img[33:36, x1:x2] = 255            # nét ngang dẹt
    img[37:Y1, x1 + 10:x2] = 255       # nửa dưới, cao 23
    boxes = segment_cca(img, cfg)
    assert len(boxes) == 7
    assert position_hits(boxes, cfg)[4]
