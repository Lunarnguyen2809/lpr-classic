import pytest

from lpr.config import load_config
from lpr.evaluate import (
    char_box_prf,
    count_match,
    iou,
    iou_1d,
    match_detections,
    position_hits,
    prf,
    template_intervals,
)
from lpr.types import CharBox


@pytest.fixture(scope="module")
def cfg():
    return load_config("config/default.yaml")


def test_iou_identical_disjoint_partial():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0  # chạm cạnh
    assert iou((0, 0, 10, 10), (5, 5, 5, 15)) == 0.0  # suy biến
    # giao 5x10=50, hợp 100+100-50=150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_prf_hand_numbers_and_zero():
    r = prf(8, 2, 4)
    assert r["precision"] == pytest.approx(0.8)
    assert r["recall"] == pytest.approx(8 / 12)
    assert r["f1"] == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert prf(0, 3, 2)["f1"] == 0.0


def test_match_detections_greedy():
    gts = [(0, 0, 10, 10), (6, 0, 16, 10)]
    preds = [(5, 0, 15, 10)]   # IoU: gt0 = 1/3, gt1 = 9/11
    assert match_detections(preds, gts, 0.3) == (1, 0, 1)
    preds2 = [(5, 0, 15, 10), (0, 0, 10, 10)]
    # pred0 khớp gt1 (IoU cao hơn), pred1 khớp gt0
    assert match_detections(preds2, gts, 0.5) == (2, 0, 0)
    assert match_detections(preds, gts, 0.95) == (0, 1, 2)
    assert match_detections([], gts, 0.5) == (0, 0, 2)
    assert match_detections([(0, 0, 10, 10)], [(50, 50, 60, 60)], 0.0) == (0, 1, 1)
    # thứ tự tham lam: A-Y 0.9 chọn trước, B-Y 0.7 và A-X 0.6 bị loại, B-X 7/15 < 0.5
    # (tối ưu toàn cục là 2 cặp A-X + B-Y, nhưng tham lam chỉ được 1)
    gt_x, gt_y = (0, 0, 15, 10), (0, 0, 10, 10)
    pred_a, pred_b = (0, 0, 9, 10), (0, 0, 7, 10)
    assert match_detections([pred_a, pred_b], [gt_x, gt_y], 0.5) == (1, 1, 1)


def test_iou_1d():
    assert iou_1d((0, 10), (0, 10)) == 1.0
    assert iou_1d((0, 10), (10, 20)) == 0.0
    assert iou_1d((0, 10), (5, 15)) == pytest.approx(1 / 3)


def test_template_intervals_default(cfg):
    iv = template_intervals(cfg)
    assert len(iv) == 7
    assert iv[0] == pytest.approx((7.75, 30.25))
    assert iv[6] == pytest.approx((189.75, 212.25))


def _boxes(cfg):
    return [CharBox(round(lo), 5, round(hi), 60) for lo, hi in template_intervals(cfg)]


def test_position_hits(cfg):
    boxes = _boxes(cfg)
    assert position_hits(boxes, cfg) == [True] * 7
    assert position_hits([], cfg) == [False] * 7
    assert position_hits(boxes[1:], cfg) == [False] + [True] * 6


def test_position_hits_default_threshold_without_config(cfg):
    c = {k: v for k, v in cfg.items() if k != "evaluate"}
    assert position_hits(_boxes(cfg), c) == [True] * 7


def test_count_match():
    boxes = [CharBox(0, 0, 1, 1)] * 7
    assert count_match(boxes, "皖AY339S")
    assert not count_match(boxes[:6], "皖AY339S")


def test_char_box_prf():
    gt = [CharBox(0, 0, 10, 10), CharBox(20, 0, 30, 10), CharBox(40, 0, 50, 10)]
    pred = [CharBox(0, 0, 10, 10), CharBox(20, 0, 30, 10), CharBox(100, 0, 110, 10)]
    r = char_box_prf(pred, gt, 0.5)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)


def _slot(cfg, i=2):
    lo, hi = template_intervals(cfg)[i]
    return lo, hi, (lo + hi) / 2


def test_position_hits_narrow_centered_box_hits(cfg):
    """Số "1" rộng 9 px đặt giữa khuôn: IoU ≈ 0.4 < 0.5 nhưng vẫn trúng nhờ luật tâm."""
    lo, hi, c = _slot(cfg)
    box = CharBox(round(c - 4.5), 5, round(c + 4.5), 60)
    assert iou_1d((box.x1, box.x2), (lo, hi)) < 0.5
    assert position_hits([box], cfg) == [False, False, True, False, False, False, False]


def test_position_hits_wide_merged_blob_centered_misses(cfg):
    """Khối dính rộng 2.5× khuôn, tâm trùng tâm khuôn: IoU < 0.5 và vượt ngưỡng hộp hẹp, không trúng."""
    lo, hi, c = _slot(cfg)
    w = (hi - lo) * 2.5
    box = CharBox(round(c - w / 2), 5, round(c + w / 2), 60)
    assert iou_1d((box.x1, box.x2), (lo, hi)) < 0.5
    assert not position_hits([box], cfg)[2]


def test_position_hits_normal_width_off_center_misses(cfg):
    """Hộp ~22 px (bình thường) lệch tâm ~10 px: IoU < 0.5 và không phải hộp hẹp, nên trượt."""
    lo, hi, c = _slot(cfg)
    box = CharBox(round(lo + 10), 5, round(lo + 10 + 22), 60)
    assert iou_1d((box.x1, box.x2), (lo, hi)) < 0.5
    assert lo <= (box.x1 + box.x2) / 2 < hi
    assert not position_hits([box], cfg)[2]


def test_position_hits_zero_width_box_misses(cfg):
    """Hộp rộng 0 đặt giữa khuôn không trúng."""
    _, _, c = _slot(cfg)
    box = CharBox(round(c), 5, round(c), 60)
    assert not position_hits([box], cfg)[2]


def test_position_hits_narrow_off_slot_misses(cfg):
    """Hộp hẹp nằm ngoài khuôn (tâm lệch ra ngoài [x1, x2)) không trúng."""
    lo, hi, _ = _slot(cfg)
    box = CharBox(round(hi) + 1, 5, round(hi) + 10, 60)
    assert not position_hits([box], cfg)[2]


def test_position_hits_iou_still_hits(cfg):
    """Hộp rộng 1.6× khuôn nhưng IoU ≥ 0.5 vẫn trúng theo luật IoU (khuôn nằm trọn trong hộp: 1/1.6 = 0.625)."""
    lo, hi, c = _slot(cfg)
    w = (hi - lo) * 1.6
    box = CharBox(round(c - w / 2), 5, round(c + w / 2), 60)
    cfg2 = {**cfg, "evaluate": {**cfg["evaluate"], "char_iou_threshold": 0.5}}
    assert iou_1d((box.x1, box.x2), (lo, hi)) >= 0.5
    assert position_hits([box], cfg2)[2]
