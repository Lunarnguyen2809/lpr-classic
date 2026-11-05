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
