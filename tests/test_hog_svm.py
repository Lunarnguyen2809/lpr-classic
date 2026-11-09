"""Test localize/hog_svm (SPEC mục 8.7), dùng ảnh tổng hợp, không cần dataset."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from lpr.config import load_config
from lpr.io_utils import write_image
from lpr.localize import hog_svm
from lpr.localize.hog_svm import (
    build_patch_dataset,
    extract_hog,
    load_model,
    localize_edge_svm,
    save_model,
    score_candidates,
    train_svm,
)
from lpr.types import Candidate, PlateRecord

BOX = (260, 200, 460, 262)


def _plate_img() -> np.ndarray:
    img = np.full((400, 720, 3), 200, np.uint8)
    plate = np.full((BOX[3] - BOX[1], BOX[2] - BOX[0], 3), 255, np.uint8)
    for x in range(10, plate.shape[1] - 10, 22):
        cv2.rectangle(plate, (x, 10), (x + 8, plate.shape[0] - 10), (0, 0, 0), -1)
    img[BOX[1]:BOX[3], BOX[0]:BOX[2]] = plate
    return img


def _cand(bbox, score=0.0) -> Candidate:
    x1, y1, x2, y2 = bbox
    quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return Candidate(bbox=bbox, quad=quad, score=score, method="edge", features={"aspect": 3.0})


@pytest.fixture
def cfg() -> dict:
    return load_config()


@pytest.fixture
def toy(cfg):
    """Dữ liệu HOG: dương = sọc dọc (giống chữ), âm = nhiễu/phẳng."""
    rng = np.random.default_rng(0)
    pos, neg = [], []
    for _ in range(30):
        p = np.full((40, 120), 255, np.uint8)
        for x in range(6, 114, 14):
            cv2.rectangle(p, (x, 6), (x + 6, 34), 0, -1)
        pos.append(extract_hog(p, cfg))
        neg.append(extract_hog(rng.integers(0, 256, (40, 120)).astype(np.uint8), cfg))
    X = np.stack(pos + neg)
    y = np.array([1] * 30 + [0] * 30)
    return X, y


def test_extract_hog_shape_and_resize(cfg):
    a = extract_hog(np.zeros((50, 200), np.uint8), cfg)
    b = extract_hog(np.full((13, 31), 90, np.uint8), cfg)
    w, h = cfg["hog_svm"]["patch_size"]
    ppc, cpb = cfg["hog_svm"]["pixels_per_cell"], cfg["hog_svm"]["cells_per_block"]
    n = ((w // ppc[0] - cpb[0] + 1) * (h // ppc[1] - cpb[1] + 1)
         * cpb[0] * cpb[1] * cfg["hog_svm"]["orientations"])
    assert a.ndim == 1 and a.shape == b.shape == (n,)
    assert np.issubdtype(a.dtype, np.floating)
    assert np.array_equal(extract_hog(np.zeros((50, 200, 3), np.uint8), cfg), a)  # BGR -> xám


def test_train_svm_cv_and_fit(cfg, toy):
    X, y = toy
    model, cv = train_svm(X, y, cfg)
    assert cv["n_splits"] == 5 and len(cv["folds"]) == 5
    assert cv["n_pos"] == 30 and cv["n_neg"] == 30
    for m in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= cv[m] <= 1.0
    assert cv["accuracy"] > 0.9
    assert (model.predict(X) == y).mean() > 0.95
    _, cv2_ = train_svm(X, y, cfg)  # seed cố định -> lặp lại được
    assert cv2_["folds"] == cv["folds"]


def test_save_load_roundtrip(cfg, toy, tmp_path):
    model, _ = train_svm(*toy, cfg)
    path = str(tmp_path / "sub" / "dir" / "svm.joblib")
    save_model(model, path)  # tự tạo thư mục cha
    m2 = load_model(path)
    np.testing.assert_allclose(m2.decision_function(toy[0]), model.decision_function(toy[0]))


def test_score_candidates_order_and_fields(cfg, toy):
    model, _ = train_svm(*toy, cfg)
    img = _plate_img()
    cands = [_cand((20, 20, 220, 82), 0.9), _cand(BOX, 0.1), _cand((500, 300, 700, 360), 0.5)]
    out = score_candidates(img, cands, model, cfg)
    assert len(out) == 3
    assert all(c.method == "edge_svm" for c in out)
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)
    assert out[0].bbox == BOX  # vùng biển thật phải đứng đầu
    assert out[0].features == {"aspect": 3.0} and out[0].quad.shape == (4, 2)
    assert cands[1].method == "edge" and cands[1].score == 0.1  # input không bị sửa
    assert score_candidates(img, [], model, cfg) == []


def test_localize_edge_svm_sorted_with_spec_fields(cfg, toy):
    model, _ = train_svm(*toy, cfg)
    cands = localize_edge_svm(_plate_img(), model, cfg)
    assert cands
    assert all(isinstance(c, Candidate) and c.method == "edge_svm" for c in cands)
    assert all(c.quad.shape == (4, 2) and len(c.bbox) == 4 for c in cands)
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)


def _records(tmp_path, split="tune", n=2) -> list[PlateRecord]:
    recs = []
    for i in range(n):
        p = str(tmp_path / f"im{i}.png")
        write_image(p, _plate_img())
        recs.append(PlateRecord(image_id=f"im{i}", path=p, source="ccpd", subset="base",
                                split=split, bbox=BOX))
    return recs


def test_build_patch_dataset_labels(cfg, tmp_path):
    recs = _records(tmp_path)
    X, y = build_patch_dataset(recs, cfg)
    neg = cfg["hog_svm"]["neg_per_pos"]
    assert X.shape[0] == y.shape[0] == len(recs) * (1 + neg)
    assert set(np.unique(y)) == {0, 1}
    assert int((y == 1).sum()) == len(recs) and int((y == 0).sum()) == len(recs) * neg
    X2, y2 = build_patch_dataset(recs, cfg)  # tái lập được
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)


def test_build_patch_dataset_rejects_non_tune(cfg, tmp_path):
    with pytest.raises(ValueError):
        build_patch_dataset(_records(tmp_path, split="test"), cfg)


def test_random_boxes_iou_below_threshold():
    rng = np.random.default_rng(1)
    boxes = hog_svm._random_boxes(BOX, (400, 720), 10, rng)
    assert boxes
    for b in boxes:
        assert hog_svm.iou(b, BOX) < hog_svm.NEG_IOU_MAX
        assert 0 <= b[0] < b[2] <= 720 and 0 <= b[1] < b[3] <= 400


def test_random_boxes_respects_aspect_and_is_deterministic():
    lo, hi = 3.0, 4.0
    boxes = hog_svm._random_boxes(BOX, (400, 720), 20, np.random.default_rng(7), aspect=(lo, hi))
    assert boxes
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        # bh = int(bw / ratio) cắt phần lẻ: w/h >= lo, và w/(h+1) < hi
        assert w / h >= lo
        assert w / (h + 1) < hi
    again = hog_svm._random_boxes(BOX, (400, 720), 20, np.random.default_rng(7), aspect=(lo, hi))
    assert boxes == again
