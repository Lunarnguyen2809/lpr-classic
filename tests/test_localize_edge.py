"""Test định vị bằng cạnh (SPEC mục 8.6), dùng ảnh tổng hợp."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from lpr.config import load_config
from lpr.evaluate import iou
from lpr.localize.edge import (
    binarize_edges,
    candidate_features,
    extract_regions,
    localize_edge,
    sobel_x_map,
)
from lpr.types import Candidate, DebugSteps

BOX = (260, 200, 460, 262)  # 200×62, aspect ~3.2


def _plate_img(angle: float = 0.0) -> np.ndarray:
    img = np.full((400, 720, 3), 200, np.uint8)
    plate = np.full((BOX[3] - BOX[1], BOX[2] - BOX[0], 3), 255, np.uint8)
    for x in range(10, plate.shape[1] - 10, 22):  # vạch đen giả ký tự
        cv2.rectangle(plate, (x, 10), (x + 8, plate.shape[0] - 10), (0, 0, 0), -1)
    img[BOX[1]:BOX[3], BOX[0]:BOX[2]] = plate
    if angle:
        m = cv2.getRotationMatrix2D((360.0, 231.0), angle, 1.0)
        img = cv2.warpAffine(img, m, (720, 400), borderValue=(200, 200, 200))
    return img


@pytest.fixture
def cfg() -> dict:
    return load_config()


def test_finds_plate(cfg):
    cands = localize_edge(_plate_img(), cfg)
    assert cands and all(isinstance(c, Candidate) and c.method == "edge" for c in cands)
    assert max(iou(c.bbox, BOX) for c in cands) >= 0.5
    c = cands[0]
    assert c.quad.shape == (4, 2) and c.quad.dtype == np.float32
    assert c.quad[0, 0] < c.quad[1, 0] and c.quad[0, 1] < c.quad[3, 1]  # TL, TR, BL


def test_sorted_topk_and_bounds(cfg):
    cfg["localize"]["top_k"] = 1
    cands = localize_edge(_plate_img(), cfg)
    assert len(cands) <= 1
    cfg["localize"]["top_k"] = 5
    cands = localize_edge(_plate_img(), cfg)
    assert len(cands) <= 5
    assert [c.score for c in cands] == sorted((c.score for c in cands), reverse=True)
    for c in cands:
        assert 0 <= c.bbox[0] < c.bbox[2] <= 720 and 0 <= c.bbox[1] < c.bbox[3] <= 400


def test_other_width_maps_back(cfg):
    cfg["preprocess"]["resize_width"] = 640
    cands = localize_edge(_plate_img(), cfg)
    assert max(iou(c.bbox, BOX) for c in cands) >= 0.5


def test_features_angle_convention():
    mask = np.zeros((300, 400), np.uint8)
    for ang in (0.0, 15.0, -15.0):
        rect = ((200.0, 150.0), (120.0, 40.0), ang)
        m = np.zeros_like(mask)
        cv2.fillPoly(m, [np.round(cv2.boxPoints(rect)).astype(np.int32)], 255)
        f = candidate_features(extract_regions(m)[0], m.shape, m, m)
        assert f["angle"] == pytest.approx(ang, abs=1.5)
        assert f["aspect"] == pytest.approx(3.0, rel=0.05)
        assert f["fill_ratio"] > 0.95 and f["edge_density"] > 0.95
        assert f["area_ratio"] == pytest.approx(120 * 40 / (300 * 400), rel=0.05)
    # biển dựng đứng: góc vẫn trong (-90, 90], aspect vẫn >= 1
    m = np.zeros_like(mask)
    cv2.rectangle(m, (100, 50), (140, 170), 255, -1)
    f = candidate_features(extract_regions(m)[0], m.shape, m, m)
    assert abs(abs(f["angle"]) - 90) < 2 and f["aspect"] > 1


def test_sobel_otsu_dtype():
    g = np.zeros((50, 50), np.uint8)
    g[:, 25:] = 255
    e = sobel_x_map(g, 3)
    assert e.dtype == np.uint8 and e.max() == 255
    assert set(np.unique(binarize_edges(e))) <= {0, 255}


def test_empty_and_flat_images(cfg):
    assert localize_edge(np.full((300, 600, 3), 128, np.uint8), cfg) == []
    assert localize_edge(np.zeros((300, 600, 3), np.uint8), cfg) == []


def test_debug_steps(cfg):
    cands, steps = localize_edge(_plate_img(), cfg, return_debug=True)
    assert isinstance(steps, DebugSteps)
    for k in ("resized", "gray", "clahe", "denoise", "sobel", "otsu", "morph", "candidates"):
        assert k in steps.images
    assert steps.images["candidates"].shape[:2] == steps.images["resized"].shape[:2]
    assert isinstance(cands, list)


def _gap_plate_img(gap: int = 40) -> np.ndarray:
    """Biển có khoảng trống rộng giữa hai cụm ký tự: kernel close mặc định không nối được."""
    img = np.full((400, 720, 3), 200, np.uint8)
    plate = np.full((62, 260, 3), 255, np.uint8)
    for x in list(range(10, 100, 22)) + list(range(100 + gap, 250, 22)):
        cv2.rectangle(plate, (x, 10), (x + 8, 52), (0, 0, 0), -1)
    img[200:262, 230:490] = plate
    return img


def test_multiscale_close_recovers_fragmented_plate(cfg):
    gt = (230, 200, 490, 262)
    img = _gap_plate_img()
    cfg["localize"]["extra_close_kernels"] = []
    single = localize_edge(img, cfg)
    cfg["localize"]["extra_close_kernels"] = [[13, 3], [61, 9]]
    multi = localize_edge(img, cfg)
    best = lambda cs: max((iou(c.bbox, gt) for c in cs), default=0.0)  # noqa: E731
    assert best(multi) >= 0.5
    assert best(multi) >= best(single)


def test_multiscale_nms_no_duplicates(cfg):
    cfg["localize"]["extra_close_kernels"] = [[13, 3], [33, 7], [45, 9]]
    cands = localize_edge(_plate_img(), cfg)
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            assert iou(a.bbox, b.bbox) <= cfg["localize"]["nms_iou"] + 1e-6
    assert [c.score for c in cands] == sorted((c.score for c in cands), reverse=True)
    assert len(cands) <= cfg["localize"]["top_k"]


def test_extra_kernels_optional(cfg):
    del cfg["localize"]["extra_close_kernels"]
    assert max(iou(c.bbox, BOX) for c in localize_edge(_plate_img(), cfg)) >= 0.5
