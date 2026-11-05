"""Test nắn biển (SPEC mục 12)."""
from __future__ import annotations

import numpy as np
import pytest

import cv2

from lpr.rectify import (corner_error, crop_bbox, order_quad, quad_from_candidate, rectify,
                         warp_plate)
from lpr.types import Candidate

EXPECTED = np.array([[154, 383], [363, 402], [386, 473], [177, 454]], np.float32)


@pytest.mark.parametrize("perm", [(0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0), (1, 3, 0, 2)])
def test_order_quad_shuffled(perm):
    out = order_quad(EXPECTED[list(perm)])
    assert out.shape == (4, 2) and out.dtype == np.float32
    np.testing.assert_allclose(out, EXPECTED)


def test_order_quad_rect_and_rotated():
    rect = np.array([[0, 0], [100, 0], [100, 30], [0, 30]], np.float32)
    np.testing.assert_allclose(order_quad(rect[[2, 0, 3, 1]]), rect)
    th = np.deg2rad(20)
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rot = ((rect - [50, 15]) @ r.T + [200, 100]).astype(np.float32)  # TL, TR, BR, BL sau xoay
    np.testing.assert_allclose(order_quad(rot[[3, 1, 0, 2]]), rot, atol=1e-4)


CFG = {"rectify": {"plate_size": [220, 70]}}
SIZE = (220, 70)
GT_QUAD = np.array([[100, 80], [300, 60], [320, 160], [90, 150]], np.float32)  # không phải HCN


def _blue_plate_image() -> np.ndarray:
    img = np.full((260, 420, 3), (90, 140, 90), np.uint8)
    cv2.fillConvexPoly(img, GT_QUAD.astype(np.int32), (170, 60, 20))  # xanh dương (BGR)
    for i in range(6):  # "ký tự" trắng
        cx = 130 + i * 28
        cv2.rectangle(img, (cx, 100), (cx + 14, 135), (255, 255, 255), -1)
    return img


def _candidate() -> Candidate:
    x1, y1 = GT_QUAD.min(axis=0).astype(int)
    x2, y2 = GT_QUAD.max(axis=0).astype(int) + 1
    box = cv2.boxPoints(cv2.minAreaRect(GT_QUAD))
    return Candidate((int(x1), int(y1), int(x2), int(y2)), order_quad(box), 1.0, "edge")


def test_warp_plate_recovers_pattern():
    # ảnh nền: HCN xoay 20 độ, nửa trái đỏ, nửa phải xanh lá
    w, h = 220, 70
    pat = np.zeros((h, w, 3), np.uint8)
    pat[:, : w // 2] = (0, 0, 255)
    pat[:, w // 2:] = (0, 255, 0)
    th = np.deg2rad(20)
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    quad = ((dst - [w / 2, h / 2]) @ r.T + [300, 200]).astype(np.float32)
    m = cv2.getPerspectiveTransform(dst, quad)
    big = cv2.warpPerspective(pat, m, (600, 400), borderMode=cv2.BORDER_REPLICATE)
    out = warp_plate(big, quad, SIZE)
    assert out.shape == (70, 220, 3) and out.dtype == np.uint8
    assert tuple(out[35, 20]) == (0, 0, 255) and tuple(out[35, 200]) == (0, 255, 0)
    assert np.abs(out.astype(int) - pat.astype(int)).mean() < 12
    # thứ tự điểm đầu vào không ảnh hưởng
    np.testing.assert_array_equal(warp_plate(big, quad[[2, 0, 3, 1]], SIZE), out)


def test_crop_bbox_size_and_clipping():
    img = np.full((100, 200, 3), 7, np.uint8)
    assert crop_bbox(img, (10, 10, 60, 40), SIZE).shape == (70, 220, 3)
    out = crop_bbox(img, (-30, -20, 300, 150), SIZE)  # vượt biên -> cắt về ảnh
    assert out.shape == (70, 220, 3) and (out == 7).all()
    with pytest.raises(ValueError):
        crop_bbox(img, (250, 10, 300, 40), SIZE)


@pytest.mark.parametrize("method", ["bbox", "quad"])
def test_rectify_shape(method):
    img = _blue_plate_image()
    c = _candidate()
    out = rectify(img, c.bbox, GT_QUAD, method, CFG)
    assert out.shape == (70, 220, 3) and out.dtype == np.uint8
    gray = rectify(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), c.bbox, GT_QUAD, method, CFG)
    assert gray.shape == (70, 220, 3)


def test_rectify_quad_none_falls_back_to_bbox():
    img = _blue_plate_image()
    c = _candidate()
    np.testing.assert_array_equal(rectify(img, c.bbox, None, "quad", CFG),
                                  rectify(img, c.bbox, None, "bbox", CFG))
    with pytest.raises(ValueError):
        rectify(img, c.bbox, None, "nope", CFG)  # type: ignore[arg-type]


def test_rectify_quad_upright_blue_filled():
    out = rectify(_blue_plate_image(), _candidate().bbox, GT_QUAD, "quad", CFG)
    blue = (out[..., 0] > 120) & (out[..., 2] < 80)
    assert blue.mean() > 0.6  # biển lấp đầy ảnh nắn (còn lại là ký tự trắng)


def test_corner_error():
    q = GT_QUAD
    assert corner_error(q, q) == 0.0
    shifted = q + np.array([3, 4], np.float32)
    assert corner_error(shifted, q) == pytest.approx(5.0, abs=1e-4)
    assert corner_error(q[[2, 0, 3, 1]], shifted) == pytest.approx(5.0, abs=1e-4)


def test_quad_from_candidate_methods():
    img = _blue_plate_image()
    c = _candidate()
    qm = quad_from_candidate(img, c, "minarearect")
    qa = quad_from_candidate(img, c, "approxpoly")
    for q in (qm, qa):
        assert q.shape == (4, 2) and q.dtype == np.float32
    np.testing.assert_allclose(qm, order_quad(c.quad))
    assert corner_error(qa, GT_QUAD) < 3.0
    assert corner_error(qa, GT_QUAD) < corner_error(qm, GT_QUAD)
    with pytest.raises(ValueError):
        quad_from_candidate(img, c, "bad")


def test_approxpoly_falls_back_without_blue():
    img = np.full((260, 420, 3), 128, np.uint8)
    c = _candidate()
    np.testing.assert_allclose(quad_from_candidate(img, c, "approxpoly"), order_quad(c.quad))
