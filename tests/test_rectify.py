"""Test nắn biển (SPEC mục 12), hiện chỉ order_quad."""
from __future__ import annotations

import numpy as np
import pytest

from lpr.rectify import order_quad

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
