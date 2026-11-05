"""Test tiền xử lý ảnh (SPEC mục 8.4), dùng ảnh tổng hợp."""
from __future__ import annotations

import numpy as np
import pytest

from lpr.config import load_config
from lpr.preprocess import (
    apply_clahe,
    blackhat,
    denoise,
    get_channel,
    preprocess,
    resize_keep_width,
    to_gray,
)


@pytest.fixture
def bgr() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(120, 200, 3), dtype=np.uint8)


def test_resize_shrink(bgr):
    out, scale = resize_keep_width(bgr, 100)
    assert out.shape == (60, 100, 3)
    assert out.dtype == np.uint8
    assert scale == pytest.approx(0.5)


def test_resize_enlarge(bgr):
    out, scale = resize_keep_width(bgr, 400)
    assert out.shape == (240, 400, 3)
    assert out.dtype == np.uint8
    assert scale == pytest.approx(2.0)


def test_resize_gray(bgr):
    out, _ = resize_keep_width(to_gray(bgr), 100)
    assert out.shape == (60, 100)


def test_to_gray_copy(bgr):
    g = to_gray(bgr)
    assert g.shape == (120, 200) and g.dtype == np.uint8
    g2 = to_gray(g)
    assert g2 is not g and np.array_equal(g, g2)


@pytest.mark.parametrize("name", ["gray", "H", "S", "V", "L", "a", "b"])
def test_get_channel(bgr, name):
    ch = get_channel(bgr, name)
    assert ch.shape == (120, 200)
    assert ch.dtype == np.uint8


def test_get_channel_unknown(bgr):
    with pytest.raises(ValueError):
        get_channel(bgr, "X")


def test_clahe(bgr):
    out = apply_clahe(to_gray(bgr), 2.0, (8, 8))
    assert out.shape == (120, 200) and out.dtype == np.uint8


_PARAMS = {
    "none": {},
    "gaussian": {"ksize": 5},
    "median": {"ksize": 5},
    "bilateral": {"d": 5, "sigma_color": 17, "sigma_space": 17},
}


@pytest.mark.parametrize("method", list(_PARAMS))
def test_denoise_methods(bgr, method):
    g = to_gray(bgr)
    out = denoise(g, method, _PARAMS[method])
    assert out.shape == g.shape and out.dtype == np.uint8


def test_denoise_none_is_copy(bgr):
    g = to_gray(bgr)
    out = denoise(g, "none", {})
    assert out is not g and np.array_equal(out, g)


def test_denoise_unknown(bgr):
    with pytest.raises(ValueError):
        denoise(to_gray(bgr), "bogus", {})


def test_blackhat(bgr):
    out = blackhat(to_gray(bgr), (17, 5))
    assert out.shape == (120, 200) and out.dtype == np.uint8


def test_preprocess_default_cfg(bgr):
    cfg = load_config()
    out, scale, steps = preprocess(bgr, cfg)
    assert out.shape == (480, 800) and out.dtype == np.uint8
    assert scale == pytest.approx(4.0)
    assert list(steps.images) == ["resized", "gray", "clahe", "denoise"]
    assert steps.images["resized"].shape == (480, 800, 3)
