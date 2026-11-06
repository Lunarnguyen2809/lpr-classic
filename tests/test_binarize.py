"""Test nhị phân ảnh biển (SPEC mục 8.9), dùng ảnh tổng hợp."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from lpr.binarize import binarize_plate, clean_binary, ensure_text_white
from lpr.config import load_config

H, W = 70, 220  # kích thước biển sau nắn


def _plate(dark_text: bool = True) -> np.ndarray:
    """Biển xám giả: nền sáng + 7 vạch tối (dark_text) hoặc nền tối + vạch sáng."""
    bg, fg = (200, 40) if dark_text else (40, 200)
    img = np.full((H, W), bg, np.uint8)
    for i in range(7):
        x = 15 + i * 29
        cv2.rectangle(img, (x, 12), (x + 12, H - 12), fg, -1)
    return img


def _shading(img: np.ndarray) -> np.ndarray:
    """Thêm gradient sáng trái → phải để mô phỏng chiếu sáng không đều."""
    ramp = np.linspace(-60, 60, W, dtype=np.float32)[None, :]
    return np.clip(img.astype(np.float32) + ramp, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("method", ["otsu", "adaptive"])
def test_binarize_output_format(method):
    out = binarize_plate(_plate(), method, {"block_size": 25, "C": 7})
    assert out.shape == (H, W)
    assert out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}


@pytest.mark.parametrize("method", ["otsu", "adaptive"])
def test_binarize_separates_text(method):
    out = ensure_text_white(binarize_plate(_plate(), method, {"block_size": 25, "C": 7}))
    assert out[H // 2, 15 + 6] == 255     # giữa vạch đầu tiên là ký tự
    assert out[H // 2, 15 + 20] == 0      # khe giữa hai vạch là nền


def test_adaptive_handles_uneven_light():
    out = ensure_text_white(binarize_plate(_shading(_plate()), "adaptive", {"block_size": 25, "C": 7}))
    for i in (0, 6):                       # vạch ở đầu tối và đầu sáng đều phải là chữ
        assert out[H // 2, 15 + i * 29 + 6] == 255


@pytest.mark.parametrize("dark_text", [True, False])
def test_adaptive_flat_background_stays_black(dark_text):
    """Nền phẳng không được thành trắng, ký tự phải đặc chứ không chỉ còn viền."""
    out = binarize_plate(_plate(dark_text), "adaptive", {"block_size": 25, "C": 7})
    assert out[3, 3] == 0                  # góc nền phẳng
    assert out[H // 2, 15 + 6] == 255      # tâm vạch ký tự
    assert np.count_nonzero(out) / out.size < 0.5


def test_adaptive_even_block_size_ok():
    out = binarize_plate(_plate(), "adaptive", {"block_size": 24, "C": 7})
    assert set(np.unique(out)) <= {0, 255}


def test_binarize_with_config_params():
    cfg = load_config("config/default.yaml")["binarize"]
    out = binarize_plate(_plate(), cfg["method"], cfg["adaptive"])
    assert out.shape == (H, W)


def test_binarize_bad_method():
    with pytest.raises(ValueError):
        binarize_plate(_plate(), "xyz", {})  # type: ignore[arg-type]


def test_binarize_rejects_color():
    with pytest.raises(ValueError):
        binarize_plate(np.zeros((H, W, 3), np.uint8), "otsu", {})


@pytest.mark.parametrize("dark_text", [True, False])
def test_ensure_text_white_both_polarities(dark_text):
    b = binarize_plate(_plate(dark_text), "otsu", {})
    out = ensure_text_white(b)
    assert np.count_nonzero(out) / out.size <= 0.5
    assert out[H // 2, 15 + 6] == 255


def test_ensure_text_white_idempotent_and_copy():
    b = ensure_text_white(binarize_plate(_plate(), "otsu", {}))
    again = ensure_text_white(b)
    assert np.array_equal(b, again)
    assert again is not b


def test_clean_binary_removes_speck_keeps_strokes():
    b = ensure_text_white(binarize_plate(_plate(), "otsu", {}))
    b[3, 3] = 255                          # nhiễu 1 pixel ở nền
    out = clean_binary(b, (2, 2))
    assert out[3, 3] == 0
    assert out[H // 2, 15 + 6] == 255
    assert set(np.unique(out)) <= {0, 255}
