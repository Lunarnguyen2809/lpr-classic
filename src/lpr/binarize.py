# OWNER: B
"""Nhị phân ảnh biển đã nắn (SPEC mục 8.9)."""
from __future__ import annotations

import numpy as np

from lpr.types import BinarizeMethod


def binarize_plate(plate_gray: np.ndarray, method: BinarizeMethod, params: dict) -> np.ndarray:
    """Nhị phân ảnh biển xám.

    Input: ảnh xám uint8, method ∈ {otsu, adaptive}, params (ví dụ {"block_size": 25, "C": 7}).
    Output: ảnh nhị phân uint8 0/255.
    """
    raise NotImplementedError("TODO(B): binarize_plate")


def ensure_text_white(binary: np.ndarray) -> np.ndarray:
    """Bảo đảm ký tự = 255: đảo ảnh nếu tỉ lệ pixel trắng > 0.5.

    Biển xanh CCPD có chữ trắng, biển vàng có chữ đen. Output: ảnh nhị phân mới.
    """
    raise NotImplementedError("TODO(B): ensure_text_white")


def clean_binary(binary: np.ndarray, open_k: tuple[int, int] = (2, 2)) -> np.ndarray:
    """Mở hình thái học để bỏ nhiễu nhỏ. Output: ảnh nhị phân uint8 0/255."""
    raise NotImplementedError("TODO(B): clean_binary")
