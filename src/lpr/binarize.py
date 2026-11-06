# OWNER: B
"""Nhị phân ảnh biển đã nắn (SPEC mục 8.9)."""
from __future__ import annotations

import cv2
import numpy as np

from lpr.types import BinarizeMethod


def binarize_plate(plate_gray: np.ndarray, method: BinarizeMethod, params: dict) -> np.ndarray:
    """Nhị phân ảnh biển xám.

    Input: ảnh xám uint8, method ∈ {otsu, adaptive}, params (ví dụ {"block_size": 25, "C": 7}).
    Output: ảnh nhị phân uint8 0/255.
    """
    if plate_gray.ndim != 2:
        raise ValueError(f"cần ảnh xám 2 chiều, nhận shape {plate_gray.shape}")
    gray = plate_gray if plate_gray.dtype == np.uint8 else plate_gray.astype(np.uint8)

    if method == "otsu":
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return b
    if method == "adaptive":
        bs = int(params.get("block_size", 25))
        if bs % 2 == 0:                    # OpenCV bắt buộc block_size lẻ
            bs += 1
        bs = max(bs, 3)
        c = float(params.get("C", 7))
        # adaptiveThreshold chỉ "bắt" nét tối hơn nền xung quanh ít nhất C (THRESH_BINARY_INV),
        # nên trước hết đưa ảnh về dạng chữ tối: dùng Otsu đoán cực tính, chữ sáng (biển xanh)
        # thì đảo xám. Nếu dùng THRESH_BINARY với C > 0 trên chữ sáng, nền phẳng thành trắng
        # và ký tự chỉ còn viền rỗng.
        _, o = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if np.count_nonzero(o) / o.size <= 0.5:   # phần sáng ít hơn -> chữ sáng
            gray = cv2.bitwise_not(gray)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, bs, c)
    raise ValueError(f"method không hợp lệ: {method!r} (chỉ nhận 'otsu' | 'adaptive')")


def ensure_text_white(binary: np.ndarray) -> np.ndarray:
    """Bảo đảm ký tự = 255: đảo ảnh nếu tỉ lệ pixel trắng > 0.5.

    Biển xanh CCPD có chữ trắng, biển vàng có chữ đen. Output: ảnh nhị phân mới.
    """
    if np.count_nonzero(binary) / binary.size > 0.5:
        return cv2.bitwise_not(binary)
    return binary.copy()


def clean_binary(binary: np.ndarray, open_k: tuple[int, int] = (2, 2)) -> np.ndarray:
    """Mở hình thái học để bỏ nhiễu nhỏ. Output: ảnh nhị phân uint8 0/255.

    open_k dạng (rộng, cao), giống quy ước trong localize/edge.py.
    """
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (int(open_k[0]), int(open_k[1])))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
