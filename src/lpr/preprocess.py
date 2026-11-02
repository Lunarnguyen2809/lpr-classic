# OWNER: A
"""Tiền xử lý ảnh: resize, đổi hệ màu, CLAHE, khử nhiễu (SPEC mục 8.4)."""
from __future__ import annotations

import numpy as np

from lpr.types import DebugSteps, DenoiseMethod


def resize_keep_width(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Resize ảnh về chiều rộng `width`, giữ tỉ lệ.

    Input: ảnh BGR hoặc xám uint8, chiều rộng đích.
    Output: (ảnh đã resize, scale = rộng mới / rộng cũ).
    """
    raise NotImplementedError("TODO(A): resize_keep_width")


def to_gray(img: np.ndarray) -> np.ndarray:
    """Đổi ảnh BGR sang xám uint8 2 chiều."""
    raise NotImplementedError("TODO(A): to_gray")


def to_hsv(img: np.ndarray) -> np.ndarray:
    """Đổi ảnh BGR sang HSV (OpenCV, H trong [0, 180))."""
    raise NotImplementedError("TODO(A): to_hsv")


def to_lab(img: np.ndarray) -> np.ndarray:
    """Đổi ảnh BGR sang Lab (OpenCV, uint8)."""
    raise NotImplementedError("TODO(A): to_lab")


def get_channel(img: np.ndarray, name: str) -> np.ndarray:
    """Lấy một kênh từ ảnh BGR.

    Input: ảnh BGR, `name` ∈ {gray, H, S, V, L, a, b}.
    Output: ảnh 1 kênh uint8.
    """
    raise NotImplementedError("TODO(A): get_channel")


def apply_clahe(gray: np.ndarray, clip_limit: float, tile: tuple[int, int]) -> np.ndarray:
    """Cân bằng histogram thích nghi (CLAHE) cho ảnh xám.

    Input: ảnh xám uint8, clip_limit, kích thước lưới tile.
    Output: ảnh xám uint8 đã tăng tương phản.
    """
    raise NotImplementedError("TODO(A): apply_clahe")


def denoise(gray: np.ndarray, method: DenoiseMethod, params: dict) -> np.ndarray:
    """Khử nhiễu ảnh xám.

    Input: ảnh xám uint8, method ∈ {none, gaussian, median, bilateral},
    params = cfg["preprocess"][method] (ví dụ {"ksize": 5} hoặc {"d", "sigma_color", "sigma_space"}).
    Output: ảnh xám uint8 đã lọc (bản sao nếu method = none).
    """
    raise NotImplementedError("TODO(A): denoise")


def blackhat(gray: np.ndarray, kernel: tuple[int, int]) -> np.ndarray:
    """Phép black-hat hình thái học với kernel chữ nhật (rộng, cao).

    Input: ảnh xám uint8. Output: ảnh xám uint8 làm nổi chi tiết tối trên nền sáng.
    """
    raise NotImplementedError("TODO(A): blackhat")


def preprocess(img: np.ndarray, cfg: dict) -> tuple[np.ndarray, float, DebugSteps]:
    """Pipeline tiền xử lý đầy đủ theo cfg["preprocess"]: resize → gray → CLAHE → denoise.

    Input: ảnh BGR gốc, cfg toàn cục.
    Output: (ảnh xám đã xử lý, scale = mới / cũ, DebugSteps chứa ảnh từng bước).
    """
    raise NotImplementedError("TODO(A): preprocess")
