# OWNER: A
"""Tiền xử lý ảnh: resize, đổi hệ màu, CLAHE, khử nhiễu (SPEC mục 8.4)."""
from __future__ import annotations

import cv2
import numpy as np

from lpr.types import DebugSteps, DenoiseMethod


def resize_keep_width(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Resize ảnh về chiều rộng `width`, giữ tỉ lệ.

    Input: ảnh BGR hoặc xám uint8, chiều rộng đích.
    Output: (ảnh đã resize, scale = rộng mới / rộng cũ).
    """
    old_h, old_w = img.shape[:2]
    scale = width / old_w
    new_h = max(1, round(old_h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    out = cv2.resize(img, (width, new_h), interpolation=interp)
    return out, scale


def to_gray(img: np.ndarray) -> np.ndarray:
    """Đổi ảnh BGR sang xám uint8 2 chiều."""
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_hsv(img: np.ndarray) -> np.ndarray:
    """Đổi ảnh BGR sang HSV (OpenCV, H trong [0, 180))."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def to_lab(img: np.ndarray) -> np.ndarray:
    """Đổi ảnh BGR sang Lab (OpenCV, uint8)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def get_channel(img: np.ndarray, name: str) -> np.ndarray:
    """Lấy một kênh từ ảnh BGR.

    Input: ảnh BGR, `name` ∈ {gray, H, S, V, L, a, b}.
    Output: ảnh 1 kênh uint8.
    """
    if name == "gray":
        return to_gray(img)
    if name in ("H", "S", "V"):
        return to_hsv(img)[:, :, "HSV".index(name)].copy()
    if name in ("L", "a", "b"):
        return to_lab(img)[:, :, "Lab".index(name)].copy()
    raise ValueError(f"Kênh không hợp lệ: {name}")


def apply_clahe(gray: np.ndarray, clip_limit: float, tile: tuple[int, int]) -> np.ndarray:
    """Cân bằng histogram thích nghi (CLAHE) cho ảnh xám.

    Input: ảnh xám uint8, clip_limit, kích thước lưới tile.
    Output: ảnh xám uint8 đã tăng tương phản.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile))
    return clahe.apply(gray)


def denoise(gray: np.ndarray, method: DenoiseMethod, params: dict) -> np.ndarray:
    """Khử nhiễu ảnh xám.

    Input: ảnh xám uint8, method ∈ {none, gaussian, median, bilateral},
    params = cfg["preprocess"][method] (ví dụ {"ksize": 5} hoặc {"d", "sigma_color", "sigma_space"}).
    Output: ảnh xám uint8 đã lọc (bản sao nếu method = none).
    """
    if method == "none":
        return gray.copy()
    if method == "gaussian":
        k = params["ksize"]
        return cv2.GaussianBlur(gray, (k, k), 0)
    if method == "median":
        return cv2.medianBlur(gray, params["ksize"])
    if method == "bilateral":
        return cv2.bilateralFilter(gray, params["d"], params["sigma_color"], params["sigma_space"])
    raise ValueError(f"Phương pháp khử nhiễu không hợp lệ: {method}")


def blackhat(gray: np.ndarray, kernel: tuple[int, int]) -> np.ndarray:
    """Phép black-hat hình thái học với kernel chữ nhật (rộng, cao).

    Input: ảnh xám uint8. Output: ảnh xám uint8 làm nổi chi tiết tối trên nền sáng.
    """
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel[0], kernel[1]))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)


def preprocess(img: np.ndarray, cfg: dict) -> tuple[np.ndarray, float, DebugSteps]:
    """Pipeline tiền xử lý đầy đủ theo cfg["preprocess"]: resize → gray → CLAHE → denoise.

    Input: ảnh BGR gốc, cfg toàn cục.
    Output: (ảnh xám đã xử lý, scale = mới / cũ, DebugSteps chứa ảnh từng bước).
    """
    pp = cfg["preprocess"]
    steps = DebugSteps()
    resized, scale = resize_keep_width(img, pp["resize_width"])
    steps.images["resized"] = resized
    gray = to_gray(resized)
    steps.images["gray"] = gray
    clahe_cfg = pp["clahe"]
    eq = apply_clahe(gray, clahe_cfg["clip_limit"], tuple(clahe_cfg["tile"]))
    steps.images["clahe"] = eq
    method = pp["denoise"]
    out = denoise(eq, method, pp.get(method, {}))
    steps.images["denoise"] = out
    return out, scale, steps
