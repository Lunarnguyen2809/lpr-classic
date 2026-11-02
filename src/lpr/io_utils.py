# OWNER: SHARED
"""Tiện ích đọc/ghi ảnh, CSV và đo thời gian (xem SPEC mục 8.1)."""
from __future__ import annotations

import csv
import os
import time

import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """Đọc ảnh màu BGR uint8, hỗ trợ đường dẫn Unicode (np.fromfile + cv2.imdecode).

    Raise FileNotFoundError nếu không có file, ValueError nếu không giải mã được.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không giải mã được ảnh: {path}")
    return img


def write_image(path: str, img: np.ndarray) -> None:
    """Ghi ảnh ra `path` (định dạng theo đuôi file), tự tạo thư mục cha."""
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Không mã hóa được ảnh: {path}")
    buf.tofile(path)


def ensure_dir(path: str) -> str:
    """Tạo thư mục nếu chưa có, trả về chính đường dẫn đó."""
    os.makedirs(path, exist_ok=True)
    return path


def write_csv(rows: list[dict], path: str) -> None:
    """Ghi danh sách dict ra CSV UTF-8. Cột lấy theo thứ tự xuất hiện của khóa."""
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class Timer:
    """Context manager đo thời gian; sau khi thoát, `.ms` là số mili giây."""

    def __init__(self) -> None:
        self.ms: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0


def timer() -> Timer:
    """Tạo Timer mới: `with timer() as t: ...; print(t.ms)`."""
    return Timer()
