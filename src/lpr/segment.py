# OWNER: B
"""Tách ký tự trên ảnh biển nhị phân 220×70 (SPEC mục 8.10)."""
from __future__ import annotations

import numpy as np

from lpr.types import CharBox, SegmentMethod


def split_lines(binary: np.ndarray) -> list[tuple[int, int]]:
    """Chiếu ngang để tìm các dòng chữ.

    Input: ảnh nhị phân (ký tự = 255). Output: list khoảng (y0, y1), y1 không bao gồm.
    """
    raise NotImplementedError("TODO(B): split_lines")


def segment_cca(binary: np.ndarray, cfg: dict) -> list[CharBox]:
    """Tách ký tự bằng connected components, lọc theo cfg["segment"] (chiều cao, tỉ lệ, diện tích).

    Output: list CharBox (chưa cần sắp xếp).
    """
    raise NotImplementedError("TODO(B): segment_cca")


def segment_projection(binary: np.ndarray, cfg: dict) -> list[CharBox]:
    """Tách ký tự bằng chiếu dọc, dùng cfg["segment"]["projection"] (min_gap, min_width).

    Output: list CharBox.
    """
    raise NotImplementedError("TODO(B): segment_projection")


def merge_split_boxes(boxes: list[CharBox], plate_w: int, plate_h: int) -> list[CharBox]:
    """Gộp mảnh nhỏ sát nhau (ký tự Hán vỡ nét), cắt đôi hộp rộng gấp ~2 lần trung vị.

    Output: list CharBox mới.
    """
    raise NotImplementedError("TODO(B): merge_split_boxes")


def segment(binary: np.ndarray, method: SegmentMethod, cfg: dict) -> list[CharBox]:
    """Tách ký tự theo method ∈ {cca, projection}, rồi merge_split_boxes.

    Output: list CharBox sắp từ trái sang phải.
    """
    raise NotImplementedError("TODO(B): segment")
