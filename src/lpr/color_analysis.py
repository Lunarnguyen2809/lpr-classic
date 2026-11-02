# OWNER: B
"""Phân tích khả năng tách biển/nền của từng kênh màu (SPEC mục 8.5, thí nghiệm E1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lpr.types import BBox, PlateRecord


def plate_mask(shape: tuple[int, int], bbox: BBox) -> np.ndarray:
    """Tạo mặt nạ biển số.

    Input: (cao, rộng) của ảnh, bbox (x1, y1, x2, y2).
    Output: ảnh uint8 cùng kích thước, 255 trong bbox, 0 ngoài bbox.
    """
    raise NotImplementedError("TODO(B): plate_mask")


def channel_histograms(img: np.ndarray, bbox: BBox, channels: list[str],
                       bins: int = 64) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Tính histogram trong và ngoài vùng biển cho từng kênh.

    Input: ảnh BGR, bbox biển, danh sách tên kênh (gray, H, S, V, L, a, b), số bin.
    Output: dict kênh -> (hist trong biển, hist ngoài biển), mỗi hist đã chuẩn hóa tổng = 1.
    """
    raise NotImplementedError("TODO(B): channel_histograms")


def separability(h_in: np.ndarray, h_out: np.ndarray) -> float:
    """Khoảng cách Bhattacharyya giữa hai histogram đã chuẩn hóa (càng lớn càng dễ tách)."""
    raise NotImplementedError("TODO(B): separability")


def channel_ranking(records: list[PlateRecord], channels: list[str]) -> pd.DataFrame:
    """Xếp hạng kênh màu theo độ tách trung bình trên tập ảnh.

    Input: danh sách PlateRecord, danh sách kênh.
    Output: DataFrame cột channel, mean_separability, std; sắp giảm dần theo mean_separability.
    """
    raise NotImplementedError("TODO(B): channel_ranking")
