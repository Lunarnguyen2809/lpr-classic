# OWNER: SHARED
"""Kiểu dữ liệu chung cho toàn bộ pipeline (xem SPEC mục 6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

BBox = tuple[int, int, int, int]          # (x1, y1, x2, y2), pixel, x2/y2 không bao gồm
Quad = np.ndarray                          # shape (4, 2), float32, thứ tự TL, TR, BR, BL
Split = Literal["tune", "test", "qual"]
DenoiseMethod = Literal["none", "gaussian", "median", "bilateral"]
BinarizeMethod = Literal["otsu", "adaptive"]
SegmentMethod = Literal["cca", "projection"]
CropMethod = Literal["bbox", "quad"]


@dataclass
class PlateRecord:
    image_id: str
    path: str
    source: Literal["ccpd", "vn"]
    subset: str
    split: Split
    bbox: BBox
    quad: Quad | None = None               # None với ảnh VN
    text: str | None = None
    n_lines: int = 1
    meta: dict = field(default_factory=dict)  # area, tilt_h, tilt_v, brightness, blur


@dataclass
class Candidate:
    bbox: BBox                             # trong tọa độ ảnh GỐC (đã nhân lại scale)
    quad: Quad                             # từ minAreaRect, TL, TR, BR, BL
    score: float                           # heuristic hoặc SVM, càng cao càng giống biển
    method: str                            # "edge" | "edge_svm"
    features: dict = field(default_factory=dict)  # aspect, area_ratio, fill_ratio, edge_density, angle


@dataclass
class CharBox:
    x1: int; y1: int; x2: int; y2: int     # trong tọa độ ảnh biển 220×70
    line: int = 0                          # 0 = dòng trên / dòng duy nhất


@dataclass
class DebugSteps:
    images: dict[str, np.ndarray] = field(default_factory=dict)  # tên bước -> ảnh, giữ thứ tự chèn
