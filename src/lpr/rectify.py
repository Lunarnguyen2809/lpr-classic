# OWNER: A
"""Nắn biển số về hình chữ nhật chuẩn 220×70 (SPEC mục 8.8)."""
from __future__ import annotations

import numpy as np

from lpr.types import BBox, Candidate, CropMethod, Quad


def order_quad(pts: np.ndarray) -> Quad:
    """Sắp 4 điểm bất kỳ về thứ tự TL, TR, BR, BL.

    Input: mảng (4, 2). Output: Quad float32 (4, 2).
    """
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    # xếp theo góc quanh tâm (chiều kim đồng hồ trong hệ y xuống), rồi chọn đỉnh đầu sao cho
    # cạnh đầu hướng +x gần ngang nhất (đúng cho mọi tỉ lệ khung, góc xoay < 45°)
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    e = np.roll(p, -1, axis=0) - p
    start = int(np.argmin(np.abs(np.arctan2(e[:, 1], e[:, 0]))))
    return np.roll(p, -start, axis=0).astype(np.float32)


def quad_from_candidate(img: np.ndarray, cand: Candidate, method: str) -> Quad:
    """Ước lượng tứ giác biển từ ứng viên.

    Input: ảnh BGR gốc, ứng viên, method ∈ {minarearect, approxpoly}.
    Output: Quad TL, TR, BR, BL theo tọa độ ảnh gốc.
    """
    raise NotImplementedError("TODO(A): quad_from_candidate")


def crop_bbox(img: np.ndarray, bbox: BBox, size: tuple[int, int]) -> np.ndarray:
    """Cắt bbox (cắt về trong biên ảnh) và resize về size = (rộng, cao)."""
    raise NotImplementedError("TODO(A): crop_bbox")


def warp_plate(img: np.ndarray, quad: Quad, size: tuple[int, int]) -> np.ndarray:
    """Biến đổi phối cảnh (homography) tứ giác `quad` về ảnh chữ nhật size = (rộng, cao)."""
    raise NotImplementedError("TODO(A): warp_plate")


def rectify(img: np.ndarray, bbox: BBox, quad: Quad | None, method: CropMethod,
            cfg: dict) -> np.ndarray:
    """Nắn biển theo method (bbox: crop_bbox, quad: warp_plate), kích thước cfg["rectify"]["plate_size"].

    Output: ảnh BGR 220×70.
    """
    raise NotImplementedError("TODO(A): rectify")


def corner_error(pred: Quad, gt: Quad) -> float:
    """Trung bình khoảng cách Euclid giữa 4 góc tương ứng của hai quad (pixel)."""
    raise NotImplementedError("TODO(A): corner_error")
