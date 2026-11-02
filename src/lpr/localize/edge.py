# OWNER: A
"""Phương pháp A: định vị biển số bằng cạnh Sobel-x + hình thái học + heuristic (SPEC mục 8.6)."""
from __future__ import annotations

import numpy as np

from lpr.types import Candidate, DebugSteps


def sobel_x_map(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Tính |Sobel theo x| và chuẩn hóa về uint8 [0, 255].

    Input: ảnh xám uint8, kích thước kernel Sobel. Output: bản đồ cạnh uint8.
    """
    raise NotImplementedError("TODO(A): sobel_x_map")


def binarize_edges(edge: np.ndarray) -> np.ndarray:
    """Nhị phân bản đồ cạnh bằng Otsu. Output: uint8 0/255."""
    raise NotImplementedError("TODO(A): binarize_edges")


def morph_plate_regions(binary: np.ndarray, close_k: tuple[int, int],
                        open_k: tuple[int, int]) -> np.ndarray:
    """Đóng bằng kernel nằm ngang để nối nét ký tự, rồi mở để bỏ nhiễu.

    Input: ảnh nhị phân 0/255, kích thước kernel close/open dạng (rộng, cao).
    Output: mặt nạ vùng ứng viên uint8 0/255.
    """
    raise NotImplementedError("TODO(A): morph_plate_regions")


def extract_regions(mask: np.ndarray) -> list[tuple]:
    """Tìm contour ngoài của mặt nạ và trả về list cv2.minAreaRect ((cx, cy), (w, h), angle)."""
    raise NotImplementedError("TODO(A): extract_regions")


def candidate_features(rect: tuple, img_shape: tuple[int, int],
                       edge: np.ndarray, mask: np.ndarray) -> dict:
    """Tính đặc trưng hình học của một vùng.

    Input: minAreaRect, (cao, rộng) ảnh, bản đồ cạnh, mặt nạ vùng.
    Output: dict {aspect, area_ratio, fill_ratio, edge_density, angle}.
    """
    raise NotImplementedError("TODO(A): candidate_features")


def filter_candidates(rects: list[tuple], img_shape: tuple[int, int], edge: np.ndarray,
                      mask: np.ndarray, cfg: dict) -> list[Candidate]:
    """Lọc vùng theo ngưỡng cfg["localize"], tính điểm heuristic, sắp giảm dần, giữ top_k.

    Output: list Candidate (tọa độ ảnh đã resize, method = "edge").
    """
    raise NotImplementedError("TODO(A): filter_candidates")


def localize_edge(img: np.ndarray, cfg: dict,
                  return_debug: bool = False) -> list[Candidate] | tuple[list[Candidate], DebugSteps]:
    """Chạy toàn bộ phương pháp A trên ảnh BGR gốc.

    Output: list Candidate với tọa độ theo ảnh gốc; nếu return_debug thì trả thêm DebugSteps.
    """
    raise NotImplementedError("TODO(A): localize_edge")
