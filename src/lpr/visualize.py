# OWNER: A
"""Vẽ ứng viên, hộp ký tự và lưới ảnh debug (SPEC mục 8.12)."""
from __future__ import annotations

import numpy as np

from lpr.types import BBox, Candidate, CharBox, DebugSteps


def draw_candidates(img: np.ndarray, cands: list[Candidate], gt: BBox | None = None) -> np.ndarray:
    """Vẽ quad/bbox các ứng viên kèm điểm, và bbox thật (nếu có) lên bản sao ảnh BGR."""
    raise NotImplementedError("TODO(A): draw_candidates")


def draw_char_boxes(plate: np.ndarray, boxes: list[CharBox]) -> np.ndarray:
    """Vẽ hộp ký tự lên bản sao ảnh biển (BGR hoặc xám). Output: ảnh BGR."""
    raise NotImplementedError("TODO(A): draw_char_boxes")


def save_debug_grid(debug: DebugSteps, path: str, cols: int = 3) -> None:
    """Ghép các ảnh trong DebugSteps thành lưới `cols` cột, có tiêu đề từng bước, rồi lưu ra `path`."""
    raise NotImplementedError("TODO(A): save_debug_grid")


def save_comparison(images: dict[str, np.ndarray], path: str) -> None:
    """Đặt nhiều ảnh cạnh nhau, mỗi ảnh có tiêu đề là khóa của dict, rồi lưu ra `path`."""
    raise NotImplementedError("TODO(A): save_comparison")
