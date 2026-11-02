# OWNER: B
"""Metric định vị và tách ký tự (SPEC mục 8.11 và 9)."""
from __future__ import annotations

from lpr.types import BBox, CharBox


def iou(a: BBox, b: BBox) -> float:
    """IoU của hai bbox (x1, y1, x2, y2), x2/y2 không bao gồm. Output trong [0, 1]."""
    raise NotImplementedError("TODO(B): iou")


def match_detections(preds: list[BBox], gts: list[BBox], thr: float) -> tuple[int, int, int]:
    """Ghép tham lam dự đoán với nhãn theo IoU ≥ thr. Output: (TP, FP, FN)."""
    raise NotImplementedError("TODO(B): match_detections")


def prf(tp: int, fp: int, fn: int) -> dict:
    """Output: {"precision", "recall", "f1"}, trả 0 khi mẫu số bằng 0."""
    raise NotImplementedError("TODO(B): prf")


def iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    """IoU của hai khoảng trên một trục. Output trong [0, 1]."""
    raise NotImplementedError("TODO(B): iou_1d")


def template_intervals(cfg: dict) -> list[tuple[float, float]]:
    """Đổi cfg["segment"]["char_template_mm"] (biển 440×140 mm) sang pixel theo cfg["rectify"]["plate_size"].

    Output: 7 khoảng (x0, x1) pixel.
    """
    raise NotImplementedError("TODO(B): template_intervals")


def position_hits(boxes: list[CharBox], cfg: dict) -> list[bool]:
    """Với mỗi vị trí i trong 7 vị trí khuôn, True nếu có hộp có IoU trục x ≥ 0.5 với khoảng i (SPEC 9.3).

    Output: list 7 bool.
    """
    raise NotImplementedError("TODO(B): position_hits")


def count_match(boxes: list[CharBox], text: str) -> bool:
    """True nếu số hộp ký tự bằng độ dài chuỗi biển."""
    raise NotImplementedError("TODO(B): count_match")


def char_box_prf(pred: list[CharBox], gt: list[CharBox], thr: float) -> dict:
    """Precision/Recall/F1 theo hộp ký tự, khớp theo IoU ≥ thr (SPEC 9.4)."""
    raise NotImplementedError("TODO(B): char_box_prf")
