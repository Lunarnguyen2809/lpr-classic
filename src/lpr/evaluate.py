# OWNER: B
"""Metric định vị và tách ký tự (SPEC mục 8.11 và 9)."""
from __future__ import annotations

from lpr.types import BBox, CharBox

PLATE_WIDTH_MM = 440  # chiều rộng biển xanh Trung Quốc (mm), khớp char_template_mm


def iou(a: BBox, b: BBox) -> float:
    """IoU của hai bbox (x1, y1, x2, y2), x2/y2 không bao gồm. Output trong [0, 1]."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(preds: list[BBox], gts: list[BBox], thr: float) -> tuple[int, int, int]:
    """Ghép tham lam dự đoán với nhãn theo IoU ≥ thr. Output: (TP, FP, FN)."""
    pairs = sorted(
        ((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)),
        key=lambda t: -t[0],
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    for v, i, j in pairs:
        if v < thr or v <= 0:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
    tp = len(used_p)
    return tp, len(preds) - tp, len(gts) - tp


def prf(tp: int, fp: int, fn: int) -> dict:
    """Output: {"precision", "recall", "f1"}, trả 0 khi mẫu số bằng 0."""
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    """IoU của hai khoảng trên một trục. Output trong [0, 1]."""
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def template_intervals(cfg: dict) -> list[tuple[float, float]]:
    """Đổi cfg["segment"]["char_template_mm"] (biển 440×140 mm) sang pixel theo cfg["rectify"]["plate_size"].

    Output: 7 khoảng (x0, x1) pixel.
    """
    scale = cfg["rectify"]["plate_size"][0] / PLATE_WIDTH_MM
    return [(lo * scale, hi * scale) for lo, hi in cfg["segment"]["char_template_mm"]]


def position_hits(boxes: list[CharBox], cfg: dict) -> list[bool]:
    """Với mỗi vị trí i trong 7 vị trí khuôn, True nếu có hộp có IoU trục x ≥ 0.5 với khoảng i (SPEC 9.3).

    Output: list 7 bool.
    """
    thr = cfg.get("evaluate", {}).get("char_iou_threshold", 0.5)
    return [
        any(iou_1d((b.x1, b.x2), iv) >= thr for b in boxes)
        for iv in template_intervals(cfg)
    ]


def count_match(boxes: list[CharBox], text: str) -> bool:
    """True nếu số hộp ký tự bằng độ dài chuỗi biển."""
    return len(boxes) == len(text)


def char_box_prf(pred: list[CharBox], gt: list[CharBox], thr: float) -> dict:
    """Precision/Recall/F1 theo hộp ký tự, khớp theo IoU ≥ thr (SPEC 9.4)."""
    to_bbox = lambda c: (c.x1, c.y1, c.x2, c.y2)  # noqa: E731
    tp, fp, fn = match_detections([to_bbox(c) for c in pred], [to_bbox(c) for c in gt], thr)
    return prf(tp, fp, fn)
