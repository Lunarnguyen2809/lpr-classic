# OWNER: B
"""Tách ký tự trên ảnh biển nhị phân 220×70 (SPEC mục 8.10)."""
from __future__ import annotations

import cv2
import numpy as np

from lpr.types import CharBox, SegmentMethod

# Ngưỡng hình học nội bộ của CCA (không phải tham số thí nghiệm nên không đưa vào config).
FRAME_WIDTH_RATIO = 0.3    # thành phần rộng > 30% biển là viền/vạch ngang: không ký tự nào rộng vậy
FLAT_HEIGHT_RATIO = 0.2    # mảnh thấp < 20% chiều cao biển và rộng > 2× cao: mảnh viền ngang
CENTER_BAND = (0.25, 0.75) # tâm y của ký tự nằm giữa biển; ngoài dải này là ốc vít, mép viền
STACK_OVERLAP = 0.5        # hai mảnh chồng nhau ≥ 50% theo x (mảnh hẹp hơn) là cùng một ký tự
DENSE_FILL = 0.35          # fill ≥ 0.35: khối chữ đặc (không phải khung viền rỗng)
NARROW_MAX_GAP = 0.06     # mảnh hẹp cách mảnh kề ≤ 6% chiều cao biển (~4 px) thì gộp

# Ngưỡng cho merge_split_boxes, tính theo trung vị độ rộng hộp.
MERGE_NARROW = 0.75        # chỉ gộp khi ít nhất một hộp hẹp hơn 0.75 × trung vị
MERGE_MAX_GAP = 0.3        # khe giữa hai mảnh ≤ 0.3 × trung vị
MERGE_MAX_WIDTH = 1.4      # hộp sau khi gộp rộng ≤ 1.4 × trung vị
SPLIT_MIN_WIDTH = 1.7      # hộp rộng ≥ 1.7 × trung vị bị cắt
SPLIT_MAX_PARTS = 3


def split_lines(binary: np.ndarray) -> list[tuple[int, int]]:
    """Chiếu ngang để tìm các dòng chữ.

    Input: ảnh nhị phân (ký tự = 255). Output: list khoảng (y0, y1), y1 không bao gồm.
    """
    raise NotImplementedError("TODO(B): split_lines")


def _components(binary: np.ndarray) -> list[list[int]]:
    """Connected components 8-liên thông, trả về [x1, y1, x2, y2, area] (x2/y2 không bao gồm)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):  # nhãn 0 là nền
        x, y, w, h, area = (int(v) for v in stats[i])
        out.append([x, y, x + w, y + h, area])
    return out


def _fill(c: list[int]) -> float:
    """Tỉ lệ pixel trắng trong hộp bao: khung viền rỗng ≈ 0.1, ký tự / khối chữ dính ≈ 0.4–0.7."""
    return c[4] / max(1, (c[2] - c[0]) * (c[3] - c[1]))


def _is_frame_or_noise(c: list[int], plate_w: int, plate_h: int) -> bool:
    """Viền biển, mảnh viền ngang, ốc vít / dấu chấm nằm sát mép trên-dưới.

    Thành phần rộng chỉ bị coi là viền khi rỗng (fill thấp); khối đặc rộng thường là nhiều
    ký tự dính nhau, giữ lại để merge_split_boxes cắt.
    """
    x1, y1, x2, y2, _ = c
    w, h = x2 - x1, y2 - y1
    if w > FRAME_WIDTH_RATIO * plate_w and _fill(c) < DENSE_FILL:
        return True
    if h < FLAT_HEIGHT_RATIO * plate_h and w > 2 * h:
        return True
    cy = (y1 + y2) / 2
    return not (CENTER_BAND[0] * plate_h <= cy <= CENTER_BAND[1] * plate_h)


def _merge_stacked(comps: list[list[int]]) -> list[list[int]]:
    """Gộp các mảnh chồng nhau theo trục x (nét bị đứt theo chiều dọc) thành một hộp."""
    merged: list[list[int]] = []
    for c in sorted(comps, key=lambda c: c[0]):
        for m in merged:
            ov = min(m[2], c[2]) - max(m[0], c[0])
            if ov >= STACK_OVERLAP * min(m[2] - m[0], c[2] - c[0]):
                m[0], m[1] = min(m[0], c[0]), min(m[1], c[1])
                m[2], m[3] = max(m[2], c[2]), max(m[3], c[3])
                m[4] += c[4]
                break
        else:
            merged.append(list(c))
    return merged


def _merge_narrow_neighbors(comps: list[list[int]], a_lo: float, max_gap: float) -> list[list[int]]:
    """Gộp mảnh hẹp (rộng/cao < a_lo) với mảnh kề bên trái nếu khe ≤ max_gap.

    Ký tự Hán vỡ trái/phải cho hai mảnh quá hẹp, bị lọc tỉ lệ loại mất trước khi
    merge_split_boxes kịp gộp, nên phải gộp ngay ở đây. Số "1" hẹp nhưng cách xa
    ký tự bên cạnh nên không bị gộp.
    """
    out: list[list[int]] = []
    for c in sorted(comps, key=lambda c: c[0]):
        if out:
            p = out[-1]
            narrow = lambda b: (b[2] - b[0]) < a_lo * max(b[3] - b[1], 1)  # noqa: E731
            if (narrow(p) or narrow(c)) and c[0] - p[2] <= max_gap:
                out[-1] = [p[0], min(p[1], c[1]), max(p[2], c[2]), max(p[3], c[3]), p[4] + c[4]]
                continue
        out.append(list(c))
    return out


def segment_cca(binary: np.ndarray, cfg: dict) -> list[CharBox]:
    """Tách ký tự bằng connected components, lọc theo cfg["segment"] (chiều cao, tỉ lệ, diện tích).

    Các bước: CCA → bỏ viền / ốc vít → gộp mảnh chồng dọc → lọc chiều cao, tỉ lệ, diện tích
    → sắp trái sang phải. Chiều cao so với chiều cao biển (CCPD chỉ có 1 dòng).
    Output: list CharBox.
    """
    seg = cfg["segment"]
    h_lo, h_hi = seg["char_height_range"]
    a_lo, a_hi = seg["char_aspect_range"]
    min_area = seg["min_area"]
    ph, pw = binary.shape[:2]

    comps = [c for c in _components(binary) if not _is_frame_or_noise(c, pw, ph)]
    comps = _merge_narrow_neighbors(_merge_stacked(comps), a_lo, NARROW_MAX_GAP * ph)
    boxes = []
    for x1, y1, x2, y2, area in comps:
        w, h = x2 - x1, y2 - y1
        if area < min_area:
            continue
        if not (h_lo * ph <= h <= h_hi * ph):
            continue
        aspect = w / h
        stuck = a_hi < aspect <= a_hi * SPLIT_MAX_PARTS and area / (w * h) >= DENSE_FILL
        if not (a_lo <= aspect <= a_hi or stuck):   # khối chữ dính: để merge_split_boxes cắt
            continue
        boxes.append(CharBox(x1, y1, x2, y2))
    return sorted(boxes, key=lambda b: b.x1)


def segment_projection(binary: np.ndarray, cfg: dict) -> list[CharBox]:
    """Tách ký tự bằng chiếu dọc, dùng cfg["segment"]["projection"] (min_gap, min_width).

    Output: list CharBox.
    """
    raise NotImplementedError("TODO(B): segment_projection")


def merge_split_boxes(boxes: list[CharBox], plate_w: int, plate_h: int) -> list[CharBox]:
    """Gộp mảnh nhỏ sát nhau (ký tự Hán vỡ nét), cắt đôi hộp rộng gấp ~2 lần trung vị.

    - Gộp: hai hộp kề nhau cùng dòng, ít nhất một hộp hẹp (< 0.75 × trung vị), khe ≤ 0.3 × trung vị,
      hộp gộp rộng ≤ 1.4 × trung vị. Số "1" hẹp nhưng cách xa ký tự bên cạnh nên không bị gộp.
    - Cắt: hộp rộng ≥ 1.7 × trung vị chia đều thành round(w / trung vị) phần (tối đa 3).
    Cần ≥ 3 hộp để trung vị có nghĩa; ít hơn thì chỉ sao chép.
    Output: list CharBox mới, sắp theo (dòng, x1).
    """
    boxes = sorted(boxes, key=lambda b: (b.line, b.x1))
    if len(boxes) < 3:
        return [CharBox(b.x1, b.y1, b.x2, b.y2, b.line) for b in boxes]
    med = float(np.median([b.x2 - b.x1 for b in boxes]))

    merged: list[CharBox] = []
    for b in boxes:
        if merged:
            p = merged[-1]
            narrow = min(p.x2 - p.x1, b.x2 - b.x1) < MERGE_NARROW * med
            gap = b.x1 - p.x2
            width = max(p.x2, b.x2) - p.x1
            if p.line == b.line and narrow and gap <= MERGE_MAX_GAP * med and width <= MERGE_MAX_WIDTH * med:
                merged[-1] = CharBox(p.x1, min(p.y1, b.y1), max(p.x2, b.x2), max(p.y2, b.y2), p.line)
                continue
        merged.append(CharBox(b.x1, b.y1, b.x2, b.y2, b.line))

    out: list[CharBox] = []
    for b in merged:
        w = b.x2 - b.x1
        k = min(SPLIT_MAX_PARTS, int(round(w / med))) if w >= SPLIT_MIN_WIDTH * med else 1
        if k <= 1:
            out.append(b)
            continue
        edges = np.linspace(b.x1, b.x2, k + 1)
        for i in range(k):
            out.append(CharBox(int(round(edges[i])), b.y1, int(round(edges[i + 1])), b.y2, b.line))
    return out


def segment(binary: np.ndarray, method: SegmentMethod, cfg: dict) -> list[CharBox]:
    """Tách ký tự theo method ∈ {cca, projection}, rồi merge_split_boxes.

    Output: list CharBox sắp từ trái sang phải.
    """
    if method == "cca":
        boxes = segment_cca(binary, cfg)
    elif method == "projection":
        boxes = segment_projection(binary, cfg)
    else:
        raise ValueError(f"method không hợp lệ: {method!r} (cca | projection)")
    ph, pw = binary.shape[:2]
    return sorted(merge_split_boxes(boxes, pw, ph), key=lambda b: (b.line, b.x1))
