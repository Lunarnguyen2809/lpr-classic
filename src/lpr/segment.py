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
NARROW_ASPECT = 0.15       # mảnh có rộng/cao < 0.15 là "mảnh hẹp" để gộp; tách khỏi char_aspect_range (#14)
NARROW_MAX_GAP = 0.06     # mảnh hẹp cách mảnh kề ≤ 6% chiều cao biển (~4 px) thì gộp

# Ngưỡng cho merge_split_boxes, tính theo trung vị độ rộng hộp.
MERGE_NARROW = 0.75        # chỉ gộp khi ít nhất một hộp hẹp hơn 0.75 × trung vị
MERGE_MAX_GAP = 0.3        # khe giữa hai mảnh ≤ 0.3 × trung vị
MERGE_MAX_WIDTH = 1.4      # hộp sau khi gộp rộng ≤ 1.4 × trung vị
SPLIT_MIN_WIDTH = 1.7      # hộp rộng ≥ 1.7 × trung vị bị cắt
SPLIT_MAX_PARTS = 3


def _n_parts(w: float, med: float, gap: float) -> int:
    """Số ký tự trong một khối rộng w: round((w + khe) / (trung vị rộng + khe)).

    Khối k ký tự dính rộng k × med + (k − 1) × khe, nên chia cho med + khe (cộng thêm một khe
    ở tử) mới ra đúng k. Chia w / med như trước thì khối 2 ký tự có khe rộng dễ ra 3 (#14).
    """
    if med <= 0 or w < SPLIT_MIN_WIDTH * med:
        return 1
    return min(SPLIT_MAX_PARTS, int(round((w + gap) / (med + gap))))

# Ngưỡng cho chiếu ngang (split_lines) và chiếu dọc (segment_projection).
LINE_TEXT_ROW = 0.03       # hàng có > 3% pixel trắng là hàng có chữ
LINE_FRAME_ROW = 0.85      # hàng > 85% trắng là viền biển
LINE_MAX_GAP = 0.05        # nối hai đoạn hàng cách nhau ≤ 5% chiều cao
LINE_MIN_HEIGHT = 0.2      # dòng chữ cao ≥ 20% chiều cao biển
PROJ_COL_RATIO = 0.1       # cột có ≥ 10% chiều cao dòng là trắng mới là cột chữ (bỏ vệt nhiễu mảnh)
VALLEY_SEARCH = 0.3        # tìm thung lũng trong ± 0.3 × trung vị quanh vị trí cắt dự kiến


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Các đoạn liên tiếp True trong mảng 1 chiều, dạng (đầu, cuối) với cuối không bao gồm."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Nối các đoạn cách nhau ≤ max_gap phần tử trống."""
    out: list[tuple[int, int]] = []
    for a, b in runs:
        if out and a - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def split_lines(binary: np.ndarray) -> list[tuple[int, int]]:
    """Chiếu ngang để tìm các dòng chữ.

    Input: ảnh nhị phân (ký tự = 255). Output: list khoảng (y0, y1), y1 không bao gồm.
    Hàng có > 3% pixel trắng là hàng chữ, hàng > 85% trắng là viền (bỏ). Nối đoạn cách nhau
    ≤ 5% chiều cao, bỏ dòng thấp hơn 20% chiều cao. Không có dòng nào thì trả [(0, h)].
    """
    h, w = binary.shape[:2]
    prof = np.count_nonzero(binary, axis=1) / max(w, 1)
    text_rows = (prof > LINE_TEXT_ROW) & (prof < LINE_FRAME_ROW)
    runs = _merge_runs(_runs(text_rows), max_gap=max(1, int(LINE_MAX_GAP * h)))
    lines = [(a, b) for a, b in runs if b - a >= LINE_MIN_HEIGHT * h]
    return lines or [(0, h)]


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


def _is_frame_or_noise(c: list[int], plate_w: int, plate_h: int, check_flat: bool = True) -> bool:
    """Viền biển, mảnh viền ngang (bỏ qua nếu check_flat = False), ốc vít / dấu chấm sát mép trên-dưới.

    Thành phần rộng chỉ bị coi là viền khi rỗng (fill thấp); khối đặc rộng thường là nhiều
    ký tự dính nhau, giữ lại để merge_split_boxes cắt.
    """
    x1, y1, x2, y2, _ = c
    w, h = x2 - x1, y2 - y1
    if w > FRAME_WIDTH_RATIO * plate_w and _fill(c) < DENSE_FILL:
        return True
    if check_flat and h < FLAT_HEIGHT_RATIO * plate_h and w > 2 * h:
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


def _merge_narrow_neighbors(comps: list[list[int]], max_gap: float) -> list[list[int]]:
    """Gộp mảnh hẹp (rộng/cao < NARROW_ASPECT) với mảnh kề bên trái nếu khe ≤ max_gap.

    Ký tự Hán vỡ trái/phải cho hai mảnh quá hẹp, bị lọc tỉ lệ loại mất trước khi
    merge_split_boxes kịp gộp, nên phải gộp ngay ở đây. Số "1" hẹp nhưng cách xa
    ký tự bên cạnh nên không bị gộp. Ngưỡng tách riêng khỏi char_aspect_range: hạ min của
    bộ lọc (để giữ nét "1") không được làm đổi luật gộp mảnh chữ Hán.
    """
    out: list[list[int]] = []
    for c in sorted(comps, key=lambda c: c[0]):
        if out:
            p = out[-1]
            narrow = lambda b: (b[2] - b[0]) < NARROW_ASPECT * max(b[3] - b[1], 1)  # noqa: E731
            if (narrow(p) or narrow(c)) and c[0] - p[2] <= max_gap:
                out[-1] = [p[0], min(p[1], c[1]), max(p[2], c[2]), max(p[3], c[3]), p[4] + c[4]]
                continue
        out.append(list(c))
    return out


def segment_cca(binary: np.ndarray, cfg: dict) -> list[CharBox]:
    """Tách ký tự bằng connected components, lọc theo cfg["segment"] (chiều cao, tỉ lệ, diện tích).

    Các bước: CCA → bỏ viền / ốc vít → gộp mảnh chồng dọc → bỏ mảnh dẹt → gộp mảnh hẹp
    → lọc chiều cao, tỉ lệ, diện tích
    → sắp trái sang phải. Chiều cao so với chiều cao biển (CCPD chỉ có 1 dòng).
    Output: list CharBox.
    """
    seg = cfg["segment"]
    h_lo, h_hi = seg["char_height_range"]
    a_lo, a_hi = seg["char_aspect_range"]
    min_area = seg["min_area"]
    ph, pw = binary.shape[:2]

    # lọc mảnh dẹt SAU khi gộp mảnh chồng dọc: nét ngang của chữ Hán (thấp, dẹt) phải được gộp
    # vào ký tự trước; mảnh dẹt còn lại một mình (vệt nhiễu, vạch) mới bị bỏ (#14)
    comps = [c for c in _components(binary) if not _is_frame_or_noise(c, pw, ph, check_flat=False)]
    comps = [c for c in _merge_stacked(comps)
             if not (c[3] - c[1] < FLAT_HEIGHT_RATIO * ph and c[2] - c[0] > 2 * (c[3] - c[1]))]
    comps = _merge_narrow_neighbors(comps, NARROW_MAX_GAP * ph)
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


def _without_noise(binary: np.ndarray) -> np.ndarray:
    """Bản sao ảnh đã xóa viền rỗng, mảnh viền ngang, ốc vít (cùng luật với CCA)."""
    ph, pw = binary.shape[:2]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        keep[i] = not _is_frame_or_noise([x, y, x + w, y + h, area], pw, ph)
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def _cut_at_valleys(a: int, b: int, prof: np.ndarray, med: float, gap: float = 0.0) -> list[tuple[int, int]]:
    """Cắt dải cột [a, b) rộng ≥ 1.7 × trung vị thành k phần tại cột ít pixel trắng nhất
    quanh mỗi vị trí cắt dự kiến (± 0.3 × trung vị)."""
    w = b - a
    k = _n_parts(w, med, gap)
    if k <= 1:
        return [(a, b)]
    cuts, start = [], a
    for j in range(1, k):
        guess = a + j * w / k
        lo = max(start + 1, int(guess - VALLEY_SEARCH * med))
        hi = min(b - 1, int(guess + VALLEY_SEARCH * med) + 1)
        if hi <= lo:
            continue
        c = lo + int(np.argmin(prof[lo:hi]))
        cuts.append(c)
        start = c
    edges = [a, *cuts, b]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]


def segment_projection(binary: np.ndarray, cfg: dict) -> list[CharBox]:
    """Tách ký tự bằng chiếu dọc, dùng cfg["segment"]["projection"] (min_gap, min_width).

    Các bước: bỏ viền / ốc vít → split_lines → với mỗi dòng: profile cột (số pixel trắng),
    cột ≥ 10% chiều cao dòng là cột chữ → nối dải cách nhau < min_gap, bỏ dải hẹp hơn
    min_width → dải rộng (chữ dính) cắt tại thung lũng của profile → co hộp theo y →
    lọc chiều cao, tỉ lệ, diện tích như CCA.
    Output: list CharBox (line = chỉ số dòng).
    """
    seg = cfg["segment"]
    pj = seg["projection"]
    h_lo, h_hi = seg["char_height_range"]
    a_lo, a_hi = seg["char_aspect_range"]
    ph = binary.shape[0]

    clean = _without_noise(binary)
    lines = split_lines(clean)
    ref_h = ph / len(lines)          # 1 dòng: so với chiều cao biển; nhiều dòng: chia đều
    boxes: list[CharBox] = []
    for li, (y0, y1) in enumerate(lines):
        band = clean[y0:y1]
        prof = np.count_nonzero(band, axis=0)
        cols = prof >= max(1.0, PROJ_COL_RATIO * (y1 - y0))
        runs = _merge_runs(_runs(cols), max_gap=int(pj["min_gap"]) - 1)
        # dải hẹp (mảnh ký tự Hán vỡ trái/phải) nối với dải kề nếu khe nhỏ, cùng luật như CCA
        narrow_w, gap = NARROW_ASPECT * (y1 - y0), NARROW_MAX_GAP * ph
        merged: list[tuple[int, int]] = []
        for a, b in runs:
            if merged and a - merged[-1][1] <= gap and min(b - a, merged[-1][1] - merged[-1][0]) < narrow_w:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        runs = [(a, b) for a, b in merged if b - a >= pj["min_width"]]
        if not runs:
            continue
        med = float(np.median([b - a for a, b in runs]))
        gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
        med_gap = float(np.median(gaps)) if gaps else 0.0
        for a0, b0 in runs:
            for a, b in _cut_at_valleys(a0, b0, prof, med, med_gap):
                rows = np.flatnonzero(np.count_nonzero(band[:, a:b], axis=1))
                if rows.size == 0:
                    continue
                by0, by1 = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
                h, w = by1 - by0, b - a
                area = int(np.count_nonzero(band[:, a:b]))
                if area < seg["min_area"] or not (h_lo * ref_h <= h <= h_hi * ref_h):
                    continue
                if not (a_lo <= w / h <= a_hi):
                    continue
                boxes.append(CharBox(a, by0, b, by1, li))
    return sorted(boxes, key=lambda bx: (bx.line, bx.x1))


def merge_split_boxes(boxes: list[CharBox], plate_w: int, plate_h: int) -> list[CharBox]:
    """Gộp mảnh nhỏ sát nhau (ký tự Hán vỡ nét), cắt đôi hộp rộng gấp ~2 lần trung vị.

    - Gộp: hai hộp kề nhau cùng dòng, ít nhất một hộp hẹp (< 0.75 × trung vị), khe ≤ 0.3 × trung vị,
      hộp gộp rộng ≤ 1.4 × trung vị. Số "1" hẹp nhưng cách xa ký tự bên cạnh nên không bị gộp.
    - Cắt: hộp rộng ≥ 1.7 × trung vị chia đều thành round((w + khe) / (trung vị + khe)) phần
      (tối đa 3), khe = trung vị khe giữa các hộp kề nhau.
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

    gaps = [q.x1 - p.x2 for p, q in zip(merged, merged[1:]) if q.line == p.line and q.x1 > p.x2]
    med_gap = float(np.median(gaps)) if gaps else 0.0
    out: list[CharBox] = []
    for b in merged:
        w = b.x2 - b.x1
        k = _n_parts(w, med, med_gap)
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
