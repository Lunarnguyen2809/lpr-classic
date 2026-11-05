# OWNER: A
"""Phương pháp A: định vị biển số bằng cạnh Sobel-x + hình thái học + heuristic (SPEC mục 8.6)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from lpr.preprocess import preprocess
from lpr.types import Candidate, DebugSteps
from lpr.rectify import order_quad

PLATE_ASPECT = 440 / 140                   # tỉ lệ biển xanh Trung Quốc ≈ 3.14


def sobel_x_map(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Tính |Sobel theo x| và chuẩn hóa về uint8 [0, 255].

    Input: ảnh xám uint8, kích thước kernel Sobel. Output: bản đồ cạnh uint8.
    """
    g = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    return cv2.normalize(np.abs(g), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def binarize_edges(edge: np.ndarray) -> np.ndarray:
    """Nhị phân bản đồ cạnh bằng Otsu. Output: uint8 0/255."""
    _, b = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return b


def morph_plate_regions(binary: np.ndarray, close_k: tuple[int, int],
                        open_k: tuple[int, int]) -> np.ndarray:
    """Đóng bằng kernel nằm ngang để nối nét ký tự, rồi mở để bỏ nhiễu.

    Input: ảnh nhị phân 0/255, kích thước kernel close/open dạng (rộng, cao).
    Output: mặt nạ vùng ứng viên uint8 0/255.
    """
    close_se = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k[0], close_k[1]))
    open_se = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k[0], open_k[1]))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_se)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_se)


def extract_regions(mask: np.ndarray) -> list[tuple]:
    """Tìm contour ngoài của mặt nạ và trả về list cv2.minAreaRect ((cx, cy), (w, h), angle)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.minAreaRect(c) for c in contours if len(c) >= 3]


def _rect_angle(box: np.ndarray) -> float:
    """Góc (độ) của cạnh dài so với trục x, chuẩn hóa về (-90, 90]; biển nằm ngang ≈ 0.

    Tính từ 4 đỉnh thay vì đọc angle của minAreaRect vì quy ước angle đổi giữa các bản OpenCV
    (ở 5.0 hình chữ nhật ngang trả (w, h) = (ngắn, dài), angle = -90).
    """
    e1, e2 = box[1] - box[0], box[2] - box[1]
    v = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    a = math.degrees(math.atan2(v[1], v[0]))
    if a > 90:
        a -= 180
    elif a <= -90:
        a += 180
    return a


def candidate_features(rect: tuple, img_shape: tuple[int, int],
                       edge: np.ndarray, mask: np.ndarray) -> dict:
    """Tính đặc trưng hình học của một vùng.

    Input: minAreaRect, (cao, rộng) ảnh, bản đồ cạnh, mặt nạ vùng.
    Output: dict {aspect, area_ratio, fill_ratio, edge_density, angle}.
    `edge` phải là ảnh nhị phân 0/255 (edge_density = tỉ lệ pixel > 0 trong hình chữ nhật).
    """
    h, w = img_shape
    box = cv2.boxPoints(rect).astype(np.float32)
    rw, rh = rect[1]
    long_s, short_s = max(rw, rh), min(rw, rh)
    rect_area = max(rw * rh, 1e-6)
    # chỉ xét ROI bao quanh rect để nhanh
    x0, y0 = np.floor(box.min(axis=0)).astype(int)
    x1, y1 = np.ceil(box.max(axis=0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    inside = np.zeros((max(y1 - y0, 0), max(x1 - x0, 0)), np.uint8)
    if inside.size:
        cv2.fillPoly(inside, [np.round(box - [x0, y0]).astype(np.int32)], 1)
    n_in = int(inside.sum())
    if n_in:
        fill = float(((mask[y0:y1, x0:x1] > 0) & (inside > 0)).sum()) / n_in
        dens = float(((edge[y0:y1, x0:x1] > 0) & (inside > 0)).sum()) / n_in
    else:
        fill = dens = 0.0
    return {
        "aspect": float(long_s / short_s) if short_s > 1e-6 else float("inf"),
        "area_ratio": float(rect_area / (h * w)),
        "fill_ratio": fill,
        # edge_density: tỉ lệ pixel cạnh trong rect, với `edge` là bản đồ cạnh ĐÃ nhị phân (0/255)
        "edge_density": dens,
        "angle": float(_rect_angle(box)),
    }


def filter_candidates(rects: list[tuple], img_shape: tuple[int, int], edge: np.ndarray,
                      mask: np.ndarray, cfg: dict) -> list[Candidate]:
    """Lọc vùng theo ngưỡng cfg["localize"], tính điểm heuristic, sắp giảm dần, giữ top_k.

    Output: list Candidate (tọa độ ảnh đã resize, method = "edge").
    """
    lc = cfg["localize"]
    a_lo, a_hi = lc["aspect_range"]
    r_lo, r_hi = lc["area_ratio_range"]
    out: list[Candidate] = []
    h, w = img_shape
    for rect in rects:
        f = candidate_features(rect, img_shape, edge, mask)
        if not (a_lo <= f["aspect"] <= a_hi and r_lo <= f["area_ratio"] <= r_hi):
            continue
        if f["fill_ratio"] < lc["min_fill_ratio"] or f["edge_density"] < lc["min_edge_density"]:
            continue
        if abs(f["angle"]) > lc["max_angle"]:
            continue
        # score = 0.4*edge_density + 0.3*fill_ratio + 0.3*exp(-ln(aspect/3.14)^2 / (2*0.3^2))
        # (edge_density, fill_ratio trong [0,1]; số hạng cuối = 1 khi aspect đúng tỉ lệ biển 440:140)
        closeness = math.exp(-math.log(f["aspect"] / PLATE_ASPECT) ** 2 / (2 * 0.3 ** 2))
        score = 0.4 * f["edge_density"] + 0.3 * f["fill_ratio"] + 0.3 * closeness
        box = cv2.boxPoints(rect).astype(np.float32)
        x1, y1 = np.floor(box.min(axis=0)).astype(int)
        x2, y2 = np.ceil(box.max(axis=0)).astype(int)  # x2/y2 exclusive
        out.append(Candidate(bbox=(int(x1), int(y1), int(x2), int(y2)), quad=order_quad(box),
                             score=float(score), method="edge", features=f))
    out.sort(key=lambda c: -c.score)
    return out[: lc["top_k"]]


def localize_edge(img: np.ndarray, cfg: dict,
                  return_debug: bool = False) -> list[Candidate] | tuple[list[Candidate], DebugSteps]:
    """Chạy toàn bộ phương pháp A trên ảnh BGR gốc.

    Output: list Candidate với tọa độ theo ảnh gốc; nếu return_debug thì trả thêm DebugSteps.
    """
    lc = cfg["localize"]
    gray, scale, steps = preprocess(img, cfg)
    edge = sobel_x_map(gray, lc["sobel_ksize"])
    binary = binarize_edges(edge)
    mask = morph_plate_regions(binary, tuple(lc["close_kernel"]), tuple(lc["open_kernel"]))
    rects = extract_regions(mask)
    cands = filter_candidates(rects, gray.shape[:2], binary, mask, cfg)
    ih, iw = img.shape[:2]
    for c in cands:
        # quay về tọa độ ảnh gốc: chia scale, bbox cắt trong biên ảnh
        c.quad = (c.quad / scale).astype(np.float32)
        x1, y1, x2, y2 = (v / scale for v in c.bbox)
        bx1 = min(max(0, int(math.floor(x1))), iw - 1)
        by1 = min(max(0, int(math.floor(y1))), ih - 1)
        bx2 = max(min(iw, int(math.ceil(x2))), bx1 + 1)
        by2 = max(min(ih, int(math.ceil(y2))), by1 + 1)
        c.bbox = (bx1, by1, bx2, by2)
    if not return_debug:
        return cands
    steps.images["sobel"] = edge
    steps.images["otsu"] = binary
    steps.images["morph"] = mask
    vis = steps.images["resized"].copy()
    for c in cands:
        cv2.polylines(vis, [np.round(c.quad * scale).astype(np.int32)], True, (0, 255, 0), 2)
    steps.images["candidates"] = vis
    return cands, steps
