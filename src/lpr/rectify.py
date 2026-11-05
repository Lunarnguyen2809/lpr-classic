# OWNER: A
"""Nắn biển số về hình chữ nhật chuẩn 220×70 (SPEC mục 8.8)."""
from __future__ import annotations

import cv2
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


_S_MIN = 70          # độ bão hòa tối thiểu của "xanh biển"


def _plate_mask(crop: np.ndarray) -> np.ndarray:
    """Mặt nạ biển trong vùng cắt: HSV xanh dương (biển CCPD) rồi đóng lỗ ký tự trắng."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (95, _S_MIN, 40), (135, 255, 255))
    k = max(3, int(round(min(crop.shape[:2]) * 0.12)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _approx_quad(img: np.ndarray, cand: Candidate) -> Quad | None:
    """Tinh chỉnh 4 góc bằng approxPolyDP trên mặt nạ biển; None nếu thất bại/không hợp lý."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = cand.bbox
    pad = 0.15
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    mask = _plate_mask(img[cy1:cy2, cx1:cx2])
    # chỉ giữ phần mặt nạ nằm trong quad ứng viên (chống rò nền xanh)
    cq = cand.quad.astype(np.float32) - np.array([cx1, cy1], np.float32)
    roi = np.zeros_like(mask)
    cv2.fillConvexPoly(roi, np.round(cq).astype(np.int32), 255)
    mask = cv2.bitwise_and(mask, roi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # chọn contour lớn nhất chứa tâm bbox, nếu không có thì lớn nhất
    ctr = np.array([(x1 + x2) / 2 - cx1, (y1 + y2) / 2 - cy1], np.float32)
    inside = [c for c in cnts if cv2.pointPolygonTest(c, (float(ctr[0]), float(ctr[1])), False) >= 0]
    cnt = max(inside or cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    per = cv2.arcLength(hull, True)
    quad = None
    # tăng dần epsilon (1%..15% chu vi) tới khi còn đúng 4 đỉnh
    for eps in np.arange(0.01, 0.151, 0.005):
        ap = cv2.approxPolyDP(hull, eps * per, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(np.float32)
            break
        if len(ap) < 4:
            break
    if quad is None:
        return None
    quad += np.array([cx1, cy1], np.float32)
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return None
    ref = max(1, (x2 - x1) * (y2 - y1))
    ratio = cv2.contourArea(quad.reshape(-1, 1, 2)) / ref
    # cận trên 3.0 chỉ để phòng thủ: không thể xảy ra khi mặt nạ bị cắt theo quad ứng viên
    if ratio < 0.3 or ratio > 3.0:
        return None
    return order_quad(quad)


def quad_from_candidate(img: np.ndarray, cand: Candidate, method: str) -> Quad:
    """Ước lượng tứ giác biển từ ứng viên.

    Input: ảnh BGR gốc, ứng viên, method ∈ {minarearect, approxpoly}.
    Output: Quad TL, TR, BR, BL theo tọa độ ảnh gốc.
    """
    if method == "minarearect":
        return order_quad(cand.quad).astype(np.float32)
    if method == "approxpoly":
        # mặt nạ xanh dương trong bbox nới 15% -> contour lớn nhất -> convex hull -> approxPolyDP
        # (epsilon tăng dần tới 4 đỉnh). Thất bại/không hợp lý (diện tích <30% hay >300% bbox,
        # không lồi) thì lùi về quad minAreaRect của ứng viên.
        q = _approx_quad(img, cand)
        return q if q is not None else order_quad(cand.quad).astype(np.float32)
    raise ValueError(f"method không hợp lệ: {method!r} (minarearect | approxpoly)")


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def crop_bbox(img: np.ndarray, bbox: BBox, size: tuple[int, int]) -> np.ndarray:
    """Cắt bbox (cắt về trong biên ảnh) và resize về size = (rộng, cao)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, min(w, int(x1))), max(0, min(w, int(x2)))
    y1, y2 = max(0, min(h, int(y1))), max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox {bbox} rỗng sau khi cắt về ảnh {w}x{h}")
    crop = img[y1:y2, x1:x2]
    interp = cv2.INTER_AREA if crop.shape[1] > size[0] else cv2.INTER_LINEAR
    return cv2.resize(crop, (int(size[0]), int(size[1])), interpolation=interp)


def warp_plate(img: np.ndarray, quad: Quad, size: tuple[int, int]) -> np.ndarray:
    """Biến đổi phối cảnh (homography) tứ giác `quad` về ảnh chữ nhật size = (rộng, cao)."""
    w, h = int(size[0]), int(size[1])
    src = order_quad(quad)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def rectify(img: np.ndarray, bbox: BBox, quad: Quad | None, method: CropMethod,
            cfg: dict) -> np.ndarray:
    """Nắn biển theo method (bbox: crop_bbox, quad: warp_plate), kích thước cfg["rectify"]["plate_size"].

    Output: ảnh BGR 220×70.
    """
    pw, ph = cfg["rectify"]["plate_size"]
    size = (int(pw), int(ph))
    img = _to_bgr(img)
    if method == "quad":
        # quad None (vd. ảnh VN không có góc) -> lùi về cắt bbox
        if quad is None:
            return crop_bbox(img, bbox, size)
        return warp_plate(img, quad, size)
    if method == "bbox":
        return crop_bbox(img, bbox, size)
    raise ValueError(f"method không hợp lệ: {method!r} (bbox | quad)")


def corner_error(pred: Quad, gt: Quad) -> float:
    """Trung bình khoảng cách Euclid giữa 4 góc tương ứng của hai quad (pixel)."""
    p, g = order_quad(pred), order_quad(gt)
    return float(np.linalg.norm(p - g, axis=1).mean())
