# OWNER: B
"""Phân tích khả năng tách biển/nền của từng kênh màu (SPEC mục 8.5, thí nghiệm E1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lpr.io_utils import read_image
from lpr.preprocess import get_channel
from lpr.types import BBox, PlateRecord

# OpenCV lưu H trong [0, 180), các kênh còn lại trong [0, 256). Dùng đúng dải để bin không bị trống nửa.
CHANNEL_RANGE = {"H": (0, 180)}
DEFAULT_RANGE = (0, 256)


def plate_mask(shape: tuple[int, int], bbox: BBox) -> np.ndarray:
    """Tạo mặt nạ biển số.

    Input: (cao, rộng) của ảnh, bbox (x1, y1, x2, y2).
    Output: ảnh uint8 cùng kích thước, 255 trong bbox, 0 ngoài bbox.
    bbox được cắt về trong biên ảnh; x2, y2 không bao gồm.
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, min(w, int(x1))), max(0, min(w, int(x2)))
    y1, y2 = max(0, min(h, int(y1))), max(0, min(h, int(y2)))
    mask = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _norm_hist(values: np.ndarray, bins: int, rng: tuple[int, int]) -> np.ndarray:
    """Histogram chuẩn hóa tổng = 1; không có pixel nào thì trả về toàn 0."""
    hist, _ = np.histogram(values, bins=bins, range=rng)
    total = hist.sum()
    return hist / total if total > 0 else hist.astype(float)


def channel_histograms(img: np.ndarray, bbox: BBox, channels: list[str],
                       bins: int = 64) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Tính histogram trong và ngoài vùng biển cho từng kênh.

    Input: ảnh BGR, bbox biển, danh sách tên kênh (gray, H, S, V, L, a, b), số bin.
    Output: dict kênh -> (hist trong biển, hist ngoài biển), mỗi hist đã chuẩn hóa tổng = 1.
    "Ngoài biển" là toàn bộ phần ảnh còn lại (xe, đường, nền), không chỉ một viền quanh biển.
    """
    inside = plate_mask(img.shape[:2], bbox) > 0
    out = {}
    for ch in channels:
        c = get_channel(img, ch)
        rng = CHANNEL_RANGE.get(ch, DEFAULT_RANGE)
        out[ch] = (_norm_hist(c[inside], bins, rng), _norm_hist(c[~inside], bins, rng))
    return out


def separability(h_in: np.ndarray, h_out: np.ndarray) -> float:
    """Khoảng cách Bhattacharyya giữa hai histogram đã chuẩn hóa (càng lớn càng dễ tách).

    D = sqrt(1 − BC), BC = Σ sqrt(p·q) là hệ số Bhattacharyya (độ chồng lấn hai phân bố).
    Dùng dạng này (giống cv2.HISTCMP_BHATTACHARYYA) vì nó nằm trong [0, 1]: 0 = hai phân bố
    giống hệt, 1 = không chồng lấn. Dạng −ln(BC) không bị chặn trên, một ảnh tách hẳn sẽ kéo
    lệch giá trị trung bình khi gộp nhiều ảnh.
    """
    p = np.asarray(h_in, dtype=float)
    q = np.asarray(h_out, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p, q = p / p.sum(), q / q.sum()
    bc = float(np.sum(np.sqrt(p * q)))
    return float(np.sqrt(max(0.0, 1.0 - min(bc, 1.0))))


def separability_table(records: list[PlateRecord], channels: list[str],
                       bins: int = 64) -> pd.DataFrame:
    """Độ tách từng ảnh × kênh. Output: DataFrame cột image_id, subset, channel, separability."""
    rows = []
    for r in records:
        hists = channel_histograms(read_image(r.path), r.bbox, channels, bins)
        for ch, (h_in, h_out) in hists.items():
            rows.append({"image_id": r.image_id, "subset": r.subset, "channel": ch,
                         "separability": separability(h_in, h_out)})
    return pd.DataFrame(rows)


def channel_ranking(records: list[PlateRecord], channels: list[str]) -> pd.DataFrame:
    """Xếp hạng kênh màu theo độ tách trung bình trên tập ảnh.

    Input: danh sách PlateRecord, danh sách kênh.
    Output: DataFrame cột channel, mean_separability, std; sắp giảm dần theo mean_separability.
    """
    df = separability_table(records, channels)
    if df.empty:
        return pd.DataFrame(columns=["channel", "mean_separability", "std"])
    g = df.groupby("channel")["separability"]
    out = pd.DataFrame({"mean_separability": g.mean(), "std": g.std(ddof=0)}).reset_index()
    return out.sort_values("mean_separability", ascending=False, ignore_index=True)
