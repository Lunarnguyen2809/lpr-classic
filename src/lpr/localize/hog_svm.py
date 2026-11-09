# OWNER: A
"""Phương pháp A+SVM: chấm điểm lại ứng viên bằng HOG + LinearSVC (SPEC mục 8.7)."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import cv2
import joblib
import numpy as np
from skimage.feature import hog

from lpr.evaluate import iou
from lpr.io_utils import ensure_dir, read_image
from lpr.localize.edge import localize_edge
from lpr.types import BBox, Candidate, PlateRecord

if TYPE_CHECKING:
    from sklearn.svm import LinearSVC

NEG_IOU_MAX = 0.3          # ứng viên/vùng ngẫu nhiên có IoU với bbox thật < ngưỡng này mới là mẫu âm
RANDOM_ASPECT = (2.0, 6.0)  # tỉ lệ rộng/cao của vùng ngẫu nhiên (cùng dải với aspect_range của edge)
RANDOM_WIDTH_FRAC = (0.08, 0.45)  # rộng vùng ngẫu nhiên so với rộng ảnh
RANDOM_TRIES = 30


def _to_gray(patch: np.ndarray) -> np.ndarray:
    if patch.ndim == 3:
        return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return patch


def extract_hog(patch: np.ndarray, cfg: dict) -> np.ndarray:
    """Trích đặc trưng HOG.

    Input: patch xám uint8 (bất kỳ kích thước, sẽ resize về cfg["hog_svm"]["patch_size"]).
    Output: vector đặc trưng 1 chiều float.
    Patch 3 kênh (BGR) được đổi sang xám cho an toàn.
    """
    hc = cfg["hog_svm"]
    w, h = hc["patch_size"]
    gray = _to_gray(patch)
    if gray.shape[0] < 1 or gray.shape[1] < 1:
        raise ValueError("patch rỗng")
    interp = cv2.INTER_AREA if gray.shape[1] > w or gray.shape[0] > h else cv2.INTER_LINEAR
    gray = cv2.resize(gray, (int(w), int(h)), interpolation=interp)
    feat = hog(
        gray,
        orientations=int(hc["orientations"]),
        pixels_per_cell=tuple(hc["pixels_per_cell"]),
        cells_per_block=tuple(hc["cells_per_block"]),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def _crop(gray: np.ndarray, bbox: BBox) -> np.ndarray | None:
    """Cắt bbox (x1, y1, x2, y2) trong biên ảnh; None nếu rỗng."""
    ih, iw = gray.shape[:2]
    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(iw, bbox[2]), min(ih, bbox[3])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return gray[y1:y2, x1:x2]


def _random_boxes(gt: BBox, shape: tuple[int, int], n: int, rng: np.random.Generator) -> list[BBox]:
    """Sinh tối đa n vùng ngẫu nhiên dạng chữ nhật ngang, IoU với gt < NEG_IOU_MAX."""
    ih, iw = shape
    out: list[BBox] = []
    for _ in range(RANDOM_TRIES * max(n, 1)):
        if len(out) >= n:
            break
        bw = int(rng.uniform(*RANDOM_WIDTH_FRAC) * iw)
        bh = int(bw / rng.uniform(*RANDOM_ASPECT))
        if bw < 8 or bh < 4 or bw >= iw or bh >= ih:
            continue
        x1 = int(rng.integers(0, iw - bw + 1))
        y1 = int(rng.integers(0, ih - bh + 1))
        box = (x1, y1, x1 + bw, y1 + bh)
        if iou(box, gt) < NEG_IOU_MAX:
            out.append(box)
    return out


def build_patch_dataset(records: list[PlateRecord], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Dựng tập patch huấn luyện, chỉ từ split tune.

    Dương: crop bbox thật. Âm: ứng viên sai của localize_edge (IoU < 0.3) + vùng ngẫu nhiên
    (tổng neg_per_pos âm cho mỗi dương).
    Output: (X đặc trưng HOG shape (N, D), y nhãn 0/1 shape (N,)).
    Raise ValueError nếu có record không thuộc split tune. Ngẫu nhiên dùng cfg["seed"]
    (mỗi ảnh một generator riêng seed + chỉ số) nên kết quả lặp lại được.
    """
    bad = [r.image_id for r in records if r.split != "tune"]
    if bad:
        raise ValueError(f"build_patch_dataset chỉ nhận split tune, gặp {len(bad)} record khác: {bad[:3]}")
    n_neg = int(cfg["hog_svm"]["neg_per_pos"])
    seed = int(cfg.get("seed", 42))
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for i, r in enumerate(records):
        img = read_image(r.path)
        gray = _to_gray(img)
        pos = _crop(gray, r.bbox)
        if pos is None:
            continue
        feats.append(extract_hog(pos, cfg))
        labels.append(1)
        neg_boxes: list[BBox] = []
        for c in localize_edge(img, cfg):  # đã sắp theo điểm giảm dần
            if len(neg_boxes) >= n_neg:
                break
            if iou(c.bbox, r.bbox) < NEG_IOU_MAX:
                neg_boxes.append(c.bbox)
        rng = np.random.default_rng(seed + i)
        neg_boxes += _random_boxes(r.bbox, gray.shape[:2], n_neg - len(neg_boxes), rng)
        for b in neg_boxes:
            patch = _crop(gray, b)
            if patch is not None:
                feats.append(extract_hog(patch, cfg))
                labels.append(0)
    if not feats:
        raise ValueError("không dựng được mẫu nào")
    return np.stack(feats), np.asarray(labels, dtype=np.int64)


def train_svm(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple["LinearSVC", dict]:
    """Huấn luyện LinearSVC với C = cfg["hog_svm"]["C"].

    Output: (model đã fit trên toàn bộ dữ liệu, dict kết quả cross-validation 5 fold).
    Dict CV: {"n_splits", "n_samples", "n_pos", "n_neg", "accuracy", "precision", "recall", "f1"
    (trung bình các fold), "<metric>_std", "folds": [dict metric từng fold]}.
    CV dùng StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]); model cuối fit trên toàn bộ X.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import LinearSVC

    seed = int(cfg.get("seed", 42))
    C = float(cfg["hog_svm"]["C"])

    def make() -> "LinearSVC":
        return LinearSVC(C=C, random_state=seed, max_iter=10000)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[dict] = []
    for tr, te in skf.split(X, y):
        pred = make().fit(X[tr], y[tr]).predict(X[te])
        folds.append({
            "accuracy": float(accuracy_score(y[te], pred)),
            "precision": float(precision_score(y[te], pred, zero_division=0)),
            "recall": float(recall_score(y[te], pred, zero_division=0)),
            "f1": float(f1_score(y[te], pred, zero_division=0)),
        })
    cv: dict = {"n_splits": n_splits, "n_samples": int(len(y)),
                "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}
    for m in ("accuracy", "precision", "recall", "f1"):
        vals = np.array([f[m] for f in folds])
        cv[m] = float(vals.mean())
        cv[m + "_std"] = float(vals.std())
    cv["folds"] = folds
    return make().fit(X, y), cv


def save_model(model, path: str) -> None:
    """Lưu model bằng joblib, tự tạo thư mục cha."""
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    joblib.dump(model, path)


def load_model(path: str):
    """Nạp model đã lưu bằng joblib."""
    return joblib.load(path)


def score_candidates(img: np.ndarray, candidates: list[Candidate], model, cfg: dict) -> list[Candidate]:
    """Chấm điểm lại ứng viên bằng SVM.

    Input: ảnh BGR gốc, danh sách ứng viên, model, cfg.
    Output: list Candidate mới với score = decision_function, method = "edge_svm", sắp giảm dần.
    Ứng viên có bbox rỗng sau khi cắt trong biên ảnh bị bỏ. Không lọc theo score_threshold
    (để tính Recall@k); ai cần thì lọc `score >= cfg["hog_svm"]["score_threshold"]`.
    """
    gray = _to_gray(img)
    kept: list[Candidate] = []
    feats: list[np.ndarray] = []
    for c in candidates:
        patch = _crop(gray, c.bbox)
        if patch is None:
            continue
        kept.append(c)
        feats.append(extract_hog(patch, cfg))
    if not kept:
        return []
    scores = model.decision_function(np.stack(feats))
    out = [Candidate(bbox=c.bbox, quad=c.quad.copy(), score=float(s), method="edge_svm",
                     features=dict(c.features)) for c, s in zip(kept, scores)]
    out.sort(key=lambda c: -c.score)
    return out


def localize_edge_svm(img: np.ndarray, model, cfg: dict) -> list[Candidate]:
    """localize_edge rồi score_candidates. Output: list Candidate theo tọa độ ảnh gốc."""
    return score_candidates(img, localize_edge(img, cfg), model, cfg)
