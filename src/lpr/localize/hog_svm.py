# OWNER: A
"""Phương pháp A+SVM: chấm điểm lại ứng viên bằng HOG + LinearSVC (SPEC mục 8.7)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lpr.types import Candidate, PlateRecord

if TYPE_CHECKING:
    from sklearn.svm import LinearSVC


def extract_hog(patch: np.ndarray, cfg: dict) -> np.ndarray:
    """Trích đặc trưng HOG.

    Input: patch xám uint8 (bất kỳ kích thước, sẽ resize về cfg["hog_svm"]["patch_size"]).
    Output: vector đặc trưng 1 chiều float.
    """
    raise NotImplementedError("TODO(A): extract_hog")


def build_patch_dataset(records: list[PlateRecord], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Dựng tập patch huấn luyện, chỉ từ split tune.

    Dương: crop bbox thật. Âm: ứng viên sai của localize_edge (IoU < 0.3) + vùng ngẫu nhiên
    (tổng neg_per_pos âm cho mỗi dương).
    Output: (X đặc trưng HOG shape (N, D), y nhãn 0/1 shape (N,)).
    """
    raise NotImplementedError("TODO(A): build_patch_dataset")


def train_svm(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple["LinearSVC", dict]:
    """Huấn luyện LinearSVC với C = cfg["hog_svm"]["C"].

    Output: (model đã fit trên toàn bộ dữ liệu, dict kết quả cross-validation 5 fold).
    """
    raise NotImplementedError("TODO(A): train_svm")


def save_model(model, path: str) -> None:
    """Lưu model bằng joblib, tự tạo thư mục cha."""
    raise NotImplementedError("TODO(A): save_model")


def load_model(path: str):
    """Nạp model đã lưu bằng joblib."""
    raise NotImplementedError("TODO(A): load_model")


def score_candidates(img: np.ndarray, candidates: list[Candidate], model, cfg: dict) -> list[Candidate]:
    """Chấm điểm lại ứng viên bằng SVM.

    Input: ảnh BGR gốc, danh sách ứng viên, model, cfg.
    Output: list Candidate mới với score = decision_function, method = "edge_svm", sắp giảm dần.
    """
    raise NotImplementedError("TODO(A): score_candidates")


def localize_edge_svm(img: np.ndarray, model, cfg: dict) -> list[Candidate]:
    """localize_edge rồi score_candidates. Output: list Candidate theo tọa độ ảnh gốc."""
    raise NotImplementedError("TODO(A): localize_edge_svm")
