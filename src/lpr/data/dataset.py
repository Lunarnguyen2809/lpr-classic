# OWNER: SHARED
"""Nạp nhãn, split và bản ghi biển số (xem SPEC mục 5.5 và 8.3)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from lpr.config import get
from lpr.types import CharBox, PlateRecord, Split

META_COLS = ["area", "tilt_h", "tilt_v", "brightness", "blur"]
QUAD_COLS = ["tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"]


def load_annotations(cfg: dict) -> pd.DataFrame:
    """Đọc annotations.csv và gộp cột `split` từ splits.csv (left join theo image_id).

    Nếu chưa có splits.csv thì cột `split` để trống (NaN).
    """
    df = pd.read_csv(get(cfg, "paths.annotations"), dtype={"image_id": str, "text": str})
    splits_path = get(cfg, "paths.splits")
    if splits_path and os.path.isfile(splits_path):
        sp = pd.read_csv(splits_path, dtype=str)
        df = df.merge(sp, on="image_id", how="left")
    else:
        df["split"] = np.nan
    return df


def _row_to_record(row: pd.Series) -> PlateRecord:
    quad = np.array(row[QUAD_COLS].to_numpy(dtype=np.float32)).reshape(4, 2)
    split = row["split"] if isinstance(row.get("split"), str) else "test"
    return PlateRecord(
        image_id=row["image_id"],
        path=row["file"],
        source="ccpd",
        subset=row["subset"],
        split=split,
        bbox=(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
        quad=quad,
        text=row["text"],
        n_lines=int(row["n_lines"]),
        meta={k: int(row[k]) for k in META_COLS},
    )


def load_records(cfg: dict, split: Split | None = None,
                 subsets: list[str] | None = None) -> list[PlateRecord]:
    """Trả về danh sách PlateRecord của CCPD, lọc theo split và/hoặc subset nếu có."""
    df = load_annotations(cfg)
    if split is not None:
        df = df[df["split"] == split]
    if subsets is not None:
        df = df[df["subset"].isin(subsets)]
    return [_row_to_record(r) for _, r in df.iterrows()]


def load_vn_records(cfg: dict) -> list[PlateRecord]:
    """Đọc ảnh VN tự chụp (data/vn_custom/annotations.csv), split luôn là "qual", quad = None.

    Cột `file` là đường dẫn tương đối so với thư mục chứa annotations.csv nếu không tồn tại từ gốc repo.
    """
    path = get(cfg, "paths.vn")
    if not path or not os.path.isfile(path):
        return []
    df = pd.read_csv(path, dtype={"image_id": str, "file": str})
    base = os.path.dirname(path)
    out = []
    for _, r in df.iterrows():
        f = r["file"]
        if not os.path.isfile(f):
            f = os.path.join(base, f)
        out.append(PlateRecord(
            image_id=r["image_id"],
            path=f,
            source="vn",
            subset="vn",
            split="qual",
            bbox=(int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])),
            quad=None,
            text=None,
            n_lines=int(r["n_lines"]),
        ))
    return out


def load_char_labels(cfg: dict) -> dict[str, list[CharBox]]:
    """Đọc hộp ký tự gán tay: image_id -> list CharBox (tọa độ ảnh biển 220×70), sắp theo char_idx."""
    path = get(cfg, "paths.char_labels")
    if not path or not os.path.isfile(path):
        return {}
    df = pd.read_csv(path, dtype={"image_id": str})
    out: dict[str, list[CharBox]] = {}
    for image_id, g in df.sort_values(["image_id", "char_idx"]).groupby("image_id", sort=False):
        out[image_id] = [CharBox(int(r.x1), int(r.y1), int(r.x2), int(r.y2)) for r in g.itertuples()]
    return out
