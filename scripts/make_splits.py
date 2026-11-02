# OWNER: SHARED
"""Chia tập tune/test phân tầng theo subset, ảnh VN gán split `qual` (SPEC mục 5.3, 5.5).

Ví dụ:
    python scripts/make_splits.py --annotations data/annotations.csv --vn data/vn_custom/annotations.csv --out data/splits.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from lpr.config import get, load_config

for _s in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, không in được tiếng Việt
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def make_splits(ann: pd.DataFrame, vn: pd.DataFrame | None, tune_ratio: float, seed: int) -> pd.DataFrame:
    """Chia mỗi subset thành tune (round(n * tune_ratio)) và test; ảnh VN đều là `qual`.

    Input: DataFrame annotations (cần cột image_id, subset), DataFrame VN (cần cột image_id) hoặc None.
    Output: DataFrame (image_id, split), giữ thứ tự image_id của annotations rồi tới VN.
    """
    split_of: dict[str, str] = {}
    for subset, g in ann.groupby("subset", sort=False):
        ids = g["image_id"].sort_values().reset_index(drop=True)
        n_tune = int(round(len(ids) * tune_ratio))
        tune = set(ids.sample(n=n_tune, random_state=seed))
        for i in ids:
            split_of[i] = "tune" if i in tune else "test"
    rows = [{"image_id": i, "split": split_of[i]} for i in ann["image_id"]]
    if vn is not None:
        rows += [{"image_id": i, "split": "qual"} for i in vn["image_id"]]
    return pd.DataFrame(rows, columns=["image_id", "split"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--vn", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ann_path = args.annotations or get(cfg, "paths.annotations")
    vn_path = args.vn or get(cfg, "paths.vn")
    out = args.out or get(cfg, "paths.splits")

    ann = pd.read_csv(ann_path, dtype={"image_id": str})
    vn = pd.read_csv(vn_path, dtype={"image_id": str}) if vn_path and os.path.isfile(vn_path) else None
    df = make_splits(ann, vn, get(cfg, "data.tune_ratio", 0.3), get(cfg, "seed", 42))
    df.to_csv(out, index=False, encoding="utf-8")

    summary = df.merge(ann[["image_id", "subset"]], on="image_id", how="left").fillna({"subset": "vn"})
    print(pd.crosstab(summary["subset"], summary["split"], margins=True))
    print(f"Đã ghi {len(df)} dòng vào {out}")


if __name__ == "__main__":
    main()
