# OWNER: SHARED
"""Lấy mẫu ảnh CCPD theo seed, copy sang data/ccpd_subset và ghi annotations.csv (SPEC mục 5.3, 5.5).

Ví dụ:
    python scripts/build_subset.py --raw data/raw --out data/ccpd_subset --annotations data/annotations.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import random
import shutil

from tqdm import tqdm

from lpr.config import get, load_config
from lpr.data.ccpd import list_subset_files, parse_ccpd_filename
from lpr.io_utils import ensure_dir, write_csv

COLUMNS = ["image_id", "source", "subset", "file", "orig_name", "x1", "y1", "x2", "y2",
           "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y",
           "text", "n_lines", "area", "tilt_h", "tilt_v", "brightness", "blur"]

for _s in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, không in được tiếng Việt
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def valid_files(raw_root: str, subset: str) -> list[tuple[str, dict]]:
    """Trả về (đường dẫn, kết quả parse) của các ảnh có chuỗi biển hợp lệ, sắp theo tên file."""
    out = []
    for path in list_subset_files(raw_root, subset):
        try:
            info = parse_ccpd_filename(path)
        except ValueError:
            continue
        if info["text"] is not None:
            out.append((path, info))
    return out


def build_rows(subset: str, picked: list[tuple[str, dict]], out_dir: str) -> list[dict]:
    rows = []
    sub_dir = ensure_dir(os.path.join(out_dir, subset))
    for i, (src, info) in enumerate(picked):
        image_id = f"{subset}_{i:04d}"
        dst = os.path.join(sub_dir, f"{image_id}.jpg")
        shutil.copy2(src, dst)
        q = info["quad"].astype(int)
        x1, y1, x2, y2 = info["bbox"]
        rows.append({
            "image_id": image_id, "source": "ccpd", "subset": subset,
            "file": os.path.relpath(dst).replace("\\", "/"),
            "orig_name": os.path.basename(src),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "tl_x": q[0, 0], "tl_y": q[0, 1], "tr_x": q[1, 0], "tr_y": q[1, 1],
            "br_x": q[2, 0], "br_y": q[2, 1], "bl_x": q[3, 0], "bl_y": q[3, 1],
            "text": info["text"], "n_lines": 1,
            **{k: info[k] for k in ("area", "tilt_h", "tilt_v", "brightness", "blur")},
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--annotations", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = args.raw or get(cfg, "paths.raw")
    out = args.out or get(cfg, "paths.subset")
    ann = args.annotations or get(cfg, "paths.annotations")
    sizes: dict[str, int] = get(cfg, "data.subsets")
    rng = random.Random(get(cfg, "seed", 42))

    rows: list[dict] = []
    for subset, n in tqdm(sizes.items(), desc="subset"):
        pool = valid_files(raw, subset)
        if len(pool) < n:
            raise SystemExit(f"Subset {subset} chỉ có {len(pool)} ảnh hợp lệ, cần {n}.")
        # Xóa ảnh cũ để không lẫn kết quả lần chạy trước.
        shutil.rmtree(os.path.join(out, subset), ignore_errors=True)
        picked = rng.sample(pool, n)
        rows += build_rows(subset, picked, out)
        print(f"{subset}: {len(pool)} ảnh hợp lệ, lấy {n}")

    write_csv([{c: r[c] for c in COLUMNS} for r in rows], ann)
    print(f"Đã ghi {len(rows)} dòng vào {ann}")


if __name__ == "__main__":
    main()
