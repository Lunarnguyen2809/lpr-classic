# OWNER: B
"""Kiểm tra binarize trên vài ảnh thật (mỗi subset 1 ảnh trong split tune) để so otsu vs adaptive.

Chạy:
    python experiments/check_binarize.py                 # 5 ảnh: base, tilt, blur, db, challenge
    python experiments/check_binarize.py --block 31 --C 5 --open 2 2

Output:
    results/figures/binarize/compare.png   cột: biển màu | xám | otsu | adaptive  (mỗi hàng 1 ảnh)
    bảng in ra console: số blob "giống ký tự" so với số ký tự thật.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from lpr.binarize import binarize_plate, clean_binary, ensure_text_white
from lpr.config import load_config
from lpr.data.dataset import load_records
from lpr.io_utils import read_image, write_image
from lpr.preprocess import to_gray
from lpr.rectify import rectify

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

SUBSETS = ["base", "tilt", "blur", "db", "challenge"]
SCALE = 2  # phóng to ảnh 220×70 cho dễ nhìn


def char_like_blobs(binary: np.ndarray, cfg: dict) -> int:
    """Đếm thành phần liên thông có chiều cao / tỉ lệ giống ký tự (ước lượng thô, chưa có segment)."""
    seg = cfg["segment"]
    h = binary.shape[0]
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cnt = 0
    for x, y, w, bh, area in stats[1:]:
        if area < seg["min_area"]:
            continue
        if not (seg["char_height_range"][0] * h <= bh <= seg["char_height_range"][1] * h):
            continue
        if not (seg["char_aspect_range"][0] <= w / bh <= seg["char_aspect_range"][1]):
            continue
        cnt += 1
    return cnt


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    out = cv2.resize(out, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--split", default="tune")
    ap.add_argument("--per-subset", type=int, default=1)
    ap.add_argument("--block", type=int, default=None, help="ghi đè adaptive.block_size")
    ap.add_argument("--C", type=float, default=None, help="ghi đè adaptive.C")
    ap.add_argument("--open", type=int, nargs=2, default=[2, 2], help="kernel clean_binary (rộng cao)")
    ap.add_argument("--crop", default="quad", choices=["quad", "bbox"])
    ap.add_argument("--out", default="results/figures/binarize/compare.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ad = dict(cfg["binarize"]["adaptive"])
    if args.block is not None:
        ad["block_size"] = args.block
    if args.C is not None:
        ad["C"] = args.C

    recs = []
    for s in SUBSETS:
        recs += load_records(cfg, split=args.split, subsets=[s])[: args.per_subset]

    rows = []
    print(f"adaptive={ad}  open={tuple(args.open)}  crop={args.crop}")
    print(f"{'image_id':<16}{'text':<10}{'n':>3}{'otsu':>6}{'adapt':>7}")
    for r in recs:
        plate = rectify(read_image(r.path), r.bbox, r.quad, args.crop, cfg)
        gray = to_gray(plate)
        outs = {}
        for m, p in (("otsu", {}), ("adaptive", ad)):
            outs[m] = clean_binary(ensure_text_white(binarize_plate(gray, m, p)), tuple(args.open))
        n = len(r.text or "")
        k_o, k_a = char_like_blobs(outs["otsu"], cfg), char_like_blobs(outs["adaptive"], cfg)
        print(f"{r.image_id:<16}{r.text or '':<10}{n:>3}{k_o:>6}{k_a:>7}")
        rows.append(np.hstack([
            label(plate, r.image_id),
            label(gray, "gray"),
            label(outs["otsu"], f"otsu {k_o}/{n}"),
            label(outs["adaptive"], f"adaptive {k_a}/{n}"),
        ]))

    sep = lambda w: np.full((4, w, 3), 128, np.uint8)  # noqa: E731
    grid = rows[0]
    for row in rows[1:]:
        grid = np.vstack([grid, sep(row.shape[1]), row])
    write_image(args.out, grid)
    print(f"đã lưu {args.out}")


if __name__ == "__main__":
    main()
