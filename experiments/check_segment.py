# OWNER: B
"""Kiểm tra tách ký tự CCA trên biển thật và độ khớp của khuôn char_template_mm.

Chạy:
    python experiments/check_segment.py                  # 10 biển tune (2 mỗi subset) + kiểm tra khuôn 5 biển
    python experiments/check_segment.py --per-subset 4 --template-n 10

Output:
    results/figures/segment/cca.png  mỗi hàng: biển xám | nhị phân + hộp ký tự (xanh lá) + khuôn 7 vị trí (đỏ)
    console: số hộp / 7 và position_hits mỗi biển; bảng độ lệch tâm hộp so với tâm khuôn (px và mm).
Crop bằng quad thật (rectify "quad"), binarize theo cfg["binarize"], segment theo method "cca".
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from lpr.binarize import binarize_plate, clean_binary, ensure_text_white
from lpr.config import load_config
from lpr.data.dataset import load_records
from lpr.evaluate import PLATE_WIDTH_MM, count_match, position_hits, template_intervals
from lpr.io_utils import read_image, write_image
from lpr.preprocess import to_gray
from lpr.rectify import rectify
from lpr.segment import segment

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

SUBSETS = ["base", "tilt", "blur", "db", "challenge"]
SCALE = 3


def plate_binary(rec, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Crop quad thật → xám → nhị phân theo config → ký tự trắng → mở bỏ nhiễu."""
    gray = to_gray(rectify(read_image(rec.path), rec.bbox, rec.quad, "quad", cfg))
    bc = cfg["binarize"]
    b = binarize_plate(gray, bc["method"], bc.get(bc["method"], {}))
    return gray, clean_binary(ensure_text_white(b))


def draw_row(gray: np.ndarray, binary: np.ndarray, boxes, cfg: dict, title: str) -> np.ndarray:
    left = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) // 2
    for x0, x1 in template_intervals(cfg):
        cv2.line(right, (int(round(x0)), 1), (int(round(x1)) - 1, 1), (0, 0, 255), 2)
    for b in boxes:
        cv2.rectangle(right, (b.x1, b.y1), (b.x2 - 1, b.y2 - 1), (0, 255, 0), 1)
    row = cv2.resize(np.hstack([left, right]), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    cv2.putText(row, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return row


def check_template(recs, cfg: dict, n: int) -> None:
    """Trên n biển base cho đúng 7 hộp, đo tâm hộp - tâm khuôn theo từng vị trí."""
    tpl = template_intervals(cfg)
    tc = np.array([(a + b) / 2 for a, b in tpl])
    tw = np.array([b - a for a, b in tpl])
    px_to_mm = PLATE_WIDTH_MM / cfg["rectify"]["plate_size"][0]
    offs, widths, used = [], [], []
    for r in recs:
        if r.subset != "base":
            continue
        _, b = plate_binary(r, cfg)
        boxes = segment(b, "cca", cfg)
        if len(boxes) != 7:
            continue
        offs.append([(bx.x1 + bx.x2) / 2 for bx in boxes] - tc)
        widths.append([bx.x2 - bx.x1 for bx in boxes])
        used.append(r.image_id)
        if len(used) == n:
            break
    if not used:
        print("không có biển base nào ra đúng 7 hộp")
        return
    offs, widths = np.array(offs), np.array(widths)
    print(f"\nKiểm tra khuôn trên {len(used)} biển base có đúng 7 hộp: {', '.join(used)}")
    print("lệch = tâm hộp - tâm khuôn (dương = hộp nằm bên phải khuôn); 1 px = "
          f"{px_to_mm:.1f} mm")
    print(f"{'vị trí':<8}{'khuôn x':>14}{'lệch TB px':>12}{'min..max px':>14}{'lệch TB mm':>12}"
          f"{'rộng hộp':>10}{'rộng khuôn':>12}")
    for i in range(7):
        o = offs[:, i]
        print(f"{i:<8}{tpl[i][0]:>6.1f}–{tpl[i][1]:<6.1f}{o.mean():>12.1f}"
              f"{o.min():>7.1f}..{o.max():<5.1f}{o.mean() * px_to_mm:>12.1f}"
              f"{widths[:, i].mean():>10.1f}{tw[i]:>12.1f}")
    print(f"trung bình |lệch| = {np.abs(offs).mean():.1f} px; lệch lớn nhất = {np.abs(offs).max():.1f} px")
    for row, iid in zip(offs, used):
        print(f"  {iid:<12} " + " ".join(f"{v:+5.1f}" for v in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--split", default="tune")
    ap.add_argument("--per-subset", type=int, default=2)
    ap.add_argument("--template-n", type=int, default=5)
    ap.add_argument("--out", default="results/figures/segment/cca.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"binarize={cfg['binarize']['method']}  segment=cca  crop=quad GT  split={args.split}")
    print(f"{'image_id':<16}{'text':<10}{'hộp':>5}{'/7':>4}{'count':>7}{'pos':>6}  hits")
    rows, total_ok, total_pos, n = [], 0, 0, 0
    for s in SUBSETS:
        for r in load_records(cfg, split=args.split, subsets=[s])[: args.per_subset]:
            gray, b = plate_binary(r, cfg)
            boxes = segment(b, "cca", cfg)
            hits = position_hits(boxes, cfg)
            cm = count_match(boxes, r.text or "")
            n += 1
            total_ok += cm
            total_pos += sum(hits)
            print(f"{r.image_id:<16}{r.text or '':<10}{len(boxes):>5}{len(r.text or ''):>4}"
                  f"{'OK' if cm else '-':>7}{sum(hits):>4}/7  {''.join('x' if h else '.' for h in hits)}")
            rows.append(draw_row(gray, b, boxes, cfg, f"{r.image_id}  {len(boxes)}/7  pos {sum(hits)}/7"))
    print(f"tổng: count_match {total_ok}/{n}, vị trí đúng {total_pos}/{7 * n}")

    sep = np.full((6, rows[0].shape[1], 3), 128, np.uint8)
    grid = rows[0]
    for row in rows[1:]:
        grid = np.vstack([grid, sep, row])
    write_image(args.out, grid)
    print(f"đã lưu {args.out}")

    check_template(load_records(cfg, split=args.split, subsets=["base"]), cfg, args.template_n)


if __name__ == "__main__":
    main()
