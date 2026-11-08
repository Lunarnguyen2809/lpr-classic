# OWNER: B
"""So sánh binarize / segment / crop trên TẬP TUNE (SPEC 9.3), mỗi lần chỉ đổi một yếu tố.

    1. binarize otsu vs adaptive      (cố định segment cca, crop quad thật)
    2. segment  cca  vs projection    (cố định binarize thắng ở 1, crop quad thật)
    3. crop     bbox vs quad          (cố định binarize, segment thắng)

"Thắng" = all_correct cao hơn trên toàn tập (hòa thì xét count_match).

Chạy:
    python experiments/check_segment_all.py
    python experiments/check_segment_all.py --limit 40     # thử nhanh

Output:
    console: bảng count_match, pos0_acc, pos1_6_acc, all_correct theo subset, và tỉ lệ đúng vị trí số "1"
    results/tables/check_segment_all.csv
    results/figures/segment/compare_{binarize,segment,crop}.png : ca bên thắng đúng hết, bên thua sai (và ngược lại)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import cv2
import numpy as np

from lpr.binarize import binarize_plate, clean_binary, ensure_text_white
from lpr.config import load_config
from lpr.data.dataset import load_records
from lpr.evaluate import count_match, position_hits, template_intervals
from lpr.io_utils import read_image, write_csv, write_image
from lpr.preprocess import to_gray
from lpr.rectify import rectify
from lpr.segment import segment

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

SUBSETS = ["base", "blur", "challenge", "db", "tilt"]


def binarize(gray: np.ndarray, method: str, cfg: dict) -> np.ndarray:
    b = binarize_plate(gray, method, cfg["binarize"].get(method, {}))
    return clean_binary(ensure_text_white(b))


def run(recs, grays, bin_m: str, seg_m: str, crop: str, cfg: dict) -> list[dict]:
    """Chạy một tổ hợp trên mọi biển, trả về list kết quả từng biển."""
    out = []
    for r in recs:
        b = binarize(grays[crop][r.image_id], bin_m, cfg)
        boxes = segment(b, seg_m, cfg)
        hits = position_hits(boxes, cfg)
        cm = count_match(boxes, r.text)
        out.append({"rec": r, "binary": b, "boxes": boxes, "hits": hits, "count_match": cm,
                    "all_correct": cm and all(hits)})
    return out


def summarize(res: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for x in res:
        by[x["rec"].subset].append(x)
        by["ALL"].append(x)
    table = {}
    for sub, xs in by.items():
        n = len(xs)
        ones = [x["hits"][i] for x in xs for i, ch in enumerate(x["rec"].text) if ch == "1"]
        table[sub] = {
            "n": n,
            "count_match": sum(x["count_match"] for x in xs) / n,
            "pos0_acc": sum(x["hits"][0] for x in xs) / n,
            "pos1_6_acc": sum(sum(x["hits"][1:]) for x in xs) / (6 * n),
            "all_correct": sum(x["all_correct"] for x in xs) / n,
            "pos_digit1": (sum(ones) / len(ones)) if ones else float("nan"),
            "n_digit1": len(ones),
        }
    return table


def print_compare(title: str, tables: dict[str, dict]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'cấu hình':<26}{'subset':<11}{'n':>4}{'count':>8}{'pos0':>7}{'pos1_6':>8}{'all':>7}"
          f"{'pos số 1':>10}")
    for name, tab in tables.items():
        for sub in SUBSETS + ["ALL"]:
            if sub not in tab:
                continue
            t = tab[sub]
            d1 = f"{t['pos_digit1']:.2f} ({t['n_digit1']})" if t["n_digit1"] else "-"
            print(f"{name:<26}{sub:<11}{t['n']:>4}{t['count_match']:>8.3f}{t['pos0_acc']:>7.3f}"
                  f"{t['pos1_6_acc']:>8.3f}{t['all_correct']:>7.3f}{d1:>10}")
        print()


def winner(tables: dict[str, dict]) -> str:
    return max(tables, key=lambda k: (tables[k]["ALL"]["all_correct"], tables[k]["ALL"]["count_match"]))


def draw(x: dict, cfg: dict, label: str) -> np.ndarray:
    img = cv2.cvtColor(x["binary"], cv2.COLOR_GRAY2BGR) // 2
    for x0, x1 in template_intervals(cfg):
        cv2.line(img, (int(round(x0)), 1), (int(round(x1)) - 1, 1), (0, 0, 255), 2)
    for b in x["boxes"]:
        cv2.rectangle(img, (b.x1, b.y1), (b.x2 - 1, b.y2 - 1), (0, 255, 0), 1)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    txt = f"{label}: {len(x['boxes'])}/7 pos {sum(x['hits'])}/7"
    cv2.putText(img, txt, (6, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def save_examples(path: str, res_a, res_b, name_a: str, name_b: str, grays, crop_a: str, crop_b: str,
                  cfg: dict, k: int = 4) -> None:
    """Ghép ảnh: k ca A đúng hết mà B sai, rồi tối đa 2 ca ngược lại. Mỗi hàng: xám A | A | B."""
    a_wins = [i for i, (a, b) in enumerate(zip(res_a, res_b)) if a["all_correct"] and not b["all_correct"]]
    b_wins = [i for i, (a, b) in enumerate(zip(res_a, res_b)) if b["all_correct"] and not a["all_correct"]]
    rows = []
    for i in a_wins[:k] + b_wins[:2]:
        r = res_a[i]["rec"]
        g = cv2.resize(cv2.cvtColor(grays[crop_a][r.image_id], cv2.COLOR_GRAY2BGR), None, fx=3, fy=3)
        cv2.putText(g, r.image_id, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        rows.append(np.hstack([g, draw(res_a[i], cfg, name_a), draw(res_b[i], cfg, name_b)]))
    print(f"{path}: {name_a} thắng riêng {len(a_wins)} biển, {name_b} thắng riêng {len(b_wins)} biển")
    if not rows:
        return
    sep = np.full((6, rows[0].shape[1], 3), 128, np.uint8)
    grid = rows[0]
    for row in rows[1:]:
        grid = np.vstack([grid, sep, row])
    write_image(path, grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    recs = load_records(cfg, split="tune")[: args.limit]   # chỉ tune, không đụng test
    grays = {c: {} for c in ("quad", "bbox")}
    for r in recs:
        img = read_image(r.path)
        for c in grays:
            grays[c][r.image_id] = to_gray(rectify(img, r.bbox, r.quad, c, cfg))
    print(f"split=tune  n={len(recs)}")

    csv_rows: list[dict] = []

    def record(exp: str, name: str, tab: dict) -> None:
        for sub, t in tab.items():
            csv_rows.append({"experiment": exp, "config": name, "subset": sub, **t})

    # 1. binarize
    res1 = {m: run(recs, grays, m, "cca", "quad", cfg) for m in ("otsu", "adaptive")}
    tab1 = {f"{m}+cca+quad": summarize(r) for m, r in res1.items()}
    print_compare("1. binarize (cca, quad)", tab1)
    best_bin = winner(tab1).split("+")[0]
    for k, t in tab1.items():
        record("binarize", k, t)

    # 2. segment
    res2 = {s: run(recs, grays, best_bin, s, "quad", cfg) for s in ("cca", "projection")}
    tab2 = {f"{best_bin}+{s}+quad": summarize(r) for s, r in res2.items()}
    print_compare(f"2. segment ({best_bin}, quad)", tab2)
    best_seg = winner(tab2).split("+")[1]
    for k, t in tab2.items():
        record("segment", k, t)

    # 3. crop
    res3 = {c: run(recs, grays, best_bin, best_seg, c, cfg) for c in ("bbox", "quad")}
    tab3 = {f"{best_bin}+{best_seg}+{c}": summarize(r) for c, r in res3.items()}
    print_compare(f"3. crop ({best_bin}, {best_seg})", tab3)
    best_crop = winner(tab3).split("+")[2]
    for k, t in tab3.items():
        record("crop", k, t)

    print(f"Tốt nhất trên tune: binarize={best_bin}, segment={best_seg}, crop={best_crop}")
    write_csv(csv_rows, "results/tables/check_segment_all.csv")
    print("đã lưu results/tables/check_segment_all.csv")

    other_bin = "adaptive" if best_bin == "otsu" else "otsu"
    save_examples("results/figures/segment/compare_binarize.png", res1[best_bin], res1[other_bin],
                  best_bin, other_bin, grays, "quad", "quad", cfg)
    other_seg = "projection" if best_seg == "cca" else "cca"
    save_examples("results/figures/segment/compare_segment.png", res2[best_seg], res2[other_seg],
                  best_seg, other_seg, grays, "quad", "quad", cfg)
    other_crop = "bbox" if best_crop == "quad" else "quad"
    save_examples("results/figures/segment/compare_crop.png", res3[best_crop], res3[other_crop],
                  best_crop, other_crop, grays, best_crop, other_crop, cfg)


if __name__ == "__main__":
    main()
