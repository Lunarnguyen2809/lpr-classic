# OWNER: B
"""Chạy các thí nghiệm E0–E8 (SPEC mục 8.13 và 10).

Ví dụ:
    python experiments/run_all.py --list
    python experiments/run_all.py --exp predict --split test --limit 20
    python experiments/run_all.py --exp all
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import numpy as np
import pandas as pd

from lpr.binarize import binarize_plate, clean_binary, ensure_text_white
from lpr.config import get, load_config
from lpr.data.dataset import load_char_labels, load_records
from lpr.evaluate import count_match, iou, match_detections, position_hits, prf
from lpr.io_utils import read_image, timer, write_csv
from lpr.localize.edge import localize_edge
from lpr.preprocess import to_gray
from lpr.rectify import quad_from_candidate, rectify
from lpr.segment import segment
from lpr.types import Candidate, PlateRecord

for _s in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, không in được tiếng Việt
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# Thứ tự cột theo SPEC 10.1 / 10.2. Ghi theo đúng thứ tự này để Châu Anh đọc bằng pandas không phải đổi.
LOC_COLS = ["image_id", "subset", "method", "denoise", "rank", "x1", "y1", "x2", "y2",
            "score", "iou", "is_tp", "ms"]
SEG_COLS = ["image_id", "subset", "crop_source", "crop_method", "binarize", "segmenter",
            "n_boxes", "n_expected", "count_match", "pos0_hit", "pos1_6_hits", "all_correct", "ms"]

DENOISES = ["none", "gaussian", "median", "bilateral"]
CROP_METHODS = ["bbox", "quad"]
BINARIZE_METHODS = ["otsu", "adaptive"]
SEGMENTERS = ["cca", "projection"]


def localization_rows(rec: PlateRecord, cands: list[Candidate], ms: float, method: str,
                      denoise: str, cfg: dict) -> list[dict]:
    """Một dòng cho mỗi ứng viên trong top_k (rank từ 1).

    Không có ứng viên nào thì vẫn ghi 1 dòng rank = 0, tọa độ trống, is_tp = False, để ảnh
    bị bỏ sót không biến mất khỏi file (đếm FN cần nó).
    """
    thr = get(cfg, "evaluate.iou_threshold", 0.5)
    base = {"image_id": rec.image_id, "subset": rec.subset, "method": method, "denoise": denoise}
    if not cands:
        return [{**base, "rank": 0, "x1": None, "y1": None, "x2": None, "y2": None,
                 "score": None, "iou": 0.0, "is_tp": False, "ms": round(ms, 2)}]
    rows = []
    for rank, c in enumerate(cands[: get(cfg, "localize.top_k", 5)], start=1):
        v = iou(c.bbox, rec.bbox)
        rows.append({**base, "rank": rank, "x1": c.bbox[0], "y1": c.bbox[1], "x2": c.bbox[2],
                     "y2": c.bbox[3], "score": round(float(c.score), 4), "iou": round(v, 4),
                     "is_tp": v >= thr, "ms": round(ms, 2)})
    return rows


def _plate_crops(img: np.ndarray, rec: PlateRecord, top1: Candidate | None,
                 cfg: dict) -> dict[tuple[str, str], np.ndarray | None]:
    """Ảnh xám 220×70 cho từng (crop_source, crop_method). None khi crop pred mà không có ứng viên."""
    crops: dict[tuple[str, str], np.ndarray | None] = {}
    for m in CROP_METHODS:
        crops[("gt", m)] = to_gray(rectify(img, rec.bbox, rec.quad, m, cfg))
        if top1 is None:
            crops[("pred", m)] = None
        else:
            # quad pred lấy theo rectify.quad_method trong config (giống pipeline thật), không dùng quad GT
            quad = quad_from_candidate(img, top1, cfg["rectify"]["quad_method"]) if m == "quad" else None
            crops[("pred", m)] = to_gray(rectify(img, top1.bbox, quad, m, cfg))
    return crops


def segmentation_rows(rec: PlateRecord, img: np.ndarray, top1: Candidate | None,
                      cfg: dict) -> list[dict]:
    """Một dòng cho mỗi tổ hợp crop_source × crop_method × binarize × segmenter (16 dòng / ảnh).

    ms = thời gian binarize + segment (không gồm crop) để so riêng hai bước này ở E7.
    Crop pred khi không có ứng viên: n_boxes = 0, mọi cột đúng/sai = False.
    """
    n_exp = len(rec.text or "")
    rows = []
    for (src, cm), gray in _plate_crops(img, rec, top1, cfg).items():
        for bm in BINARIZE_METHODS:
            for sm in SEGMENTERS:
                base = {"image_id": rec.image_id, "subset": rec.subset, "crop_source": src,
                        "crop_method": cm, "binarize": bm, "segmenter": sm}
                if gray is None:
                    rows.append({**base, "n_boxes": 0, "n_expected": n_exp, "count_match": False,
                                 "pos0_hit": False, "pos1_6_hits": 0, "all_correct": False,
                                 "ms": 0.0})
                    continue
                with timer() as t:
                    b = clean_binary(ensure_text_white(binarize_plate(gray, bm, cfg["binarize"].get(bm, {}))))
                    boxes = segment(b, sm, cfg)
                hits = position_hits(boxes, cfg)
                cm_ok = count_match(boxes, rec.text or "")
                rows.append({**base, "n_boxes": len(boxes), "n_expected": n_exp,
                             "count_match": cm_ok, "pos0_hit": hits[0], "pos1_6_hits": sum(hits[1:]),
                             "all_correct": cm_ok and all(hits), "ms": round(t.ms, 2)})
    return rows


def exp_predict(cfg: dict, args: argparse.Namespace) -> None:
    """E0: chạy pipeline trên split đã chọn, ghi results/predictions/localization.csv và segmentation.csv.

    - localization: method edge × 4 denoise. edge_svm chưa có (hog_svm còn TODO) nên chưa ghi.
    - segmentation: crop_source gt/pred × crop_method bbox/quad × binarize × segmenter.
      Crop pred lấy top-1 của edge với denoise trong config (phương pháp tốt nhất hiện có, SPEC 9.5).
    """
    recs = load_records(cfg, split=args.split)[: args.limit]
    best_dn = cfg["preprocess"]["denoise"]
    out_dir = os.path.join(cfg["paths"]["results"], "predictions")
    loc_rows: list[dict] = []
    seg_rows: list[dict] = []
    dcfgs = {dn: load_config(args.config, {"preprocess": {"denoise": dn}}) for dn in DENOISES}
    print(f"split={args.split} n={len(recs)} denoise={DENOISES} crop pred = edge + {best_dn}")
    for i, r in enumerate(recs, 1):
        img = read_image(r.path)
        top1 = None
        for dn, dcfg in dcfgs.items():
            with timer() as t:
                cands = localize_edge(img, dcfg)
            loc_rows += localization_rows(r, cands, t.ms, "edge", dn, cfg)
            if dn == best_dn:
                top1 = cands[0] if cands else None
        seg_rows += segmentation_rows(r, img, top1, cfg)
        if i % 25 == 0 or i == len(recs):
            print(f"  {i}/{len(recs)}")

    # sắp cột theo schema; write_csv lấy thứ tự khóa của dòng đầu
    write_csv([{k: row[k] for k in LOC_COLS} for row in loc_rows], os.path.join(out_dir, "localization.csv"))
    write_csv([{k: row[k] for k in SEG_COLS} for row in seg_rows], os.path.join(out_dir, "segmentation.csv"))
    print(f"đã ghi {len(loc_rows)} dòng localization.csv, {len(seg_rows)} dòng segmentation.csv vào {out_dir}")


def exp_color(cfg: dict, args: argparse.Namespace) -> None:
    """E1: so sánh độ tách của các kênh màu, ghi tables/e1_color.csv và figures/e1_hist.png."""
    raise NotImplementedError("TODO(B): exp_color")


def exp_filters(cfg: dict, args: argparse.Namespace) -> None:
    """E2: so sánh bộ lọc khử nhiễu với localize_edge (P/R/F1, ms/ảnh), ghi tables/e2_filters.csv."""
    raise NotImplementedError("TODO(B): exp_filters")


def exp_localization(cfg: dict, args: argparse.Namespace) -> None:
    """E3: edge vs edge_svm (P/R/F1, Recall@k theo subset), ghi tables/e3_localization.csv."""
    raise NotImplementedError("TODO(B): exp_localization")


def exp_rectify(cfg: dict, args: argparse.Namespace) -> None:
    """E4: crop bbox vs warp quad (corner_error, count_match, pos acc), ghi tables/e4_rectify.csv."""
    raise NotImplementedError("TODO(B): exp_rectify")


SUBSET_ORDER = ["base", "blur", "challenge", "db", "tilt", "ALL"]
E5_CROP = "quad"   # SPEC 10: E5 cố định crop gt; dùng quad vì E4 / tune cho thấy quad hơn bbox rõ


def _segmentation_df(cfg: dict, args: argparse.Namespace) -> pd.DataFrame:
    """Đọc results/predictions/segmentation.csv; thiếu file hoặc khác tập ảnh của split thì chạy E0 trước.

    E5, E6 chỉ gom số từ file này để mọi bảng dùng chung một lần chạy pipeline.
    """
    path = os.path.join(cfg["paths"]["results"], "predictions", "segmentation.csv")
    want = {r.image_id for r in load_records(cfg, split=args.split)[: args.limit]}
    if os.path.isfile(path):
        df = pd.read_csv(path, dtype={"image_id": str})
        if set(df["image_id"]) == want:
            return df
        print(f"{path} không khớp split={args.split} (limit={args.limit}), chạy lại E0")
    else:
        print(f"chưa có {path}, chạy E0 trước")
    exp_predict(cfg, args)
    return pd.read_csv(path, dtype={"image_id": str})


def summarize_segmentation(df: pd.DataFrame, by: list[str]) -> list[dict]:
    """Gom metric SPEC 9.3 theo `by` × subset, thêm dòng subset = ALL cho mỗi nhóm.

    Output: list dict cột by..., subset, n, count_match, pos0_acc, pos1_6_acc, all_correct, ms_mean.
    """
    rows = []
    for key, g in df.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        parts = [(s, g[g["subset"] == s]) for s in SUBSET_ORDER[:-1]] + [("ALL", g)]
        for sub, gs in parts:
            if gs.empty:
                continue
            n = len(gs)
            rows.append({**dict(zip(by, key)), "subset": sub, "n": n,
                         "count_match": round(gs["count_match"].astype(bool).mean(), 4),
                         "pos0_acc": round(gs["pos0_hit"].astype(bool).mean(), 4),
                         "pos1_6_acc": round(gs["pos1_6_hits"].sum() / (6 * n), 4),
                         "all_correct": round(gs["all_correct"].astype(bool).mean(), 4),
                         "ms_mean": round(gs["ms"].mean(), 3)})
    return rows


def _print_table(rows: list[dict], cols: list[str]) -> None:
    print("  ".join(f"{c:>12}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:>12.3f}" if isinstance(r[c], float) else f"{str(r[c]):>12}" for c in cols))


def char_box_rows(cfg: dict, args: argparse.Namespace, bin_m: str) -> dict[tuple[str, str], dict]:
    """SPEC 9.4: P/R/F1 theo hộp ký tự trên các biển đã gán tay (crop gt + quad, binarize bin_m).

    Hộp gán tay nằm trong tọa độ biển nắn từ quad thật nên chỉ tính cho crop_source = gt.
    Cộng dồn TP/FP/FN của mọi biển rồi mới tính P/R/F1 (micro). Output: {(segmenter, subset): {...}}.
    """
    labels = load_char_labels(cfg)
    recs = [r for r in load_records(cfg, split=args.split)[: args.limit] if r.image_id in labels]
    if not recs:
        return {}
    thr = get(cfg, "evaluate.char_iou_threshold", 0.5)
    acc: dict[tuple[str, str], list[int]] = {}
    for r in recs:
        gray = to_gray(rectify(read_image(r.path), r.bbox, r.quad, "quad", cfg))
        b = clean_binary(ensure_text_white(binarize_plate(gray, bin_m, cfg["binarize"].get(bin_m, {}))))
        for sm in SEGMENTERS:
            pred = segment(b, sm, cfg)
            tp, fp, fn = match_detections([(c.x1, c.y1, c.x2, c.y2) for c in pred],
                                          [(c.x1, c.y1, c.x2, c.y2) for c in labels[r.image_id]], thr)
            for sub in (r.subset, "ALL"):
                a = acc.setdefault((sm, sub), [0, 0, 0, 0])
                a[0] += 1
                a[1] += tp
                a[2] += fp
                a[3] += fn
    out = {}
    for k, (n, tp, fp, fn) in acc.items():
        m = prf(tp, fp, fn)
        out[k] = {"char_n": n, "char_precision": round(m["precision"], 4),
                  "char_recall": round(m["recall"], 4), "char_f1": round(m["f1"], 4)}
    return out


def exp_binarize(cfg: dict, args: argparse.Namespace) -> None:
    """E5: otsu vs adaptive trên crop gt + CCA, ghi tables/e5_binarize.csv.

    Crop gt bằng quad (E5_CROP). Số lấy từ segmentation.csv của E0, chia theo subset.
    """
    df = _segmentation_df(cfg, args)
    df = df[(df["crop_source"] == "gt") & (df["crop_method"] == E5_CROP) & (df["segmenter"] == "cca")]
    rows = summarize_segmentation(df, ["binarize"])
    path = os.path.join(cfg["paths"]["results"], "tables", "e5_binarize.csv")
    write_csv(rows, path)
    print(f"E5 split={args.split}: crop gt + {E5_CROP}, segment cca")
    _print_table(rows, ["binarize", "subset", "n", "count_match", "pos0_acc", "pos1_6_acc", "all_correct"])
    print(f"đã ghi {path}")


def exp_segmentation(cfg: dict, args: argparse.Namespace) -> None:
    """E6: cca vs projection × crop_source gt/pred (metric 9.3 + 9.4), ghi tables/e6_segmentation.csv.

    Binarize cố định theo cfg["binarize"]["method"] (không chọn lại theo kết quả trên test),
    crop quad. Cột char_* (SPEC 9.4) chỉ có ở crop gt và chỉ khi data/char_labels có biển của split.
    """
    bin_m = cfg["binarize"]["method"]
    df = _segmentation_df(cfg, args)
    df = df[(df["binarize"] == bin_m) & (df["crop_method"] == "quad")]
    rows = summarize_segmentation(df, ["segmenter", "crop_source"])
    chars = char_box_rows(cfg, args, bin_m)
    empty = {"char_n": 0, "char_precision": None, "char_recall": None, "char_f1": None}
    for r in rows:
        r.update(chars.get((r["segmenter"], r["subset"]), empty) if r["crop_source"] == "gt" else empty)
    path = os.path.join(cfg["paths"]["results"], "tables", "e6_segmentation.csv")
    write_csv(rows, path)
    print(f"E6 split={args.split}: binarize {bin_m}, crop quad")
    _print_table(rows, ["segmenter", "crop_source", "subset", "n", "count_match", "pos0_acc",
                        "pos1_6_acc", "all_correct"])
    if chars:
        print("SPEC 9.4 (crop gt):")
        for (sm, sub), m in sorted(chars.items()):
            print(f"  {sm:<11}{sub:<10} n={m['char_n']:<3} P={m['char_precision']:.3f} "
                  f"R={m['char_recall']:.3f} F1={m['char_f1']:.3f}")
    else:
        print("SPEC 9.4: chưa có biển nào của split này trong data/char_labels/char_boxes.csv, cột char_* để trống")
    print(f"đã ghi {path}")


def exp_runtime(cfg: dict, args: argparse.Namespace) -> None:
    """E7: thời gian từng bước (ms/ảnh trung bình, p95), ghi tables/e7_runtime.csv."""
    raise NotImplementedError("TODO(B): exp_runtime")


def exp_vn_qualitative(cfg: dict, args: argparse.Namespace) -> None:
    """E8: chạy pipeline tốt nhất trên ảnh VN (split qual), lưu ảnh minh họa vào figures/vn/."""
    raise NotImplementedError("TODO(B): exp_vn_qualitative")


EXPERIMENTS: dict[str, Callable[[dict, argparse.Namespace], None]] = {
    "predict": exp_predict,
    "color": exp_color,
    "filters": exp_filters,
    "localization": exp_localization,
    "rectify": exp_rectify,
    "binarize": exp_binarize,
    "segmentation": exp_segmentation,
    "runtime": exp_runtime,
    "vn_qualitative": exp_vn_qualitative,
}


def main() -> None:
    """CLI: --exp <tên>|all [--split test] [--limit N] [--config ...] [--list]."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=None, help="tên thí nghiệm hoặc 'all'")
    ap.add_argument("--split", default="test", choices=["tune", "test", "qual"])
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N ảnh đầu để thử nhanh")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--list", action="store_true", help="in danh sách thí nghiệm")
    args = ap.parse_args()

    if args.list:
        for i, (name, fn) in enumerate(EXPERIMENTS.items()):
            doc = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"E{i}  {name:<15} {doc}")
        return
    if not args.exp:
        ap.error("cần --exp <tên>|all hoặc --list")
    names = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        ap.error(f"không có thí nghiệm {unknown}; xem --list")

    cfg = load_config(args.config)
    for name in names:
        print(f"== {name} ==")
        EXPERIMENTS[name](cfg, args)


if __name__ == "__main__":
    main()
