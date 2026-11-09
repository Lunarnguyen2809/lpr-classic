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
from lpr.localize.hog_svm import load_model, score_candidates
from lpr.preprocess import preprocess, to_gray
from lpr.rectify import corner_error, quad_from_candidate, rectify
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


COLOR_CHANNELS = ["gray", "H", "S", "V", "L", "a", "b"]
COLOR_BINS = 64


def color_table(sep: pd.DataFrame) -> list[dict]:
    """Gom độ tách theo kênh × subset (+ ALL), xếp hạng trong từng subset (1 = tách tốt nhất).

    Input: DataFrame cột image_id, subset, channel, separability.
    """
    rows = []
    for sub in SUBSET_ORDER:
        g = sep if sub == "ALL" else sep[sep["subset"] == sub]
        if g.empty:
            continue
        agg = g.groupby("channel")["separability"].agg(["count", "mean", "std"])
        agg = agg.sort_values("mean", ascending=False)
        for rank, (ch, a) in enumerate(agg.iterrows(), 1):
            rows.append({"channel": ch, "subset": sub, "n": int(a["count"]),
                         "mean_separability": round(a["mean"], 4),
                         "std": round(0.0 if pd.isna(a["std"]) else a["std"], 4), "rank": rank})
    return rows


def exp_color(cfg: dict, args: argparse.Namespace) -> None:
    """E1: so sánh độ tách của các kênh màu, ghi tables/e1_color.csv và figures/e1_hist.png.

    Luôn chạy trên split tune: E1 dùng để chọn kênh cho tiền xử lý, chọn trên test là rò dữ liệu.
    Độ tách = khoảng cách Bhattacharyya giữa histogram trong bbox biển và phần ảnh còn lại.
    Hình: histogram trung bình trong / ngoài biển của từng kênh.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lpr.color_analysis import CHANNEL_RANGE, DEFAULT_RANGE, channel_histograms, separability

    if args.split != "tune":
        print(f"E1 bỏ qua --split {args.split}, luôn dùng tune")
    recs = load_records(cfg, split="tune")[: args.limit]
    sums = {ch: [np.zeros(COLOR_BINS), np.zeros(COLOR_BINS)] for ch in COLOR_CHANNELS}
    rows = []
    for r in recs:
        for ch, (h_in, h_out) in channel_histograms(read_image(r.path), r.bbox, COLOR_CHANNELS, COLOR_BINS).items():
            sums[ch][0] += h_in
            sums[ch][1] += h_out
            rows.append({"image_id": r.image_id, "subset": r.subset, "channel": ch,
                         "separability": separability(h_in, h_out)})
    table = color_table(pd.DataFrame(rows))
    tpath = os.path.join(cfg["paths"]["results"], "tables", "e1_color.csv")
    write_csv(table, tpath)

    mean_all = {t["channel"]: t["mean_separability"] for t in table if t["subset"] == "ALL"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, ch in zip(axes.flat, sorted(COLOR_CHANNELS, key=lambda c: -mean_all[c])):
        lo, hi = CHANNEL_RANGE.get(ch, DEFAULT_RANGE)
        x = np.linspace(lo, hi, COLOR_BINS, endpoint=False) + (hi - lo) / COLOR_BINS / 2
        ax.plot(x, sums[ch][0] / len(recs), label="trong biển", color="tab:blue")
        ax.plot(x, sums[ch][1] / len(recs), label="ngoài biển", color="tab:gray")
        ax.set_title(f"{ch}  (D = {mean_all[ch]:.3f})")
        ax.set_xlim(lo, hi)
    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    fig.suptitle(f"E1: histogram trung bình trong / ngoài biển, tune n={len(recs)}, xếp theo độ tách giảm dần")
    fig.tight_layout()
    fpath = os.path.join(cfg["paths"]["results"], "figures", "e1_hist.png")
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    fig.savefig(fpath, dpi=110)
    plt.close(fig)

    print(f"E1 split=tune n={len(recs)}, bins={COLOR_BINS}")
    _print_table([t for t in table if t["subset"] == "ALL"], ["rank", "channel", "mean_separability", "std"])
    print(f"đã ghi {tpath} và {fpath}")


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


LOC_KS = (1, 3, 5)   # Recall@k báo trong E2/E3 (top_k trong config = 5)


def summarize_localization(df: pd.DataFrame, by: list[str]) -> list[dict]:
    """Gom metric SPEC 9.1 theo `by` × subset (+ ALL). Input: dòng kiểu `localization_rows`.

    Mỗi ảnh = 1 mẫu: top-1 là dòng rank 1 (rank 0 = không có ứng viên). TP nếu top-1 đúng, FP nếu có
    top-1 mà sai, FN nếu không có TP (kể cả không có ứng viên) nên TP + FN = n. Recall@k = có ít nhất
    một dòng rank 1..k đúng. ms = thời gian cả ảnh (lặp ở mọi rank nên lấy dòng đầu của ảnh).
    Output: list dict cột by..., subset, n, tp, fp, fn, precision, recall, f1, recall_at_1/3/5, ms_mean.
    """
    img_rows = []
    for key, g in df.groupby(by + ["image_id"], sort=True):
        hit = g[g["rank"] >= 1]
        tp_ranks = hit.loc[hit["is_tp"].astype(bool), "rank"]
        first_tp = int(tp_ranks.min()) if len(tp_ranks) else 0
        img_rows.append({**dict(zip(by, key[:-1])), "subset": g["subset"].iloc[0],
                         "has_top1": bool((g["rank"] == 1).any()), "top1_ok": first_tp == 1,
                         "first_tp": first_tp, "ms": float(g["ms"].iloc[0])})
    per = pd.DataFrame(img_rows)
    rows = []
    for key, g in per.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        parts = [(s, g[g["subset"] == s]) for s in SUBSET_ORDER[:-1]] + [("ALL", g)]
        for sub, gs in parts:
            if gs.empty:
                continue
            tp = int(gs["top1_ok"].sum())
            fp = int((gs["has_top1"] & ~gs["top1_ok"]).sum())
            m = prf(tp, fp, len(gs) - tp)
            row = {**dict(zip(by, key)), "subset": sub, "n": len(gs), "tp": tp, "fp": fp,
                   "fn": len(gs) - tp, "precision": round(m["precision"], 4),
                   "recall": round(m["recall"], 4), "f1": round(m["f1"], 4)}
            for k in LOC_KS:
                row[f"recall_at_{k}"] = round(float(((gs["first_tp"] >= 1) & (gs["first_tp"] <= k)).mean()), 4)
            row["ms_mean"] = round(gs["ms"].mean(), 2)
            rows.append(row)
    return rows


def pick_best(rows: list[dict], col: str, split: str = "tune") -> str:
    """Giá trị `col` có F1 cao nhất ở dòng subset = ALL của `split` (hòa thì chọn ms_mean thấp hơn).

    Chọn tham số CHỈ trên tune (SPEC 10: không chọn trên test).
    """
    cand = [r for r in rows if r["subset"] == "ALL" and r["split"] == split]
    if not cand:
        raise ValueError(f"không có dòng ALL của split {split} để chọn {col}")
    return min(cand, key=lambda r: (-r["f1"], r["ms_mean"]))[col]


def _splits(args: argparse.Namespace) -> list[str]:
    """Tune (để chọn) rồi tới split báo cáo; trùng thì chỉ một."""
    return ["tune"] if args.split == "tune" else ["tune", args.split]


def _chosen(cfg: dict, name: str, col: str, runner, args: argparse.Namespace) -> str:
    """Đọc giá trị đã chọn (cột `chosen` = True) từ bảng E2/E3; thiếu file thì chạy `runner` trước."""
    path = os.path.join(cfg["paths"]["results"], "tables", name)
    if not os.path.isfile(path):
        print(f"chưa có {path}, chạy thí nghiệm tạo nó trước")
        runner(cfg, args)
    df = pd.read_csv(path)
    return str(df.loc[df["chosen"].astype(bool), col].iloc[0])


def _load_svm(cfg: dict):
    path = os.path.join(cfg["paths"]["models"], "svm.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"thiếu {path}; chạy: PYTHONPATH=src python experiments/train_hog_svm.py")
    return load_model(path)


def exp_filters(cfg: dict, args: argparse.Namespace) -> None:
    """E2: so sánh bộ lọc khử nhiễu với localize_edge (P/R/F1, ms/ảnh), ghi tables/e2_filters.csv.

    Chạy trên tune (để chọn) và split báo cáo (cột `split`). Cột `chosen` = True ở denoise có F1 ALL
    cao nhất trên tune; E3 dùng giá trị này.
    """
    dcfgs = {dn: load_config(args.config, {"preprocess": {"denoise": dn}}) for dn in DENOISES}
    table: list[dict] = []
    for split in _splits(args):
        recs = load_records(cfg, split=split)[: args.limit]
        print(f"E2 split={split} n={len(recs)} denoise={DENOISES}")
        rows: list[dict] = []
        for i, r in enumerate(recs, 1):
            img = read_image(r.path)
            for dn, dcfg in dcfgs.items():
                with timer() as t:
                    cands = localize_edge(img, dcfg)
                rows += localization_rows(r, cands, t.ms, "edge", dn, cfg)
            if i % 50 == 0 or i == len(recs):
                print(f"  {i}/{len(recs)}")
        table += [{"split": split, **x} for x in summarize_localization(pd.DataFrame(rows), ["method", "denoise"])]
    best = pick_best(table, "denoise")
    for x in table:
        x["chosen"] = x["denoise"] == best
    path = os.path.join(cfg["paths"]["results"], "tables", "e2_filters.csv")
    write_csv(table, path)
    print(f"E2 chọn denoise = {best} (F1 cao nhất trên tune)")
    cols = ["split", "denoise", "n", "precision", "recall", "f1", "recall_at_5", "ms_mean"]
    _print_table([x for x in table if x["subset"] == "ALL"], cols)
    print(f"đã ghi {path}")


def exp_localization(cfg: dict, args: argparse.Namespace) -> None:
    """E3: edge vs edge_svm (P/R/F1, Recall@k theo subset), ghi tables/e3_localization.csv.

    Denoise = giá trị E2 chọn trên tune. SVM chấm lại đúng danh sách ứng viên của edge, top-1 không lọc
    theo score_threshold (giữ Recall@k). ms edge_svm = localize_edge + chấm SVM. Chạy trên tune và split
    báo cáo; cột `chosen` = True ở method có F1 ALL cao nhất trên tune (E4 dùng).
    """
    dn = _chosen(cfg, "e2_filters.csv", "denoise", exp_filters, args)
    dcfg = load_config(args.config, {"preprocess": {"denoise": dn}})
    model = _load_svm(dcfg)
    table: list[dict] = []
    for split in _splits(args):
        recs = load_records(cfg, split=split)[: args.limit]
        print(f"E3 split={split} n={len(recs)} denoise={dn}")
        rows: list[dict] = []
        for i, r in enumerate(recs, 1):
            img = read_image(r.path)
            with timer() as t_edge:
                cands = localize_edge(img, dcfg)
            with timer() as t_svm:
                svm = score_candidates(img, cands, model, dcfg)
            rows += localization_rows(r, cands, t_edge.ms, "edge", dn, cfg)
            rows += localization_rows(r, svm, t_edge.ms + t_svm.ms, "edge_svm", dn, cfg)
            if i % 50 == 0 or i == len(recs):
                print(f"  {i}/{len(recs)}")
        table += [{"split": split, **x} for x in summarize_localization(pd.DataFrame(rows), ["method"])]
    best = pick_best(table, "method")
    for x in table:
        x["denoise"] = dn
        x["chosen"] = x["method"] == best
    path = os.path.join(cfg["paths"]["results"], "tables", "e3_localization.csv")
    write_csv(table, path)
    print(f"E3 chọn method = {best} (F1 cao nhất trên tune)")
    cols = ["split", "method", "subset", "n", "precision", "recall", "f1", "recall_at_1", "recall_at_3",
            "recall_at_5", "ms_mean"]
    _print_table(table, cols)
    print(f"đã ghi {path}")


def rectify_image_rows(seg_rows: list[dict], corner: dict[str, float | None], loc_ok: bool) -> list[dict]:
    """Dòng E4 của một ảnh từ `segmentation_rows`: giữ binarize = otsu, segmenter = cca (SPEC 10 cố định).

    corner = {crop_method: corner_error hoặc None}, chỉ gắn cho crop_source = pred (crop gt dùng quad
    thật nên không có sai số góc). loc_ok = top-1 định vị đúng (IoU ≥ ngưỡng): khi True, mỗi dòng pred
    được nhân thêm một dòng crop_source = pred_tp, để so bbox/quad trên đúng các ảnh đã định vị được.
    """
    out = []
    for r in seg_rows:
        if r["binarize"] != "otsu" or r["segmenter"] != "cca":
            continue
        row = {k: r[k] for k in ("image_id", "subset", "crop_source", "crop_method", "n_boxes",
                                 "count_match", "pos0_hit", "pos1_6_hits", "all_correct", "ms")}
        row["corner_error"] = corner.get(r["crop_method"]) if r["crop_source"] == "pred" else None
        out.append(row)
        if r["crop_source"] == "pred" and loc_ok:
            out.append({**row, "crop_source": "pred_tp"})
    return out


def rectify_table(df: pd.DataFrame) -> list[dict]:
    """Gom E4 theo crop_source × crop_method × subset (+ ALL): metric 9.3 + corner_error (9.2).

    corner_n / corner_mean / corner_median tính trên các dòng có corner_error (ảnh định vị đúng, có quad).
    """
    rows = summarize_segmentation(df, ["crop_source", "crop_method"])
    for r in rows:
        g = df[(df["crop_source"] == r["crop_source"]) & (df["crop_method"] == r["crop_method"])]
        if r["subset"] != "ALL":
            g = g[g["subset"] == r["subset"]]
        ce = g["corner_error"].dropna()
        r["corner_n"] = len(ce)
        r["corner_mean"] = round(float(ce.mean()), 2) if len(ce) else None
        r["corner_median"] = round(float(ce.median()), 2) if len(ce) else None
    return rows


def exp_rectify(cfg: dict, args: argparse.Namespace) -> None:
    """E4: crop bbox vs warp quad (corner_error, count_match, pos acc), ghi tables/e4_rectify.csv.

    Binarize Otsu + CCA cố định. Ứng viên top-1 = method E3 chọn trên tune (denoise của E2). crop_source:
    gt (crop từ bbox/quad thật), pred (top-1, mọi ảnh, định vị sai tính là hỏng), pred_tp (chỉ ảnh top-1
    đúng IoU ≥ 0.5). corner_error (pixel ảnh gốc, SPEC 9.2): quad pred = quad_from_candidate; với crop
    bbox, quad ngầm là 4 góc bbox pred. Chỉ tính trên ảnh định vị đúng.
    """
    method = _chosen(cfg, "e3_localization.csv", "method", exp_localization, args)
    dn = _chosen(cfg, "e2_filters.csv", "denoise", exp_filters, args)
    dcfg = load_config(args.config, {"preprocess": {"denoise": dn}})
    model = _load_svm(dcfg) if method == "edge_svm" else None
    thr = get(cfg, "evaluate.iou_threshold", 0.5)
    recs = load_records(cfg, split=args.split)[: args.limit]
    print(f"E4 split={args.split} n={len(recs)} localize={method} denoise={dn} binarize=otsu segment=cca")
    rows: list[dict] = []
    for i, r in enumerate(recs, 1):
        img = read_image(r.path)
        cands = localize_edge(img, dcfg)
        if model is not None:
            cands = score_candidates(img, cands, model, dcfg)
        top1 = cands[0] if cands else None
        loc_ok = top1 is not None and iou(top1.bbox, r.bbox) >= thr
        corner: dict[str, float | None] = {}
        if loc_ok:
            x1, y1, x2, y2 = top1.bbox
            corner["bbox"] = corner_error(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32), r.quad)
            corner["quad"] = corner_error(quad_from_candidate(img, top1, dcfg["rectify"]["quad_method"]), r.quad)
        rows += rectify_image_rows(segmentation_rows(r, img, top1, dcfg), corner, loc_ok)
        if i % 50 == 0 or i == len(recs):
            print(f"  {i}/{len(recs)}")
    table = rectify_table(pd.DataFrame(rows))
    path = os.path.join(cfg["paths"]["results"], "tables", "e4_rectify.csv")
    write_csv(table, path)
    _print_table(table, ["crop_source", "crop_method", "subset", "n", "corner_n", "corner_mean",
                         "corner_median", "count_match", "pos0_acc", "pos1_6_acc", "all_correct"])
    print(f"đã ghi {path}")


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


RUNTIME_STEPS = ["preprocess", "localize_other", "hog_svm", "quad", "rectify", "binarize", "segment",
                 "total_edge", "total"]


def runtime_table(df: pd.DataFrame) -> list[dict]:
    """Gom thời gian theo bước × subset (+ ALL). Input: cột image_id, subset, step, ms.

    Output: step, subset, n (số ảnh có bước đó), ms_mean, ms_p95. Bước theo thứ tự RUNTIME_STEPS.
    """
    rows = []
    for step in RUNTIME_STEPS:
        d = df[df["step"] == step]
        for sub in SUBSET_ORDER:
            g = d if sub == "ALL" else d[d["subset"] == sub]
            if g.empty:
                continue
            rows.append({"step": step, "subset": sub, "n": len(g), "ms_mean": round(float(g["ms"].mean()), 2),
                         "ms_p95": round(float(np.percentile(g["ms"], 95)), 2)})
    return rows


def _runtime_one(img: np.ndarray, cfg: dict, model) -> dict[str, float]:
    """Thời gian (ms) từng bước pipeline edge_svm trên một ảnh. Bước không chạy được thì vắng khỏi dict.

    localize_other = localize_edge toàn phần trừ tiền xử lý (Sobel, nhị phân cạnh, morph, lọc ứng viên, NMS).
    total_edge = tổng không có hog_svm; total = tổng cả pipeline edge_svm.
    """
    ms: dict[str, float] = {}
    with timer() as t:
        preprocess(img, cfg)
    ms["preprocess"] = t.ms
    with timer() as t:
        cands = localize_edge(img, cfg)
    ms["localize_other"] = max(0.0, t.ms - ms["preprocess"])
    with timer() as t:
        cands = score_candidates(img, cands, model, cfg)
    ms["hog_svm"] = t.ms
    if cands:
        top1 = cands[0]
        with timer() as t:
            quad = quad_from_candidate(img, top1, cfg["rectify"]["quad_method"])
        ms["quad"] = t.ms
        with timer() as t:
            gray = to_gray(rectify(img, top1.bbox, quad, "quad", cfg))
        ms["rectify"] = t.ms
        bm = cfg["binarize"]["method"]
        with timer() as t:
            b = clean_binary(ensure_text_white(binarize_plate(gray, bm, cfg["binarize"].get(bm, {}))))
        ms["binarize"] = t.ms
        with timer() as t:
            segment(b, cfg["segment"]["method"], cfg)
        ms["segment"] = t.ms
    ms["total_edge"] = sum(v for k, v in ms.items() if k != "hog_svm")
    ms["total"] = ms["total_edge"] + ms["hog_svm"]
    return ms


def exp_runtime(cfg: dict, args: argparse.Namespace) -> None:
    """E7: thời gian từng bước (ms/ảnh trung bình, p95), ghi tables/e7_runtime.csv.

    Pipeline edge_svm đầy đủ theo config (denoise, binarize, segment lấy từ cfg), crop quad từ top-1.
    Ảnh đầu chạy một lượt khởi động không tính. Chạy tuần tự một tiến trình; số tuyệt đối phụ thuộc máy.
    """
    model = _load_svm(cfg)
    recs = load_records(cfg, split=args.split)[: args.limit]
    print(f"E7 split={args.split} n={len(recs)} denoise={cfg['preprocess']['denoise']}")
    rows: list[dict] = []
    for i, r in enumerate(recs, 1):
        img = read_image(r.path)
        if i == 1:
            _runtime_one(img, cfg, model)   # khởi động (nạp thư viện, cache), bỏ
        for step, v in _runtime_one(img, cfg, model).items():
            rows.append({"image_id": r.image_id, "subset": r.subset, "step": step, "ms": v})
        if i % 100 == 0 or i == len(recs):
            print(f"  {i}/{len(recs)}")
    table = runtime_table(pd.DataFrame(rows))
    path = os.path.join(cfg["paths"]["results"], "tables", "e7_runtime.csv")
    write_csv(table, path)
    _print_table([t for t in table if t["subset"] == "ALL"], ["step", "n", "ms_mean", "ms_p95"])
    print(f"đã ghi {path}")


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
