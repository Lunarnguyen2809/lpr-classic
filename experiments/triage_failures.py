# OWNER: A
"""Triage lỗi theo bước trên tập test (SPEC 2.3): mỗi ảnh gán đúng MỘT bước lỗi chính.

Đọc results/predictions/localization.csv và segmentation.csv (do `run_all.py --exp predict` xuất, E0),
chạy lại định vị trên ảnh test chỉ để lấy quad top-1 (sai số góc) và vẽ lưới debug.

Bước lỗi (loại trừ nhau, theo thứ tự pipeline):
    loc_no_candidate  không ứng viên nào trong top_k đạt IoU >= ngưỡng (kèm nhãn phụ gt_too_small /
                      gt_too_large / other theo area_ratio của bbox thật)
    loc_rank          có ứng viên đúng nhưng không phải top-1
    rectify           top-1 đúng, sai số góc quad > ngưỡng chọn trên tune, và tách ký tự không đúng hết
    segment_count     top-1 đúng, nắn ổn, số hộp ký tự != 7
    segment_position  đủ 7 hộp nhưng có vị trí không trúng khuôn
    ok                top-1 đúng và all_correct

Ngưỡng sai số góc: phân vị 95 của corner_error trên các ảnh tune có top-1 đúng và tách đúng hết
(phạm vi nắn "tốt"); vượt ngưỡng đó mà vẫn lỗi tách thì quy cho bước nắn.

Ví dụ:
    PYTHONPATH=src python experiments/triage_failures.py
    PYTHONPATH=src python experiments/triage_failures.py --segmenter projection   # chỉ bảng, hậu tố _projection
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all  # noqa: E402  (chỉ import hàm đọc lựa chọn E2/E3 và dựng dòng segmentation, không sửa)

from lpr.binarize import binarize_plate, clean_binary, ensure_text_white
from lpr.config import get, load_config
from lpr.data.dataset import load_records
from lpr.evaluate import iou
from lpr.io_utils import ensure_dir, read_image, write_image
from lpr.localize.edge import localize_edge
from lpr.localize.hog_svm import score_candidates
from lpr.preprocess import to_gray
from lpr.rectify import corner_error, quad_from_candidate, rectify
from lpr.segment import segment
from lpr.types import Candidate, PlateRecord
from lpr.visualize import draw_candidates, draw_char_boxes, save_comparison

STAGES = ["loc_no_candidate", "loc_rank", "rectify", "segment_count", "segment_position", "ok"]
GT_STATUSES = ["ok", "segment_count", "segment_position"]
SUBTAGS = ["gt_too_small", "gt_too_large", "other"]
SUBSETS = ["base", "blur", "challenge", "db", "tilt", "ALL"]
CORNER_PCT = 95          # phân vị tune làm ngưỡng sai số góc
MAX_EXAMPLES = 12        # số ảnh tối đa mỗi bước trong results/failures/
IMG_W = 480              # bề rộng ảnh gốc thu nhỏ trong lưới debug


# ---------------------------------------------------------------- hàm thuần (có test)

def classify_segment(count_ok: bool, all_correct: bool) -> str:
    """Trạng thái tách ký tự của một crop: ok | segment_count | segment_position."""
    if all_correct:
        return "ok"
    return "segment_position" if count_ok else "segment_count"


def loc_subtag(area_ratio: float, lo: float, hi: float) -> str:
    """Nhãn phụ cho loc_no_candidate: bbox thật nằm ngoài khoảng area_ratio của bộ lọc ứng viên không?"""
    if area_ratio < lo:
        return "gt_too_small"
    if area_ratio > hi:
        return "gt_too_large"
    return "other"


def classify_stage(any_tp: bool, top1_tp: bool, corner_err: float | None, corner_thr: float,
                   count_ok: bool, all_correct: bool) -> str:
    """Bước lỗi chính của một ảnh (pipeline order, SPEC 2.3).

    any_tp: có ứng viên IoU >= ngưỡng trong top_k. top1_tp: ứng viên hạng 1 đúng. corner_err: sai số góc
    quad top-1 (pixel), None khi không tính được. count_ok, all_correct: kết quả tách ký tự trên crop pred.
    Ảnh top-1 sai không bao giờ là ok, kể cả khi crop sai vị trí vô tình tách đúng.
    """
    if not any_tp:
        return "loc_no_candidate"
    if not top1_tp:
        return "loc_rank"
    if all_correct:
        return "ok"
    if corner_err is not None and corner_err > corner_thr:
        return "rectify"
    return classify_segment(count_ok, all_correct)


# ---------------------------------------------------------------- dựng dữ liệu

def _image_summary(loc: pd.DataFrame) -> pd.DataFrame:
    """Mỗi ảnh: any_tp, top1_tp, best_rank (hạng thấp nhất đúng, 0 nếu không có), n_cands."""
    rows = []
    for iid, g in loc.groupby("image_id", sort=True):
        tp = g[g["is_tp"].astype(bool) & (g["rank"] > 0)]
        rows.append({"image_id": iid, "any_tp": not tp.empty,
                     "top1_tp": bool((tp["rank"] == 1).any()),
                     "best_rank": int(tp["rank"].min()) if not tp.empty else 0,
                     "n_cands": int((g["rank"] > 0).sum())})
    return pd.DataFrame(rows)


def _top1(img: np.ndarray, cfg: dict, method: str, model) -> tuple[list[Candidate], Candidate | None]:
    cands = localize_edge(img, cfg)
    if method == "edge_svm":
        cands = score_candidates(img, cands, model, cfg)
    return cands, (cands[0] if cands else None)


def corner_threshold(cfg: dict, args: argparse.Namespace, method: str, model, crop_m: str, bin_m: str,
                     seg_m: str) -> tuple[float, dict]:
    """Ngưỡng sai số góc từ tune (phân vị CORNER_PCT của ảnh top-1 đúng và tách đúng hết) + thống kê."""
    thr_iou = get(cfg, "evaluate.iou_threshold", 0.5)
    good, bad = [], []
    for r in load_records(cfg, split="tune")[: args.limit]:
        img = read_image(r.path)
        _, top1 = _top1(img, cfg, method, model)
        if top1 is None or iou(top1.bbox, r.bbox) < thr_iou:
            continue
        ce = corner_error(quad_from_candidate(img, top1, cfg["rectify"]["quad_method"]), r.quad)
        rows = run_all.segmentation_rows(r, img, top1, cfg)
        row = next(x for x in rows if (x["crop_source"], x["crop_method"], x["binarize"], x["segmenter"])
                   == ("pred", crop_m, bin_m, seg_m))
        (good if row["all_correct"] else bad).append(ce)
    if not good:
        raise SystemExit("tune không có ảnh top-1 đúng và tách đúng hết, không chọn được ngưỡng sai số góc")
    thr = float(np.percentile(good, CORNER_PCT))
    stats = {"tune_top1_tp_n": len(good) + len(bad), "tune_all_correct_n": len(good),
             "corner_good_median": round(float(np.median(good)), 2),
             "corner_good_p95": round(thr, 2),
             "corner_bad_median": round(float(np.median(bad)), 2) if bad else None}
    return thr, stats


def build_table(cfg: dict, args: argparse.Namespace, method: str, dn: str, model, thr: float,
                crop_m: str, bin_m: str, seg_m: str) -> tuple[pd.DataFrame, dict[str, tuple]]:
    """Bảng từng ảnh test + cache (img_path -> candidates, top1) để vẽ lưới sau."""
    pdir = os.path.join(cfg["paths"]["results"], "predictions")
    loc = pd.read_csv(os.path.join(pdir, "localization.csv"), dtype={"image_id": str})
    seg = pd.read_csv(os.path.join(pdir, "segmentation.csv"), dtype={"image_id": str})
    loc = loc[(loc["method"] == method) & (loc["denoise"] == dn)]
    seg = seg[(seg["crop_method"] == crop_m) & (seg["binarize"] == bin_m) & (seg["segmenter"] == seg_m)]
    recs = {r.image_id: r for r in load_records(cfg, split=args.split)[: args.limit]}
    if set(loc["image_id"]) != set(recs) or set(seg["image_id"]) != set(recs):
        raise SystemExit("predictions không khớp split/limit hiện tại; chạy lại: run_all.py --exp predict")
    summ = _image_summary(loc).set_index("image_id")
    pred = seg[seg["crop_source"] == "pred"].set_index("image_id")
    gt = seg[seg["crop_source"] == "gt"].set_index("image_id")
    lo, hi = get(cfg, "localize.area_ratio_range", [0.008, 0.16])
    thr_iou = get(cfg, "evaluate.iou_threshold", 0.5)
    rows, cache = [], {}
    for n, (iid, r) in enumerate(sorted(recs.items()), 1):
        img = read_image(r.path)
        cands, top1 = _top1(img, cfg, method, model)
        s = summ.loc[iid]
        if bool(s["top1_tp"]) != (top1 is not None and iou(top1.bbox, r.bbox) >= thr_iou):
            raise RuntimeError(f"{iid}: top-1 chạy lại không khớp localization.csv (method/denoise/model đổi?)")
        ce = None
        if s["top1_tp"]:
            ce = corner_error(quad_from_candidate(img, top1, cfg["rectify"]["quad_method"]), r.quad)
        ratio = (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]) / (img.shape[0] * img.shape[1])
        p, g = pred.loc[iid], gt.loc[iid]
        stage = classify_stage(bool(s["any_tp"]), bool(s["top1_tp"]), ce, thr,
                               bool(p["count_match"]), bool(p["all_correct"]))
        rows.append({"image_id": iid, "subset": r.subset, "stage": stage,
                     "subtag": loc_subtag(ratio, lo, hi) if stage == "loc_no_candidate" else "",
                     "gt_status": classify_segment(bool(g["count_match"]), bool(g["all_correct"])),
                     "best_rank": int(s["best_rank"]), "n_cands": int(s["n_cands"]),
                     "corner_error": None if ce is None else round(ce, 2),
                     "n_boxes_pred": int(p["n_boxes"]), "n_boxes_gt": int(g["n_boxes"]),
                     "area_ratio": round(ratio, 4), "tilt_h": r.meta.get("tilt_h"),
                     "tilt_v": r.meta.get("tilt_v"), "brightness": r.meta.get("brightness"),
                     "blur": r.meta.get("blur")})
        cache[iid] = (cands, top1)
        if n % 100 == 0 or n == len(recs):
            print(f"  chạy lại định vị {n}/{len(recs)}")
    return pd.DataFrame(rows), cache


# ---------------------------------------------------------------- bảng

def with_all(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, df.assign(subset="ALL")], ignore_index=True)


def stage_counts(df: pd.DataFrame) -> pd.DataFrame:
    a = with_all(df)
    rows = []
    for sub in SUBSETS:
        g = a[a["subset"] == sub]
        for st in STAGES:
            n = int((g["stage"] == st).sum())
            rows.append({"subset": sub, "stage": st, "n": n, "n_subset": len(g),
                         "pct": round(100 * n / len(g), 1) if len(g) else None})
    return pd.DataFrame(rows)


def cross_table(df: pd.DataFrame) -> pd.DataFrame:
    a = with_all(df)
    rows = []
    for sub in SUBSETS:
        g = a[a["subset"] == sub]
        for gs in GT_STATUSES:
            for st in STAGES:
                rows.append({"subset": sub, "gt_status": gs, "pred_stage": st,
                             "n": int(((g["gt_status"] == gs) & (g["stage"] == st)).sum())})
    return pd.DataFrame(rows)


def subtag_table(df: pd.DataFrame) -> pd.DataFrame:
    a = with_all(df[df["stage"] == "loc_no_candidate"])
    full = with_all(df)
    rows = []
    for sub in SUBSETS:
        for t in SUBTAGS:
            rows.append({"subset": sub, "subtag": t,
                         "n": int(((a["subset"] == sub) & (a["subtag"] == t)).sum()),
                         "n_subset": int((full["subset"] == sub).sum())})
    return pd.DataFrame(rows)


def meta_table(df: pd.DataFrame) -> pd.DataFrame:
    """Trung bình tilt/brightness/blur/area_ratio theo bước lỗi, để so ảnh lỗi với ảnh ok."""
    cols = ["area_ratio", "tilt_h", "tilt_v", "brightness", "blur", "corner_error", "n_boxes_pred"]
    g = df.groupby("stage")[cols].mean().round(3).reset_index()
    g.insert(1, "n", df.groupby("stage").size().reindex(g["stage"]).to_numpy())
    return g


# ---------------------------------------------------------------- ảnh

def _fit_w(img: np.ndarray, w: int) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = w / img.shape[1]
    return cv2.resize(img, (w, max(1, int(round(img.shape[0] * s)))), interpolation=cv2.INTER_AREA)


def _crop_views(gray: np.ndarray, cfg: dict, bin_m: str, seg_m: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Crop xám + hộp ký tự (x2 phóng đại), nhị phân + hộp, số hộp."""
    b = clean_binary(ensure_text_white(binarize_plate(gray, bin_m, cfg["binarize"].get(bin_m, {}))))
    boxes = segment(b, seg_m, cfg)
    up = lambda im: cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)  # noqa: E731
    return up(draw_char_boxes(gray, boxes)), up(draw_char_boxes(b, boxes)), len(boxes)


def _pad_w(img: np.ndarray, w: int) -> np.ndarray:
    if img.shape[1] >= w:
        return img
    out = np.full((img.shape[0], w, 3), 40, np.uint8)
    out[:, :img.shape[1]] = img
    return out


def _strip(images: dict[str, np.ndarray]) -> np.ndarray:
    """Một dải ảnh có tiêu đề, dựng bằng save_comparison (visualize.py) rồi đọc lại."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.png")
        save_comparison(images, path)
        return read_image(path)


def debug_grid(rec: PlateRecord, img: np.ndarray, cands: list[Candidate], top1: Candidate | None,
               cfg: dict, crop_m: str, bin_m: str, seg_m: str, path: str, row: pd.Series) -> None:
    """Ảnh debug một ảnh lỗi, xếp dọc: [ứng viên đỏ (top-1 đậm) + bbox thật xanh | mặt nạ morph],
    [crop pred | nhị phân pred], [crop gt | nhị phân gt]. Hộp xanh là hộp ký tự tách được."""
    _, dbg = localize_edge(img, cfg, return_debug=True)
    strips = [_strip({"cands(red) gt(green)": _fit_w(draw_candidates(img, cands, rec.bbox), IMG_W),
                      "morph mask": _fit_w(dbg.images["morph"], IMG_W)})]
    if top1 is not None:
        quad = quad_from_candidate(img, top1, cfg["rectify"]["quad_method"]) if crop_m == "quad" else None
        p_box, p_bin, p_n = _crop_views(to_gray(rectify(img, top1.bbox, quad, crop_m, cfg)), cfg, bin_m, seg_m)
        strips.append(_strip({f"pred crop n={p_n}": p_box, "pred binary": p_bin}))
    gt_gray = to_gray(rectify(img, rec.bbox, rec.quad, crop_m, cfg))
    g_box, g_bin, g_n = _crop_views(gt_gray, cfg, bin_m, seg_m)
    strips.append(_strip({f"gt crop n={g_n} ({row['gt_status']})": g_box, "gt binary": g_bin}))
    head = np.full((28, 960, 3), 40, np.uint8)
    cv2.putText(head, f"{rec.image_id} [{rec.subset}] stage={row['stage']} best_rank={row['best_rank']} "
                      f"corner={row['corner_error']}", (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    strips.insert(0, head)
    w = max(x.shape[1] for x in strips)
    write_image(path, np.vstack([_pad_w(x, w) for x in strips]))


def save_failure_grids(cfg: dict, args: argparse.Namespace, df: pd.DataFrame, cache: dict,
                       crop_m: str, bin_m: str, seg_m: str) -> None:
    recs = {r.image_id: r for r in load_records(cfg, split=args.split)[: args.limit]}
    rng = np.random.default_rng(cfg.get("seed", 42))
    base = os.path.join(cfg["paths"]["results"], "failures")
    for st in STAGES:
        if st == "ok":
            continue
        ids = sorted(df.loc[df["stage"] == st, "image_id"])
        pick = sorted(rng.choice(ids, size=min(MAX_EXAMPLES, len(ids)), replace=False)) if ids else []
        out = ensure_dir(os.path.join(base, st))
        for iid in pick:
            row = df[df["image_id"] == iid].iloc[0]
            cands, top1 = cache[iid]
            debug_grid(recs[iid], read_image(recs[iid].path), cands, top1, cfg, crop_m, bin_m, seg_m,
                       os.path.join(out, f"{iid}.png"), row)
        print(f"  {st}: {len(ids)} ảnh lỗi, vẽ {len(pick)} lưới -> {out}")


def plot_overview(counts: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"loc_no_candidate": "#e8a0b4", "loc_rank": "#f4c6d2", "rectify": "#c3a6e0",
              "segment_count": "#9ec5f0", "segment_position": "#c9dff7", "ok": "#a8dcc4"}
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bottom = np.zeros(len(SUBSETS))
    for st in STAGES:
        pct = np.array([counts[(counts["subset"] == s) & (counts["stage"] == st)]["pct"].iloc[0] for s in SUBSETS])
        n = [int(counts[(counts["subset"] == s) & (counts["stage"] == st)]["n"].iloc[0]) for s in SUBSETS]
        ax.bar(SUBSETS, pct, bottom=bottom, color=colors[st], label=st, edgecolor="white")
        for i, (p, k) in enumerate(zip(pct, n)):
            if p >= 4:
                ax.text(i, bottom[i] + p / 2, str(k), ha="center", va="center", fontsize=8, color="#333")
        bottom += pct
    ns = [int(counts[counts["subset"] == s]["n_subset"].iloc[0]) for s in SUBSETS]
    ax.set_xticks(range(len(SUBSETS)), [f"{s}\nn={n}" for s, n in zip(SUBSETS, ns)])
    ax.set_ylabel("% ảnh trong subset (số trong ô = số ảnh)")
    ax.set_ylim(0, 100)
    ax.set_title("Triage lỗi theo bước (test, mỗi ảnh một bước lỗi chính)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    ensure_dir(os.path.dirname(path))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["tune", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--crop", default="quad", choices=run_all.CROP_METHODS)
    ap.add_argument("--binarize", default=None, choices=run_all.BINARIZE_METHODS,
                    help="mặc định theo config binarize.method")
    ap.add_argument("--segmenter", default=None, choices=run_all.SEGMENTERS,
                    help="mặc định theo config segment.method; khác mặc định thì chỉ ghi bảng (hậu tố tên)")
    ap.add_argument("--no-images", action="store_true", help="bỏ qua lưới debug và biểu đồ")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bin_m = args.binarize or cfg["binarize"]["method"]
    seg_m = args.segmenter or cfg["segment"]["method"]
    default_cfg = (bin_m, seg_m) == (cfg["binarize"]["method"], cfg["segment"]["method"])
    tag = "" if default_cfg else f"_{bin_m}_{seg_m}"
    method = run_all._chosen(cfg, "e3_localization.csv", "method", run_all.exp_localization, args)
    dn = run_all._chosen(cfg, "e2_filters.csv", "denoise", run_all.exp_filters, args)
    dcfg = load_config(args.config, {"preprocess": {"denoise": dn}})
    model = run_all._load_svm(dcfg) if method == "edge_svm" else None
    print(f"triage split={args.split} localize={method} denoise={dn} crop={args.crop} "
          f"binarize={bin_m} segment={seg_m}")
    thr, tstats = corner_threshold(dcfg, args, method, model, args.crop, bin_m, seg_m)
    print(f"ngưỡng sai số góc = {thr:.2f}px (p{CORNER_PCT} tune), thống kê: {tstats}")
    df, cache = build_table(dcfg, args, method, dn, model, thr, args.crop, bin_m, seg_m)
    tdir = os.path.join(cfg["paths"]["results"], "tables")
    counts = stage_counts(df)
    ensure_dir(tdir)
    outs = {f"triage_stage_counts{tag}.csv": counts, f"triage_gt_vs_pred{tag}.csv": cross_table(df),
            f"triage_loc_subtags{tag}.csv": subtag_table(df), f"triage_stage_meta{tag}.csv": meta_table(df),
            f"triage_per_image{tag}.csv": df,
            f"triage_thresholds{tag}.csv": pd.DataFrame(
                [{"name": "corner_error_thr_px", "value": round(thr, 2),
                  "note": f"p{CORNER_PCT} corner_error tune, top-1 đúng và all_correct"},
                 {"name": "method", "value": method, "note": "E3 chọn trên tune"},
                 {"name": "denoise", "value": dn, "note": "E2 chọn trên tune"},
                 {"name": "crop/binarize/segmenter", "value": f"{args.crop}/{bin_m}/{seg_m}", "note": ""}]
                + [{"name": k, "value": v, "note": "tune"} for k, v in tstats.items()])}
    for name, t in outs.items():
        t.to_csv(os.path.join(tdir, name), index=False, encoding="utf-8")
    print(counts.pivot(index="stage", columns="subset", values="n").reindex(STAGES)[SUBSETS].to_string())
    if args.no_images or tag:
        return
    save_failure_grids(dcfg, args, df, cache, args.crop, bin_m, seg_m)
    fig = os.path.join(cfg["paths"]["results"], "figures", "triage_stages.png")
    plot_overview(counts, fig)
    print(f"đã ghi {fig}")


if __name__ == "__main__":
    main()
