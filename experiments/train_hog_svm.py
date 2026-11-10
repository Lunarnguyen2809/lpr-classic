"""Huấn luyện SVM cho localize/hog_svm trên tập tune, lưu model, đo top-1 edge vs edge_svm.

Chạy (từ gốc repo):
    PYTHONPATH=src python experiments/train_hog_svm.py                  # train + CV + lưu models/svm.joblib
    PYTHONPATH=src python experiments/train_hog_svm.py --eval-splits tune test

Chỉ train trên split tune (SPEC 8.7). `--eval-splits test` chỉ ĐO, không dùng để chỉnh tham số.
Top-1: ứng viên điểm cao nhất, đúng nếu IoU với bbox thật >= evaluate.iou_threshold.
Để khỏi chạy localize_edge hai lần, script gọi localize_edge một lần rồi score_candidates trên đúng
danh sách đó (tương đương localize_edge_svm).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from lpr.config import load_config
from lpr.data.dataset import load_records
from lpr.evaluate import iou
from lpr.io_utils import read_image
from lpr.localize.edge import localize_edge
from lpr.localize.hog_svm import build_patch_dataset, save_model, score_candidates, train_svm

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def top1_accuracy(model, cfg: dict, split: str) -> dict:
    thr = cfg["evaluate"]["iou_threshold"]
    n = hit_edge = hit_svm = n_any = 0
    for r in load_records(cfg, split=split):
        img = read_image(r.path)
        cands = localize_edge(img, cfg)
        svm = score_candidates(img, cands, model, cfg)
        n += 1
        n_any += bool(cands)
        hit_edge += bool(cands) and iou(cands[0].bbox, r.bbox) >= thr
        hit_svm += bool(svm) and iou(svm[0].bbox, r.bbox) >= thr
    return {"split": split, "n": n, "with_candidates": n_any,
            "top1_edge": hit_edge / n, "top1_edge_svm": hit_svm / n,
            "hits_edge": hit_edge, "hits_edge_svm": hit_svm}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default=None, help="mặc định <paths.models>/svm.joblib")
    ap.add_argument("--eval-splits", nargs="*", default=[], help="vd: tune test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = args.out or os.path.join(cfg["paths"]["models"], "svm.joblib")
    t0 = time.time()
    records = load_records(cfg, split="tune")
    X, y = build_patch_dataset(records, cfg)
    print(f"tune: {len(records)} ảnh -> X {X.shape}, dương {int((y == 1).sum())}, âm {int((y == 0).sum())} "
          f"({time.time() - t0:.0f}s)")
    model, cv = train_svm(X, y, cfg)
    save_model(model, out)
    print("CV 5 fold:", json.dumps({k: v for k, v in cv.items() if k != "folds"}, ensure_ascii=False))
    for i, f in enumerate(cv["folds"]):
        print(f"  fold {i}:", {k: round(v, 4) for k, v in f.items()})
    print("đã lưu", out)
    for s in args.eval_splits:
        print("top1", json.dumps(top1_accuracy(model, cfg, s), ensure_ascii=False))


if __name__ == "__main__":
    main()
