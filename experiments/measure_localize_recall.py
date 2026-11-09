"""Đo recall ứng viên và top-1 của localize_edge theo subset (issue #10).

Chạy:
    PYTHONPATH=src python experiments/measure_localize_recall.py                      # tune, config mặc định
    PYTHONPATH=src python experiments/measure_localize_recall.py --split test         # chỉ dùng 1 lần cho cấu hình đã chốt
    PYTHONPATH=src python experiments/measure_localize_recall.py --set localize.close_kernel=[25,5] --set localize.top_k=8

Chỉ số (IoU ≥ evaluate.iou_threshold, bbox ở tọa độ ảnh gốc):
    cand_recall = tỉ lệ ảnh có ÍT NHẤT MỘT ứng viên (trong top_k) trùng GT
    top1        = tỉ lệ ảnh mà ứng viên điểm cao nhất trùng GT (= Recall SPEC 9.1)
    P           = TP / (TP + FP) với top-1 (ảnh không có ứng viên chỉ tính FN)
    ms          = thời gian trung bình mỗi ảnh (một tiến trình, không đọc ảnh)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

from lpr.config import load_config
from lpr.data.dataset import load_records
from lpr.evaluate import iou
from lpr.io_utils import read_image
from lpr.localize.edge import localize_edge

sys.stdout.reconfigure(encoding="utf-8")


def _set(cfg: dict, key: str, value) -> None:
    d = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value


def _one(args):
    path, bbox, cfg, thr = args
    img = read_image(path)
    t = time.perf_counter()
    cands = localize_edge(img, cfg)
    dt = (time.perf_counter() - t) * 1000
    ious = [iou(c.bbox, bbox) for c in cands]
    return {"n": len(cands), "any": any(v >= thr for v in ious),
            "top1": bool(ious) and ious[0] >= thr, "ms": dt}


def measure(cfg: dict, split: str, workers: int = 1) -> dict:
    thr = cfg["evaluate"]["iou_threshold"]
    recs = load_records(cfg, split=split)
    jobs = [(r.path, r.bbox, cfg, thr) for r in recs]
    if workers > 1:
        with Pool(workers) as p:
            res = p.map(_one, jobs, chunksize=4)
    else:
        res = [_one(j) for j in jobs]
    agg: dict = defaultdict(lambda: defaultdict(float))
    for r, o in zip(recs, res):
        for k in (r.subset, "ALL"):
            a = agg[k]
            a["n"] += 1
            a["any"] += o["any"]
            a["top1"] += o["top1"]
            a["pred"] += o["n"] > 0
            a["ms"] += o["ms"]
            a["ncand"] += o["n"]
    out = {}
    for k, a in agg.items():
        n = a["n"]
        out[k] = {"n": int(n), "cand_recall": a["any"] / n, "top1": a["top1"] / n,
                  "P": a["top1"] / a["pred"] if a["pred"] else 0.0,
                  "ms": a["ms"] / n, "avg_cands": a["ncand"] / n}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--split", default="tune")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        _set(cfg, k, json.loads(v))
    res = measure(cfg, args.split, args.workers)
    if args.json:
        print(json.dumps(res))
        return
    print(f"split={args.split} overrides={args.set}")
    print("| subset | n | cand_recall | top1 | P | ms/ảnh | ứng viên TB |")
    print("|---|---|---|---|---|---|---|")
    for k in sorted(res, key=lambda s: (s == "ALL", s)):
        r = res[k]
        print(f"| {k} | {r['n']} | {r['cand_recall']:.3f} | {r['top1']:.3f} | {r['P']:.3f} | "
              f"{r['ms']:.0f} | {r['avg_cands']:.2f} |")


if __name__ == "__main__":
    main()
