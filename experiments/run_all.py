# OWNER: B
"""Chạy các thí nghiệm E0–E8 (SPEC mục 8.13 và 10).

Ví dụ:
    python experiments/run_all.py --list
    python experiments/run_all.py --exp predict --split test --limit 20
    python experiments/run_all.py --exp all
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable

from lpr.config import load_config

for _s in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, không in được tiếng Việt
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def exp_predict(cfg: dict, args: argparse.Namespace) -> None:
    """E0: chạy pipeline trên split đã chọn, ghi results/predictions/localization.csv và segmentation.csv."""
    raise NotImplementedError("TODO(B): exp_predict")


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


def exp_binarize(cfg: dict, args: argparse.Namespace) -> None:
    """E5: otsu vs adaptive trên crop gt + CCA, ghi tables/e5_binarize.csv."""
    raise NotImplementedError("TODO(B): exp_binarize")


def exp_segmentation(cfg: dict, args: argparse.Namespace) -> None:
    """E6: cca vs projection × crop_source gt/pred (metric 9.3 + 9.4), ghi tables/e6_segmentation.csv."""
    raise NotImplementedError("TODO(B): exp_segmentation")


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
