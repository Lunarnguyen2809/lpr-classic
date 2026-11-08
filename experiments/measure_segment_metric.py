"""Đo metric tách ký tự (count_match, position_hits) trên tập tune.

Chạy:
    python experiments/measure_segment_metric.py                    # dùng config/default.yaml
    python experiments/measure_segment_metric.py --aspect-min 0.12  # ghi đè segment.char_aspect_range[0]
    python experiments/measure_segment_metric.py --ratio 1.5        # ghi đè evaluate.char_center_max_width_ratio

Pipeline: nắn biển bằng quad thật (rectify "quad") -> to_gray -> binarize theo cfg["binarize"]
-> ensure_text_white + clean_binary -> segment "cca" -> evaluate.
Output: all_correct (count == 7 và cả 7 vị trí trúng), count, pos0, pos1_6, và số vị trí "1" trúng trên subset base.
Lưu ý: position_hits là "có hộp nào trúng" (không ghép 1-1), nên mảnh vỡ cũng có thể được tính; all_correct được chặn bởi count == 7.
"""
from __future__ import annotations

import argparse
import sys

from lpr.binarize import binarize_plate, clean_binary, ensure_text_white
from lpr.config import load_config
from lpr.data.dataset import load_records
from lpr.evaluate import count_match, position_hits
from lpr.io_utils import read_image
from lpr.preprocess import to_gray
from lpr.rectify import rectify
from lpr.segment import segment

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--split", default="tune")
    ap.add_argument("--aspect-min", type=float, default=None)
    ap.add_argument("--ratio", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.aspect_min is not None:
        cfg["segment"]["char_aspect_range"][0] = args.aspect_min
    if args.ratio is not None:
        cfg.setdefault("evaluate", {})["char_center_max_width_ratio"] = args.ratio
    print("char_aspect_range", cfg["segment"]["char_aspect_range"],
          "char_center_max_width_ratio", cfg.get("evaluate", {}).get("char_center_max_width_ratio"))

    bc = cfg["binarize"]
    n = n_count = n_all = 0
    p0 = p16 = 0.0
    ones_hit = ones_tot = 0
    for r in load_records(cfg, split=args.split):
        gray = to_gray(rectify(read_image(r.path), r.bbox, r.quad, "quad", cfg))
        binary = clean_binary(ensure_text_white(binarize_plate(gray, bc["method"], bc.get(bc["method"], {}))))
        boxes = segment(binary, "cca", cfg)
        hits = position_hits(boxes, cfg)
        cm = count_match(boxes, r.text)
        n += 1
        n_count += cm
        n_all += cm and all(hits)
        p0 += hits[0]
        p16 += sum(hits[1:]) / 6
        if r.subset == "base":
            for i, ch in enumerate(r.text):
                if ch == "1":
                    ones_tot += 1
                    ones_hit += hits[i]
    print(f"n={n} all_correct={n_all / n:.3f} count={n_count / n:.3f} "
          f"pos0={p0 / n:.3f} pos1_6={p16 / n:.3f} ones_base={ones_hit}/{ones_tot}")


if __name__ == "__main__":
    main()
