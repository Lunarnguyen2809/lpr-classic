# OWNER: B
"""Gán tay hộp ký tự cho 30 biển test (SPEC 5.5, 9.4), mỗi người 15 biển.

Chọn 30 biển cố định (seed trong config): 22 base + 2 mỗi subset khó, chia đôi để hai phần
có cùng tỉ lệ subset. Phần 1 = Bích, phần 2 = Châu Anh. Mỗi biển được nắn bằng quad thật về
220×70 (đúng hệ tọa độ SPEC 5.5), hiện kèm hộp gợi ý từ segment để chỉ cần sửa, không vẽ từ đầu.

Chạy:
    python scripts/label_chars.py --part 1          # mở cửa sổ gán 15 biển của phần 1
    python scripts/label_chars.py --part 2
    python scripts/label_chars.py --list            # in 30 biển và đã gán bao nhiêu

Trong cửa sổ (ảnh phóng to 4 lần):
    chuột trái kéo   vẽ hộp mới
    chuột phải       xóa hộp chứa điểm bấm
    n / Enter / Space  lưu, sang biển sau        p   lưu, về biển trước
    u  bỏ hộp vừa vẽ    r  về lại hộp gợi ý      c  xóa hết hộp
    q / Esc  lưu và thoát

Quy ước gán (hai người phải giống nhau thì P/R/F1 mới so được):
    - Mỗi ký tự một hộp, ôm sát phần nét chữ nhìn thấy (không ôm cả ô khuôn 45 mm).
    - Ký tự Hán đầu biển là MỘT hộp dù vỡ thành nhiều mảnh.
    - Không gán dấu chấm "·" giữa ký tự 2 và 3, không gán viền, ốc vít.
    - Ký tự bị che / quá mờ không thấy nét thì vẫn vẽ hộp theo chỗ đoán được nó nằm.

Lưu vào data/char_labels/char_boxes.csv sau mỗi biển: chỉ thay các dòng của biển đó, giữ
nguyên dòng của biển khác (của người kia). Hộp được sắp trái → phải thành char_idx 0..n-1.
Biển đã gán sẵn thì mở lại hộp đã lưu thay vì hộp gợi ý.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

from lpr.binarize import binarize_plate, clean_binary, ensure_text_white
from lpr.config import load_config
from lpr.data.dataset import load_records
from lpr.io_utils import read_image
from lpr.preprocess import to_gray
from lpr.rectify import rectify
from lpr.segment import segment
from lpr.types import CharBox, PlateRecord

for _s in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, không in được tiếng Việt
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

N_TOTAL = 30
HARD_PER_SUBSET = 2                    # mỗi subset khó lấy 2 biển, còn lại là base
HARD_SUBSETS = ["blur", "challenge", "db", "tilt"]
COLS = ["image_id", "char_idx", "x1", "y1", "x2", "y2"]
SCALE = 4
WIN = "label_chars"


# ---------- logic (không cần giao diện, có test) ----------

def pick_plates(cfg: dict, part: int | None = None) -> list[PlateRecord]:
    """30 biển test cố định theo cfg["seed"]; part = 1 hoặc 2 trả về 15 biển của phần đó.

    Mỗi subset khó lấy 2 biển, base lấy phần còn lại. Hai phần nhận biển xen kẽ trong từng
    subset, nên phần nào cũng có 11 base và 1 biển mỗi subset khó.
    """
    rng = np.random.default_rng(cfg["seed"])
    recs = load_records(cfg, split="test")
    by_sub: dict[str, list[PlateRecord]] = {}
    for r in recs:
        by_sub.setdefault(r.subset, []).append(r)
    quota = {s: HARD_PER_SUBSET for s in HARD_SUBSETS}
    quota["base"] = N_TOTAL - HARD_PER_SUBSET * len(HARD_SUBSETS)
    parts: tuple[list, list] = ([], [])
    for sub in ["base", *HARD_SUBSETS]:
        pool = sorted(by_sub.get(sub, []), key=lambda r: r.image_id)
        idx = sorted(rng.choice(len(pool), size=min(quota[sub], len(pool)), replace=False))
        for k, i in enumerate(idx):
            parts[k % 2].append(pool[i])
    if part is None:
        return parts[0] + parts[1]
    if part not in (1, 2):
        raise ValueError("part phải là 1 hoặc 2")
    return parts[part - 1]


def plate_image(rec: PlateRecord, cfg: dict) -> np.ndarray:
    """Ảnh biển BGR 220×70 nắn bằng quad thật (hệ tọa độ của char_boxes.csv)."""
    return rectify(read_image(rec.path), rec.bbox, rec.quad, "quad", cfg)


def suggest_boxes(plate_bgr: np.ndarray, cfg: dict) -> list[CharBox]:
    """Hộp gợi ý từ pipeline hiện tại (binarize + segment theo config)."""
    bm = cfg["binarize"]["method"]
    b = clean_binary(ensure_text_white(binarize_plate(to_gray(plate_bgr), bm, cfg["binarize"].get(bm, {}))))
    return segment(b, cfg["segment"]["method"], cfg)


def load_labels(path: str) -> pd.DataFrame:
    if os.path.isfile(path):
        return pd.read_csv(path, dtype={"image_id": str})
    return pd.DataFrame(columns=COLS)


def boxes_of(df: pd.DataFrame, image_id: str) -> list[CharBox]:
    g = df[df["image_id"] == image_id].sort_values("char_idx")
    return [CharBox(int(r.x1), int(r.y1), int(r.x2), int(r.y2)) for r in g.itertuples()]


def replace_boxes(df: pd.DataFrame, image_id: str, boxes: list[CharBox]) -> pd.DataFrame:
    """Bỏ mọi dòng của image_id rồi thêm boxes (sắp trái → phải, char_idx 0..n-1); dòng khác giữ nguyên."""
    keep = df[df["image_id"] != image_id]
    new = pd.DataFrame([{"image_id": image_id, "char_idx": i, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}
                        for i, b in enumerate(sorted(boxes, key=lambda b: b.x1))], columns=COLS)
    out = pd.concat([keep, new], ignore_index=True) if len(keep) else new
    return out.sort_values(["image_id", "char_idx"], ignore_index=True)[COLS]


def save_labels(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def box_from_drag(x0: int, y0: int, x1: int, y1: int, w: int, h: int, scale: int = SCALE) -> CharBox | None:
    """Đổi một lần kéo chuột (tọa độ cửa sổ) sang hộp trên biển 220×70; quá nhỏ (< 3 px, tức bấm nhầm) thì None."""
    ax, bx = sorted((x0 // scale, x1 // scale))
    ay, by = sorted((y0 // scale, y1 // scale))
    ax, ay = max(0, ax), max(0, ay)
    bx, by = min(w, bx + 1), min(h, by + 1)       # x2 / y2 không bao gồm
    if bx - ax < 3 or by - ay < 3:
        return None
    return CharBox(ax, ay, bx, by)


def remove_at(boxes: list[CharBox], x: int, y: int) -> list[CharBox]:
    """Xóa hộp nhỏ nhất chứa điểm (x, y) theo tọa độ biển; không có hộp nào thì giữ nguyên."""
    hit = [b for b in boxes if b.x1 <= x < b.x2 and b.y1 <= y < b.y2]
    if not hit:
        return list(boxes)
    victim = min(hit, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
    return [b for b in boxes if b is not victim]


# ---------- giao diện OpenCV ----------

class Labeler:
    def __init__(self, recs: list[PlateRecord], cfg: dict, path: str, start: int = 0):
        self.recs, self.cfg, self.path = recs, cfg, path
        self.df = load_labels(path)
        self.i = max(0, min(len(recs) - 1, start))
        self.drag: tuple[int, int, int, int] | None = None
        self._load()

    def _load(self) -> None:
        r = self.recs[self.i]
        self.plate = plate_image(r, self.cfg)
        self.suggested = suggest_boxes(self.plate, self.cfg)
        saved = boxes_of(self.df, r.image_id)
        self.boxes = saved if saved else list(self.suggested)
        self.added: list[CharBox] = []
        src = "đã lưu" if saved else "gợi ý"
        print(f"[{self.i + 1}/{len(self.recs)}] {r.image_id} ({r.subset}) text={r.text}  "
              f"{len(self.boxes)} hộp ({src})")

    def save(self) -> None:
        r = self.recs[self.i]
        self.df = replace_boxes(self.df, r.image_id, self.boxes)
        save_labels(self.df, self.path)
        n_exp = len(r.text or "")
        warn = "" if len(self.boxes) == n_exp else f"  CHÚ Ý: {len(self.boxes)} hộp, biển có {n_exp} ký tự"
        print(f"   đã lưu {len(self.boxes)} hộp{warn}")

    def on_mouse(self, event, x, y, flags, param) -> None:
        ph, pw = self.plate.shape[:2]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            self.drag = (self.drag[0], self.drag[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            b = box_from_drag(self.drag[0], self.drag[1], x, y, pw, ph)
            self.drag = None
            if b:
                self.boxes.append(b)
                self.added.append(b)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.boxes = remove_at(self.boxes, x // SCALE, y // SCALE)

    def render(self) -> np.ndarray:
        img = cv2.resize(self.plate, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
        for k, b in enumerate(sorted(self.boxes, key=lambda b: b.x1)):
            cv2.rectangle(img, (b.x1 * SCALE, b.y1 * SCALE), (b.x2 * SCALE - 1, b.y2 * SCALE - 1), (0, 255, 0), 2)
            cv2.putText(img, str(k), (b.x1 * SCALE + 3, b.y1 * SCALE + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if self.drag:
            x0, y0, x1, y1 = self.drag
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        r = self.recs[self.i]
        bar = np.full((48, img.shape[1], 3), 30, np.uint8)
        # putText không vẽ được chữ Hán / tiếng Việt có dấu nên thanh trạng thái để ASCII
        cv2.putText(bar, f"{self.i + 1}/{len(self.recs)}  {r.image_id}  boxes {len(self.boxes)}/{len(r.text or '')}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, "L-drag: ve | R-click: xoa | n: tiep | p: lui | u: undo | r: goi y | c: xoa het | q: thoat",
                    (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        return np.vstack([img, bar])

    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        while True:
            cv2.imshow(WIN, self.render())
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("n"), 13, 32):
                self.save()
                if self.i == len(self.recs) - 1:
                    print("hết danh sách")
                    break
                self.i += 1
                self._load()
            elif k == ord("p"):
                self.save()
                self.i = max(0, self.i - 1)
                self._load()
            elif k == ord("u") and self.added:
                last = self.added.pop()
                self.boxes = [b for b in self.boxes if b is not last]
            elif k == ord("r"):
                self.boxes, self.added = list(self.suggested), []
            elif k == ord("c"):
                self.boxes, self.added = [], []
            elif k in (ord("q"), 27):
                self.save()
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:   # bấm nút X của cửa sổ
                self.save()
                break
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=int, choices=[1, 2], help="1 = Bích, 2 = Châu Anh")
    ap.add_argument("--list", action="store_true", help="in 30 biển và tình trạng đã gán")
    ap.add_argument("--start", type=int, default=1, help="bắt đầu từ biển thứ mấy (1..15)")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = cfg["paths"]["char_labels"]
    if args.list:
        df = load_labels(path)
        done = set(df["image_id"])
        for part in (1, 2):
            recs = pick_plates(cfg, part)
            n_done = sum(r.image_id in done for r in recs)
            print(f"phần {part}: đã gán {n_done}/{len(recs)}")
            for r in recs:
                mark = "x" if r.image_id in done else " "
                print(f"  [{mark}] {r.image_id:<16}{r.subset:<11}{r.text}")
        return
    if not args.part:
        ap.error("cần --part 1|2 hoặc --list")
    Labeler(pick_plates(cfg, args.part), cfg, path, start=args.start - 1).run()


if __name__ == "__main__":
    main()
