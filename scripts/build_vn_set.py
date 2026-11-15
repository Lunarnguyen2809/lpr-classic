# OWNER: SHARED
"""Dựng bộ VN (split `qual`, ~150–200 ảnh sau lọc, hiện 187) từ selection.csv (SPEC mục 5.4).

Nguồn: brave_crawl (ảnh crawl đã duyệt tay, chưa có bbox), kaggle_tanhphp (YOLO bbox),
kaggle_duydieunguyen (YOLO polygon). Kết quả, xác định theo selection.csv:
    data/vn_custom/images/vn_0001.jpg ...   (JPEG q95, đánh số theo nguồn rồi source_ref)
    data/vn_custom/sources.csv              (nguồn gốc, license, thẻ)
    data/vn_custom/annotations.csv          (image_id,file,x1,y1,x2,y2,n_lines; chỉ ảnh Kaggle, mỗi ảnh một dòng = biển lớn nhất)
    data/vn_custom/todo_cvat.txt            (ảnh crawl chưa có bbox, gán bằng CVAT rồi nối vào annotations.csv)

selection.csv: source, source_ref, plate_colour, vehicle, n_lines, conditions (cách nhau bằng ';').
source_ref: candidate_id (crawl) hoặc đường dẫn ảnh tương đối so với --kaggle-root (Kaggle).

Ví dụ:
    python scripts/build_vn_set.py --selection data/vn_custom/selection.csv
    python scripts/build_vn_set.py --selection data/vn_custom/selection.csv --force   # ghi đè images/ đã có
    python scripts/build_vn_set.py --sheet data/vn_custom/_sheet_final.jpg            # lưới soát bbox
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from PIL import Image

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

SOURCE_ORDER = ["brave_crawl", "kaggle_tanhphp", "kaggle_duydieunguyen"]
LICENSE = {"brave_crawl": "internal_only", "kaggle_tanhphp": "Apache-2.0",
           "kaggle_duydieunguyen": "unknown"}
KAGGLE_URL = {"kaggle_tanhphp": "https://www.kaggle.com/datasets/tanhphp/vietnamese-license-plates",
              "kaggle_duydieunguyen": "https://www.kaggle.com/datasets/duydieunguyen/licenseplates"}
SOURCES_FIELDS = ["image_id", "file", "source", "source_ref", "original_url", "license",
                  "plate_colour", "vehicle", "n_lines", "conditions"]
ANN_FIELDS = ["image_id", "file", "x1", "y1", "x2", "y2", "n_lines"]


def yolo_box_to_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int) -> tuple[int, int, int, int]:
    """YOLO bbox chuẩn hoá (cx, cy, w, h) -> pixel (x1, y1, x2, y2), làm tròn, kẹp trong ảnh."""
    return _clamp((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H, W, H)


def polygon_to_xyxy(coords: list[float], W: int, H: int) -> tuple[int, int, int, int]:
    """YOLO polygon chuẩn hoá (x1 y1 ... xn yn) -> hình chữ nhật bao pixel, kẹp trong ảnh."""
    xs, ys = coords[0::2], coords[1::2]
    return _clamp(min(xs) * W, min(ys) * H, max(xs) * W, max(ys) * H, W, H)


def _clamp(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W, int(round(x2))))
    y2 = max(0, min(H, int(round(y2))))
    return x1, y1, x2, y2


def label_path(img_path: Path) -> Path:
    """.../images/<split>/x.jpg hoặc .../<split>/images/x.jpg -> thư mục labels tương ứng, đuôi .txt."""
    parts = list(img_path.parts)
    i = max(k for k, p in enumerate(parts) if p == "images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_gt_box(source: str, lab: Path, W: int, H: int):
    """Trả (bbox pixel, class) của biển lớn nhất trong file nhãn YOLO, hoặc None nếu không có nhãn."""
    if not lab.is_file():
        return None
    best, best_area = None, -1
    for line in lab.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        nums = [float(x) for x in v[1:]]
        box = (yolo_box_to_xyxy(*nums[:4], W, H) if source == "kaggle_tanhphp"
               else polygon_to_xyxy(nums, W, H))
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > best_area:
            best, best_area = (box, int(float(v[0]))), area
    return best


def is_aug(name: str) -> bool:
    return "_aug" in os.path.basename(name)


def load_selection(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["source"] not in SOURCE_ORDER:
            sys.exit(f"source lạ: {r['source']}")
        if r["source"] != "brave_crawl" and is_aug(r["source_ref"]):
            sys.exit(f"ảnh augment bị cấm: {r['source_ref']}")
    keys = [(r["source"], r["source_ref"]) for r in rows]
    if len(set(keys)) != len(keys):
        sys.exit("selection.csv có dòng trùng (source, source_ref)")
    return sorted(rows, key=lambda r: (SOURCE_ORDER.index(r["source"]), r["source_ref"]))


def check_images_dir(img_dir: Path, force: bool) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    extra = [p.name for p in img_dir.iterdir() if p.name != ".gitkeep"]
    if extra and not force:
        sys.exit(f"{img_dir} đã có {len(extra)} file (vd {extra[0]}); thêm --force để ghi đè/xoá vn_*.jpg")
    foreign = [n for n in extra if not (n.startswith("vn_") and n.endswith(".jpg"))]
    if force and foreign:
        sys.exit(f"{img_dir} có file lạ (vd {foreign[0]}); --force chỉ xoá vn_*.jpg và .gitkeep, "
                 "dọn tay file khác trước")
    if force:
        for p in img_dir.glob("vn_*.jpg"):
            p.unlink()


def build(selection: Path, root: Path, kaggle_root: Path, force: bool = False) -> dict:
    rows = load_selection(selection)
    cand = {}
    mpath = root / "candidates" / "manifest.csv"
    if mpath.is_file():
        with open(mpath, encoding="utf-8", newline="") as f:
            cand = {r["candidate_id"]: r for r in csv.DictReader(f)}
    img_dir = root / "images"
    check_images_dir(img_dir, force)
    sources, anns, todo, mismatches = [], [], [], []
    for i, r in enumerate(rows, 1):
        image_id = f"vn_{i:04d}"
        src, ref = r["source"], r["source_ref"]
        if src == "brave_crawl":
            m = cand[ref]
            src_file = root / "candidates" / m["file"]
            url = m["source_page_url"] or m["original_image_url"]
        else:
            src_file = kaggle_root / ref
            url = KAGGLE_URL[src]
        with Image.open(src_file) as im:
            im = im.convert("RGB")
            W, H = im.size
            im.save(img_dir / f"{image_id}.jpg", "JPEG", quality=95)
        rel = f"data/vn_custom/images/{image_id}.jpg"
        sources.append({"image_id": image_id, "file": rel, "source": src, "source_ref": ref,
                        "original_url": url, "license": LICENSE[src],
                        "plate_colour": r["plate_colour"], "vehicle": r["vehicle"],
                        "n_lines": r["n_lines"], "conditions": r["conditions"]})
        if src == "brave_crawl":
            todo.append(f"{image_id}\t{rel}\t{ref}")
        else:
            gt = read_gt_box(src, label_path(src_file), W, H)
            if gt is None or gt[0][2] <= gt[0][0] or gt[0][3] <= gt[0][1]:
                sys.exit(f"{ref}: không có bbox hợp lệ")
            box, cls = gt
            n_lines = r["n_lines"]
            if src == "kaggle_duydieunguyen":  # class 0=1 dòng, 1=2 dòng: nhãn là chuẩn
                n_lines = str(cls + 1)
                if n_lines != r["n_lines"]:
                    mismatches.append(f"{image_id} {ref}: selection={r['n_lines']} nhãn={n_lines}")
                    sources[-1]["n_lines"] = n_lines
            anns.append({"image_id": image_id, "file": rel, "x1": box[0], "y1": box[1],
                         "x2": box[2], "y2": box[3], "n_lines": n_lines})
    if mismatches:
        print(f"CẢNH BÁO: {len(mismatches)} ảnh duydieunguyen lệch n_lines giữa selection.csv và nhãn "
              "(đã dùng nhãn; hãy sửa selection.csv):", file=sys.stderr)
        for m in mismatches:
            print("  " + m, file=sys.stderr)
    _write_csv(root / "sources.csv", SOURCES_FIELDS, sources)
    _write_csv(root / "annotations.csv", ANN_FIELDS, anns)
    (root / "todo_cvat.txt").write_text("\n".join(todo) + ("\n" if todo else ""), encoding="utf-8")
    return {"images": len(rows), "annotated": len(anns), "todo_cvat": len(todo),
            "n_lines_mismatch": len(mismatches)}


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def contact_sheet(root: Path, out: Path, cols: int = 10, thumb: tuple[int, int] = (200, 160)) -> None:
    """Lưới toàn bộ ảnh với GT bbox (đỏ) và image_id, để soát bbox bằng mắt."""
    from PIL import ImageDraw
    with open(root / "sources.csv", encoding="utf-8", newline="") as f:
        ids = [r["image_id"] for r in csv.DictReader(f)]
    boxes = {}
    ann = root / "annotations.csv"
    if ann.is_file():
        with open(ann, encoding="utf-8", newline="") as f:
            boxes = {r["image_id"]: tuple(int(r[k]) for k in ("x1", "y1", "x2", "y2")) for r in csv.DictReader(f)}
    tw, th = thumb
    rows = (len(ids) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 14)), "white")
    for k, iid in enumerate(ids):
        im = Image.open(root / "images" / f"{iid}.jpg").convert("RGB")
        if iid in boxes:
            ImageDraw.Draw(im).rectangle(boxes[iid], outline=(255, 0, 0), width=max(2, im.width // 250))
        im.thumbnail((tw, th))
        sheet.paste(im, ((k % cols) * tw, (k // cols) * (th + 14)))
        ImageDraw.Draw(sheet).text(((k % cols) * tw + 2, (k // cols) * (th + 14) + th), iid, fill=(0, 0, 0))
    sheet.save(out, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", default="data/vn_custom/selection.csv")
    ap.add_argument("--root", default="data/vn_custom")
    ap.add_argument("--kaggle-root", default="data/raw/vn_kaggle")
    ap.add_argument("--force", action="store_true", help="ghi đè images/ đã có")
    ap.add_argument("--sheet", metavar="OUT", help="chỉ dựng contact sheet (GT đỏ + image_id) từ kết quả có sẵn rồi thoát")
    a = ap.parse_args()
    if a.sheet:
        contact_sheet(Path(a.root), Path(a.sheet))
        print(f"đã ghi {a.sheet}")
        return
    res = build(Path(a.selection), Path(a.root), Path(a.kaggle_root), a.force)
    print(res)


if __name__ == "__main__":
    main()
