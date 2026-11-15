# OWNER: SHARED
"""Test phần thuần của scripts/build_vn_set.py (không cần dữ liệu thật)."""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "build_vn_set", Path(__file__).resolve().parents[1] / "scripts" / "build_vn_set.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["build_vn_set"] = B
_spec.loader.exec_module(B)


def test_yolo_box_to_pixels():
    assert B.yolo_box_to_xyxy(0.5, 0.5, 0.5, 0.5, 100, 200) == (25, 50, 75, 150)


def test_yolo_box_clamped():
    x1, y1, x2, y2 = B.yolo_box_to_xyxy(0.99, 0.01, 0.2, 0.2, 100, 100)
    assert (x1, y1, x2, y2) == (89, 0, 100, 11)


def test_polygon_to_rect():
    poly = [0.1, 0.2, 0.5, 0.25, 0.48, 0.6, 0.12, 0.55]
    assert B.polygon_to_xyxy(poly, 200, 100) == (20, 20, 100, 60)


def test_aug_detection():
    assert B.is_aug("a/b_jpg.rf.xyz_aug1.jpg") and not B.is_aug("a/b_jpg.rf.xyz.jpg")


def test_label_path():
    p = Path("d/splits/train/images/a.jpg")
    assert B.label_path(p) == Path("d/splits/train/labels/a.txt")
    assert B.label_path(Path("d/duy/images/val/x.png")) == Path("d/duy/labels/val/x.txt")


def _mk(tmp_path):
    kg = tmp_path / "kg"
    for sub in ("tanhphp/splits/train/images", "tanhphp/splits/train/labels",
                "duy/images/train", "duy/labels/train"):
        (kg / sub).mkdir(parents=True)
    for n in ("b", "a"):
        Image.new("RGB", (100, 50), "gray").save(kg / f"tanhphp/splits/train/images/{n}.jpg")
        (kg / f"tanhphp/splits/train/labels/{n}.txt").write_text("0 0.5 0.5 0.2 0.4\n0 0.5 0.5 0.5 0.8\n")
    Image.new("RGB", (200, 100), "gray").save(kg / "duy/images/train/c.png")
    (kg / "duy/labels/train/c.txt").write_text("1 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    root = tmp_path / "vn"
    (root / "candidates" / "x").mkdir(parents=True)
    Image.new("RGB", (80, 60), "gray").save(root / "candidates/x/vnc_1.png")
    with open(root / "candidates/manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "source_page_url", "original_image_url", "file"])
        w.writerow(["vnc_1", "http://page", "http://img", "x/vnc_1.png"])
    sel = tmp_path / "sel.csv"
    rows = [
        ("kaggle_duydieunguyen", "duy/images/train/c.png", "white", "car", "2", "cctv"),
        ("kaggle_tanhphp", "tanhphp/splits/train/images/b.jpg", "white", "bike", "2", "night"),
        ("brave_crawl", "vnc_1", "yellow", "car", "1", "day"),
        ("kaggle_tanhphp", "tanhphp/splits/train/images/a.jpg", "white", "car", "2", "day;blur"),
    ]
    with open(sel, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "source_ref", "plate_colour", "vehicle", "n_lines", "conditions"])
        w.writerows(rows)
    return sel, root, kg


def test_build_naming_annotations_deterministic(tmp_path):
    sel, root, kg = _mk(tmp_path)
    res = B.build(sel, root, kg)
    assert res == {"images": 4, "annotated": 3, "todo_cvat": 1, "n_lines_mismatch": 0}
    src = list(csv.DictReader(open(root / "sources.csv", encoding="utf-8")))
    # thứ tự: crawl, tanhphp (a, b), duydieunguyen
    assert [r["source_ref"] for r in src] == ["vnc_1", "tanhphp/splits/train/images/a.jpg",
                                              "tanhphp/splits/train/images/b.jpg", "duy/images/train/c.png"]
    assert [r["image_id"] for r in src] == ["vn_0001", "vn_0002", "vn_0003", "vn_0004"]
    assert list(src[0].keys()) == B.SOURCES_FIELDS
    assert src[0]["license"] == "internal_only" and src[0]["original_url"] == "http://page"
    ann = list(csv.DictReader(open(root / "annotations.csv", encoding="utf-8")))
    assert [r["image_id"] for r in ann] == ["vn_0002", "vn_0003", "vn_0004"]
    # biển lớn nhất (0.5x0.8 của 100x50) -> 25,5,75,45
    assert (ann[0]["x1"], ann[0]["y1"], ann[0]["x2"], ann[0]["y2"]) == ("25", "5", "75", "45")
    # polygon 0.1..0.5 của 200x100 -> 20,10,100,50
    assert (ann[2]["x1"], ann[2]["y1"], ann[2]["x2"], ann[2]["y2"]) == ("20", "10", "100", "50")
    assert "vn_0001" in (root / "todo_cvat.txt").read_text(encoding="utf-8")
    first = (root / "images/vn_0002.jpg").read_bytes()
    (root / "images" / ".gitkeep").write_text("")
    with pytest.raises(SystemExit):  # images/ đã có nội dung: không ghi đè khi thiếu --force
        B.build(sel, root, kg)
    B.build(sel, root, kg, force=True)
    assert (root / "images/vn_0002.jpg").read_bytes() == first


def test_aug_rejected(tmp_path):
    sel, root, kg = _mk(tmp_path)
    txt = sel.read_text(encoding="utf-8").replace("a.jpg", "a_aug0.jpg")
    sel.write_text(txt, encoding="utf-8")
    with pytest.raises(SystemExit):
        B.build(sel, root, kg)


def test_loader_accepts_annotations(tmp_path):
    from lpr.data.dataset import load_vn_records
    sel, root, kg = _mk(tmp_path)
    B.build(sel, root, kg)
    recs = load_vn_records({"paths": {"vn": str(root / "annotations.csv")}})
    assert len(recs) == 3 and recs[0].split == "qual"


def test_duy_n_lines_from_label(tmp_path, capsys):
    sel, root, kg = _mk(tmp_path)
    sel.write_text(sel.read_text(encoding="utf-8").replace("car,2,cctv", "car,1,cctv"), encoding="utf-8")
    res = B.build(sel, root, kg)
    assert res["n_lines_mismatch"] == 1 and "vn_0004" in capsys.readouterr().err
    ann = list(csv.DictReader(open(root / "annotations.csv", encoding="utf-8")))
    src = list(csv.DictReader(open(root / "sources.csv", encoding="utf-8")))
    assert ann[2]["n_lines"] == "2" and src[3]["n_lines"] == "2"  # class 1 = 2 dòng


def test_force_refuses_foreign_files(tmp_path):
    sel, root, kg = _mk(tmp_path)
    B.build(sel, root, kg)
    (root / "images" / "notes.txt").write_text("x")
    with pytest.raises(SystemExit):
        B.build(sel, root, kg, force=True)
    assert (root / "images/vn_0001.jpg").exists()  # chưa xoá gì
