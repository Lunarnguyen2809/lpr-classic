"""Test E0 (exp_predict): các hàm dựng dòng CSV theo schema SPEC 10.1 / 10.2, dùng ảnh tổng hợp."""
from __future__ import annotations

import importlib.util
import pathlib

import cv2
import numpy as np
import pytest

from lpr.config import load_config
from lpr.types import Candidate, PlateRecord

# experiments/ không phải package nên nạp run_all.py theo đường dẫn
_spec = importlib.util.spec_from_file_location(
    "run_all", pathlib.Path(__file__).resolve().parents[1] / "experiments" / "run_all.py")
run_all = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_all)

BOX = (260, 200, 480, 270)  # 220×70, đúng tỉ lệ biển


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config("config/default.yaml")


def _record(text: str = "皖A12345") -> PlateRecord:
    x1, y1, x2, y2 = BOX
    quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return PlateRecord(image_id="base_9999", path="", source="ccpd", subset="base", split="test",
                       bbox=BOX, quad=quad, text=text)


def _image(cfg: dict) -> np.ndarray:
    """Ảnh 400×720 nền xám, biển xanh chữ trắng đặt đúng khuôn 7 vị trí."""
    img = np.full((400, 720, 3), 120, np.uint8)
    x1, y1, x2, y2 = BOX
    img[y1:y2, x1:x2] = (160, 60, 20)                      # BGR xanh dương đậm
    scale = cfg["rectify"]["plate_size"][0] / 440
    for lo, hi in cfg["segment"]["char_template_mm"]:
        cx = x1 + (lo + hi) / 2 * scale
        cv2.rectangle(img, (int(cx - 7), y1 + 10), (int(cx + 7), y2 - 11), (255, 255, 255), -1)
    return img


def _cand(bbox, score=1.0) -> Candidate:
    x1, y1, x2, y2 = bbox
    quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return Candidate(bbox=bbox, quad=quad, score=score, method="edge")


# ---------- localization ----------

def test_loc_rows_schema_and_tp(cfg):
    cands = [_cand(BOX, 0.9), _cand((0, 0, 50, 20), 0.5)]
    rows = run_all.localization_rows(_record(), cands, 12.3, "edge", "bilateral", cfg)
    assert [list(r) for r in rows] == [run_all.LOC_COLS] * 2
    assert [r["rank"] for r in rows] == [1, 2]
    assert rows[0]["is_tp"] and rows[0]["iou"] == pytest.approx(1.0)
    assert not rows[1]["is_tp"] and rows[1]["iou"] == 0.0
    assert all(r["ms"] == 12.3 and r["denoise"] == "bilateral" for r in rows)


def test_loc_rows_no_candidate_keeps_image(cfg):
    rows = run_all.localization_rows(_record(), [], 5.0, "edge", "none", cfg)
    assert len(rows) == 1
    assert rows[0]["rank"] == 0 and rows[0]["x1"] is None and rows[0]["is_tp"] is False


def test_loc_rows_capped_at_top_k(cfg):
    cands = [_cand((i, 0, i + 10, 10)) for i in range(10)]
    rows = run_all.localization_rows(_record(), cands, 1.0, "edge", "none", cfg)
    assert len(rows) == cfg["localize"]["top_k"]


# ---------- segmentation ----------

def test_seg_rows_all_combos(cfg):
    rows = run_all.segmentation_rows(_record(), _image(cfg), _cand(BOX), cfg)
    assert len(rows) == 16                       # 2 nguồn × 2 crop × 2 binarize × 2 segment
    assert all(list(r) == run_all.SEG_COLS for r in rows)
    combos = {(r["crop_source"], r["crop_method"], r["binarize"], r["segmenter"]) for r in rows}
    assert len(combos) == 16
    gt_quad = [r for r in rows if r["crop_source"] == "gt" and r["crop_method"] == "quad"]
    assert all(r["n_expected"] == 7 for r in rows)
    assert all(r["all_correct"] for r in gt_quad), gt_quad   # biển sạch: mọi cách đều đúng


def test_seg_rows_pred_without_candidate(cfg):
    rows = run_all.segmentation_rows(_record(), _image(cfg), None, cfg)
    pred = [r for r in rows if r["crop_source"] == "pred"]
    assert len(pred) == 8
    assert all(r["n_boxes"] == 0 and not r["count_match"] and not r["all_correct"] for r in pred)
    assert any(r["all_correct"] for r in rows if r["crop_source"] == "gt")


def test_seg_rows_consistent_columns(cfg):
    for r in run_all.segmentation_rows(_record(), _image(cfg), _cand(BOX), cfg):
        assert 0 <= r["pos1_6_hits"] <= 6
        assert r["count_match"] == (r["n_boxes"] == r["n_expected"])
        if r["all_correct"]:
            assert r["count_match"] and r["pos0_hit"] and r["pos1_6_hits"] == 6


# ---------- E5 / E6 ----------

import argparse  # noqa: E402

import pandas as pd  # noqa: E402


def _seg_df() -> pd.DataFrame:
    """2 ảnh (base, db) × đủ 16 tổ hợp; chỉ (gt, quad, otsu, cca) của base là đúng hết."""
    rows = []
    for iid, sub in (("base_0001", "base"), ("db_0001", "db")):
        for src in ("gt", "pred"):
            for cm in ("bbox", "quad"):
                for bm in ("otsu", "adaptive"):
                    for sm in ("cca", "projection"):
                        ok = (iid, src, cm, bm, sm) == ("base_0001", "gt", "quad", "otsu", "cca")
                        rows.append({"image_id": iid, "subset": sub, "crop_source": src, "crop_method": cm,
                                     "binarize": bm, "segmenter": sm, "n_boxes": 7 if ok else 5,
                                     "n_expected": 7, "count_match": ok, "pos0_hit": ok,
                                     "pos1_6_hits": 6 if ok else 3, "all_correct": ok, "ms": 1.0})
    return pd.DataFrame(rows)


def test_summarize_segmentation_subset_and_all():
    df = _seg_df()
    df = df[(df.crop_source == "gt") & (df.crop_method == "quad") & (df.segmenter == "cca")]
    rows = run_all.summarize_segmentation(df, ["binarize"])
    otsu = {r["subset"]: r for r in rows if r["binarize"] == "otsu"}
    assert list(otsu) == ["base", "db", "ALL"]
    assert otsu["base"]["all_correct"] == 1.0 and otsu["db"]["all_correct"] == 0.0
    assert otsu["ALL"]["n"] == 2 and otsu["ALL"]["all_correct"] == 0.5
    assert otsu["ALL"]["pos1_6_acc"] == pytest.approx((6 + 3) / 12)


@pytest.fixture
def tmp_cfg(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "_segmentation_df", lambda c, a: _seg_df())
    monkeypatch.setattr(run_all, "load_char_labels", lambda c: {})
    return {**cfg, "paths": {**cfg["paths"], "results": str(tmp_path)},
            "binarize": {**cfg["binarize"], "method": "otsu"}}


def test_exp_binarize_writes_table(tmp_cfg, tmp_path):
    run_all.exp_binarize(tmp_cfg, argparse.Namespace(split="test", limit=None))
    t = pd.read_csv(tmp_path / "tables" / "e5_binarize.csv")
    assert set(t.binarize) == {"otsu", "adaptive"}
    assert len(t) == 6                                     # 2 binarize × (base, db, ALL)
    assert t[(t.binarize == "otsu") & (t.subset == "ALL")].all_correct.item() == 0.5


def test_exp_segmentation_writes_table(tmp_cfg, tmp_path):
    run_all.exp_segmentation(tmp_cfg, argparse.Namespace(split="test", limit=None))
    t = pd.read_csv(tmp_path / "tables" / "e6_segmentation.csv")
    assert len(t) == 12                                    # 2 segmenter × 2 crop_source × 3 subset
    assert {"char_n", "char_precision", "char_recall", "char_f1"} <= set(t.columns)
    assert t.char_precision.isna().all()                   # chưa có nhãn tay
    row = t[(t.segmenter == "cca") & (t.crop_source == "gt") & (t.subset == "ALL")]
    assert row.all_correct.item() == 0.5


def test_char_box_rows_perfect_labels(cfg, monkeypatch):
    """SPEC 9.4 với nhãn tay = hộp thật của biển tổng hợp: cả hai cách tách phải đạt F1 = 1."""
    rec = _record()
    img = _image(cfg)
    gray = run_all.to_gray(run_all.rectify(img, rec.bbox, rec.quad, "quad", cfg))
    b = run_all.clean_binary(run_all.ensure_text_white(run_all.binarize_plate(gray, "otsu", {})))
    labels = run_all.segment(b, "cca", cfg)
    assert len(labels) == 7
    monkeypatch.setattr(run_all, "load_char_labels", lambda c: {rec.image_id: labels})
    monkeypatch.setattr(run_all, "load_records", lambda c, split=None: [rec])
    monkeypatch.setattr(run_all, "read_image", lambda p: img)
    out = run_all.char_box_rows(cfg, argparse.Namespace(split="test", limit=None), "otsu")
    assert out[("cca", "ALL")]["char_f1"] == pytest.approx(1.0)
    assert out[("cca", "base")]["char_n"] == 1
    assert out[("projection", "ALL")]["char_f1"] == pytest.approx(1.0)


def test_color_table_rank_per_subset():
    sep = pd.DataFrame([
        {"image_id": "a", "subset": "base", "channel": "S", "separability": 0.9},
        {"image_id": "a", "subset": "base", "channel": "gray", "separability": 0.3},
        {"image_id": "b", "subset": "db", "channel": "S", "separability": 0.2},
        {"image_id": "b", "subset": "db", "channel": "gray", "separability": 0.4},
    ])
    rows = run_all.color_table(sep)
    by = {(r["subset"], r["channel"]): r for r in rows}
    assert by[("base", "S")]["rank"] == 1 and by[("db", "gray")]["rank"] == 1
    assert by[("ALL", "S")]["mean_separability"] == pytest.approx(0.55)
    assert by[("ALL", "S")]["n"] == 2 and by[("base", "S")]["std"] == 0.0


# ---------- E2 / E3 / E4 / E7: hàm dựng bảng (không cần dataset) ----------

def _loc_df() -> pd.DataFrame:
    """3 ảnh: a (base) top-1 đúng; b (base) top-1 sai, rank 3 đúng; c (db) không có ứng viên."""
    def row(iid, sub, rank, tp, ms):
        return {"image_id": iid, "subset": sub, "method": "edge", "denoise": "none", "rank": rank,
                "is_tp": tp, "ms": ms}
    rows = [row("a", "base", 1, True, 10.0), row("a", "base", 2, False, 10.0),
            row("b", "base", 1, False, 20.0), row("b", "base", 2, False, 20.0), row("b", "base", 3, True, 20.0),
            row("c", "db", 0, False, 30.0)]
    return pd.DataFrame(rows)


def test_summarize_localization_counts_and_recall_at_k():
    rows = run_all.summarize_localization(_loc_df(), ["method"])
    by_sub = {r["subset"]: r for r in rows}
    assert list(by_sub) == ["base", "db", "ALL"]
    base, db, all_ = by_sub["base"], by_sub["db"], by_sub["ALL"]
    assert (base["tp"], base["fp"], base["fn"]) == (1, 1, 1)
    assert base["precision"] == 0.5 and base["recall"] == 0.5
    assert (base["recall_at_1"], base["recall_at_3"], base["recall_at_5"]) == (0.5, 1.0, 1.0)
    # ảnh không có ứng viên: FN nhưng không phải FP
    assert (db["tp"], db["fp"], db["fn"]) == (0, 0, 1)
    assert (all_["n"], all_["tp"], all_["fp"], all_["fn"]) == (3, 1, 1, 2)
    assert all_["precision"] == 0.5 and all_["recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert all_["recall_at_3"] == pytest.approx(2 / 3, abs=1e-4)
    assert all_["ms_mean"] == 20.0


def test_summarize_localization_groups_by_method():
    df = _loc_df()
    df2 = df.assign(method="edge_svm", is_tp=~df["is_tp"].astype(bool) & (df["rank"] >= 1))
    rows = run_all.summarize_localization(pd.concat([df, df2]), ["method"])
    assert {r["method"] for r in rows} == {"edge", "edge_svm"}
    assert all(r["n"] == 3 for r in rows if r["subset"] == "ALL")


def test_pick_best_uses_only_tune_and_breaks_ties_by_fixed_order():
    rows = [{"split": "tune", "subset": "ALL", "f1": 0.5, "ms_mean": 30.0, "denoise": "a"},
            {"split": "tune", "subset": "ALL", "f1": 0.5, "ms_mean": 10.0, "denoise": "b"},
            {"split": "tune", "subset": "ALL", "f1": 0.4, "ms_mean": 1.0, "denoise": "c"},
            {"split": "tune", "subset": "base", "f1": 0.9, "ms_mean": 1.0, "denoise": "d"},
            {"split": "test", "subset": "ALL", "f1": 0.99, "ms_mean": 1.0, "denoise": "e"}]
    # hòa F1 giữa a và b: không xét ms_mean, theo thứ tự cố định
    assert run_all.pick_best(rows, "denoise") == "a"                      # mặc định: thứ tự xuất hiện
    assert run_all.pick_best(rows, "denoise", order=["b", "a", "c"]) == "b"
    assert run_all.pick_best(rows, "denoise", order=["c", "a", "b"]) == "a"
    assert run_all.pick_best(list(reversed(rows)), "denoise", order=["a", "b"]) == "a"   # không phụ thuộc thứ tự dòng
    with pytest.raises(ValueError):
        run_all.pick_best([r for r in rows if r["split"] == "test"], "denoise")


def _chosen_df(n_tune: int) -> pd.DataFrame:
    rows = []
    for split, n in (("tune", n_tune), ("test", 7)):
        for sub in ("ALL", "base"):
            for dn in ("none", "bilateral"):
                rows.append({"split": split, "subset": sub, "denoise": dn, "n": n,
                             "chosen": dn == "bilateral"})
    return pd.DataFrame(rows)


def test_choose_from_table_requires_matching_tune_n():
    df = _chosen_df(198)
    assert run_all.choose_from_table(df, "denoise", 198) == ("bilateral", 198)
    assert run_all.choose_from_table(df, "denoise", 20) is None            # chạy với --limit khác: không dùng lại
    assert run_all.choose_from_table(df.drop(columns=["split"]), "denoise", 198) is None
    assert run_all.choose_from_table(df[df["split"] == "test"], "denoise", 198) is None   # thiếu tune
    assert run_all.choose_from_table(df.assign(chosen=False), "denoise", 198) is None


def test_records_rejects_qual_and_empty(cfg):
    with pytest.raises(SystemExit, match="qual"):
        run_all._records(cfg, "qual", None, "E2")
    with pytest.raises(SystemExit, match="không có record"):
        run_all._records(cfg, "tune", 0, "E3")


def _seg_row(src, cm, bm="otsu", sm="cca", ok=True) -> dict:
    return {"image_id": "base_1", "subset": "base", "crop_source": src, "crop_method": cm, "binarize": bm,
            "segmenter": sm, "n_boxes": 7, "n_expected": 7, "count_match": ok, "pos0_hit": ok,
            "pos1_6_hits": 6, "all_correct": ok, "ms": 1.0}


def test_rectify_image_rows_filters_and_attaches_corner():
    seg = [_seg_row(s, c, b, m) for s in ("gt", "pred") for c in ("bbox", "quad")
           for b in ("otsu", "adaptive") for m in ("cca", "projection")]
    rows = run_all.rectify_image_rows(seg, {"bbox": 8.0, "quad": 5.0}, loc_ok=True)
    assert len(rows) == 6   # gt×2 + pred×2 + pred_tp×2, chỉ otsu + cca
    ce = {(r["crop_source"], r["crop_method"]): r["corner_error"] for r in rows}
    assert ce[("pred", "bbox")] == 8.0 and ce[("pred_tp", "quad")] == 5.0
    assert ce[("gt", "quad")] is None
    # định vị sai: không có pred_tp
    rows = run_all.rectify_image_rows(seg, {}, loc_ok=False)
    assert len(rows) == 4 and not any(r["crop_source"] == "pred_tp" for r in rows)


def test_rectify_table_corner_stats_and_all_row():
    df = pd.DataFrame(run_all.rectify_image_rows(
        [_seg_row("pred", "quad"), _seg_row("pred", "bbox")], {"bbox": 10.0, "quad": 4.0}, True)
        + [{**_seg_row("pred", "quad"), "image_id": "db_1", "subset": "db", "corner_error": None,
            "count_match": False, "all_correct": False}])
    rows = run_all.rectify_table(df)
    q = {(r["crop_source"], r["subset"]): r for r in rows if r["crop_method"] == "quad"}
    assert q[("pred", "ALL")]["n"] == 2 and q[("pred", "ALL")]["corner_n"] == 1
    assert q[("pred", "ALL")]["corner_mean"] == 4.0 and q[("pred", "db")]["corner_mean"] is None
    assert q[("pred_tp", "ALL")]["n"] == 1 and q[("pred", "ALL")]["count_match"] == 0.5


def test_runtime_table_mean_p95_and_subset():
    rows = [{"image_id": f"i{i}", "subset": "base" if i < 3 else "db", "step": "total", "ms": float(i * 10)}
            for i in range(1, 5)]
    rows.append({"image_id": "i1", "subset": "base", "step": "segment", "ms": 2.0})
    t = run_all.runtime_table(pd.DataFrame(rows))
    assert [r["step"] for r in t].index("segment") < [r["step"] for r in t].index("total")  # theo RUNTIME_STEPS
    tot = {r["subset"]: r for r in t if r["step"] == "total"}
    assert list(tot) == ["base", "db", "ALL"]
    assert tot["ALL"]["n"] == 4 and tot["ALL"]["ms_mean"] == 25.0
    assert tot["ALL"]["ms_p95"] == pytest.approx(38.5)
    assert tot["base"]["n"] == 2 and tot["db"]["ms_mean"] == 35.0
