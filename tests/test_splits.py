import importlib.util
import os

import pandas as pd
import pytest

from lpr.config import get, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_config(os.path.join(ROOT, "config", "default.yaml"))
EXPECTED = get(CFG, "data.subsets")
RATIO = get(CFG, "data.tune_ratio")


def _load_make_splits():
    spec = importlib.util.spec_from_file_location("make_splits", os.path.join(ROOT, "scripts", "make_splits.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_splits


def _check(ann: pd.DataFrame, splits: pd.DataFrame, vn_ids: list[str]) -> None:
    assert not splits["image_id"].duplicated().any(), "image_id nằm ở nhiều split"
    assert set(splits["split"]) <= {"tune", "test", "qual"}
    df = ann.merge(splits, on="image_id", how="left")
    assert df["split"].notna().all()
    for subset, g in df.groupby("subset"):
        frac = (g["split"] == "tune").mean()
        assert abs(frac - RATIO) < 0.02, f"{subset}: tỉ lệ tune {frac:.3f}"
        assert set(g["split"]) == {"tune", "test"}
    qual = splits[splits["image_id"].isin(vn_ids)]
    assert len(qual) == len(vn_ids) and (qual["split"] == "qual").all()


def test_make_splits_synthetic():
    make_splits = _load_make_splits()
    ann = pd.DataFrame([{"image_id": f"{s}_{i:04d}", "subset": s}
                        for s, n in EXPECTED.items() for i in range(n)])
    vn = pd.DataFrame({"image_id": [f"vn_{i:03d}" for i in range(40)]})
    splits = make_splits(ann, vn, RATIO, 42)
    _check(ann, splits, list(vn["image_id"]))
    assert splits.equals(make_splits(ann, vn, RATIO, 42)), "không tái lập được với cùng seed"
    n_tune = (splits["split"] == "tune").sum()
    assert n_tune == sum(round(n * RATIO) for n in EXPECTED.values())


def test_real_splits():
    ann_path = os.path.join(ROOT, get(CFG, "paths.annotations"))
    sp_path = os.path.join(ROOT, get(CFG, "paths.splits"))
    if not (os.path.isfile(ann_path) and os.path.isfile(sp_path)):
        pytest.skip("chưa có data/annotations.csv hoặc data/splits.csv (chạy build_subset + make_splits)")
    ann = pd.read_csv(ann_path, dtype={"image_id": str})
    splits = pd.read_csv(sp_path, dtype=str)
    ccpd = ann[ann["source"] == "ccpd"]
    assert len(ccpd) == sum(EXPECTED.values()) == 660
    assert ccpd["subset"].value_counts().to_dict() == EXPECTED
    vn_path = os.path.join(ROOT, get(CFG, "paths.vn"))
    vn_ids = list(pd.read_csv(vn_path, dtype=str)["image_id"]) if os.path.isfile(vn_path) else []
    _check(ccpd, splits, vn_ids)
