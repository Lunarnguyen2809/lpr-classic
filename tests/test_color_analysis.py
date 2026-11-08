"""Test phân tích hệ màu (SPEC mục 8.5), dùng ảnh tổng hợp."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lpr import color_analysis as ca
from lpr.types import PlateRecord

CHANNELS = ["gray", "H", "S", "V", "L", "a", "b"]
BOX = (40, 30, 120, 60)


def _img() -> np.ndarray:
    """Nền xám trung tính, biển xanh dương bão hòa trong BOX."""
    img = np.full((100, 160, 3), 128, np.uint8)
    x1, y1, x2, y2 = BOX
    img[y1:y2, x1:x2] = (200, 60, 0)          # BGR: xanh dương
    return img


def test_plate_mask_inside_outside_and_clip():
    m = ca.plate_mask((100, 160), BOX)
    assert m.dtype == np.uint8 and m.shape == (100, 160)
    assert m[45, 80] == 255 and m[10, 10] == 0
    assert np.count_nonzero(m) == 80 * 30
    clipped = ca.plate_mask((100, 160), (-10, -10, 500, 20))   # bbox tràn ra ngoài ảnh
    assert np.count_nonzero(clipped) == 160 * 20


def test_channel_histograms_normalized(tmp_path):
    hists = ca.channel_histograms(_img(), BOX, CHANNELS, bins=32)
    assert set(hists) == set(CHANNELS)
    for h_in, h_out in hists.values():
        assert h_in.shape == h_out.shape == (32,)
        assert h_in.sum() == pytest.approx(1.0) and h_out.sum() == pytest.approx(1.0)


def test_channel_histograms_input_not_modified():
    img = _img()
    img0 = img.copy()
    ca.channel_histograms(img, BOX, CHANNELS)
    assert np.array_equal(img, img0)


def test_separability_bounds():
    p = np.array([0.5, 0.5, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.5, 0.5])
    assert ca.separability(p, p) == pytest.approx(0.0)
    assert ca.separability(p, q) == pytest.approx(1.0)
    mid = ca.separability(p, np.array([0.25, 0.25, 0.25, 0.25]))
    assert 0.0 < mid < 1.0
    assert ca.separability(np.zeros(4), p) == 0.0          # histogram rỗng không chia 0


def test_saturation_separates_blue_plate_better_than_gray():
    """Nền xám trung tính có đủ mọi mức sáng (S = 0), biển xanh bão hòa: độ sáng của biển lẫn
    vào nền nên gray tách kém, còn S tách hẳn."""
    rng = np.random.default_rng(0)
    img = np.repeat(rng.integers(0, 256, (100, 160, 1), dtype=np.uint8), 3, axis=2)
    x1, y1, x2, y2 = BOX
    img[y1:y2, x1:x2] = (200, 60, 0)
    hists = ca.channel_histograms(img, BOX, CHANNELS)
    sep = {ch: ca.separability(*hists[ch]) for ch in CHANNELS}
    assert sep["S"] == pytest.approx(1.0)
    assert sep["S"] > sep["gray"]


def test_channel_ranking_sorted(monkeypatch):
    recs = [PlateRecord(image_id=f"x{i}", path=f"p{i}", source="ccpd", subset="base",
                        split="tune", bbox=BOX) for i in range(3)]
    monkeypatch.setattr(ca, "read_image", lambda p: _img())
    df = ca.channel_ranking(recs, CHANNELS)
    assert list(df.columns) == ["channel", "mean_separability", "std"]
    assert len(df) == 7
    assert df["mean_separability"].is_monotonic_decreasing
    assert df.iloc[0]["mean_separability"] == pytest.approx(1.0)
    assert (df["std"] == 0).all()                          # 3 ảnh giống nhau


def test_channel_ranking_empty():
    df = ca.channel_ranking([], CHANNELS)
    assert df.empty and list(df.columns) == ["channel", "mean_separability", "std"]
