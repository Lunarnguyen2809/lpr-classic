"""Test visualize.py (SPEC 8.12)."""
import os

import cv2
import numpy as np
import pytest

from lpr.io_utils import read_image
from lpr.types import Candidate, CharBox, DebugSteps
from lpr.visualize import draw_candidates, draw_char_boxes, save_comparison, save_debug_grid


def _cand(score=0.9):
    quad = np.array([[20, 20], [80, 20], [80, 50], [20, 50]], np.float32)
    return Candidate(bbox=(20, 20, 81, 51), quad=quad, score=score, method="edge")


def test_draw_candidates_bgr_not_mutating():
    img = np.zeros((100, 120, 3), np.uint8)
    ref = img.copy()
    out = draw_candidates(img, [_cand()], gt=(10, 10, 60, 40))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(img, ref)
    assert out is not img
    assert tuple(out[20, 40]) == (0, 0, 255)      # cạnh trên quad: đỏ
    assert tuple(out[10, 30]) == (0, 255, 0)      # cạnh trên bbox thật: xanh lá


def test_draw_candidates_gray_and_empty():
    gray = np.full((50, 60), 128, np.uint8)
    out = draw_candidates(gray, [])
    assert out.shape == (50, 60, 3)
    assert np.all(out == 128)
    assert gray.ndim == 2


def test_draw_char_boxes_exclusive_edge_and_input_unchanged():
    plate = np.zeros((70, 220), np.uint8)
    ref = plate.copy()
    out = draw_char_boxes(plate, [CharBox(10, 10, 30, 50), CharBox(40, 10, 60, 50, line=1)])
    assert np.array_equal(plate, ref)
    assert out.shape == (70, 220, 3) and out.dtype == np.uint8
    assert tuple(out[10, 10]) == (0, 255, 0)
    assert tuple(out[49, 29]) == (0, 255, 0)      # x2-1, y2-1 là pixel cuối của hộp
    assert tuple(out[50, 30]) == (0, 0, 0)        # x2, y2 không bị vẽ
    assert tuple(out[10, 40]) == (0, 165, 255)    # dòng 1 màu cam
    assert tuple(out[30, 20]) == (0, 0, 0)        # bên trong hộp không tô


def test_draw_char_boxes_bgr_and_empty():
    plate = np.full((70, 220, 3), 7, np.uint8)
    out = draw_char_boxes(plate, [])
    assert np.array_equal(out, plate) and out is not plate


def test_save_debug_grid(tmp_path):
    dbg = DebugSteps()
    dbg.images["gray"] = np.full((40, 100), 100, np.uint8)
    dbg.images["edge"] = np.zeros((40, 100), np.uint8)
    dbg.images["color"] = np.full((60, 80, 3), 200, np.uint8)
    dbg.images["mask"] = np.random.rand(40, 100).astype(np.float32)
    path = str(tmp_path / "sub" / "dir" / "grid.png")
    save_debug_grid(dbg, path, cols=3)
    img = read_image(path)
    # 2 hàng x 3 cột, ô rộng 100, cao 60 + dải tiêu đề
    assert img.shape == (2 * (24 + 60), 300, 3)


def test_save_comparison(tmp_path):
    path = str(tmp_path / "out" / "cmp.png")
    save_comparison({"a": np.zeros((30, 50), np.uint8), "b": np.zeros((30, 50, 3), np.uint8)}, path)
    assert os.path.isfile(path)
    assert read_image(path).shape == (24 + 30, 100, 3)


def test_save_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        save_debug_grid(DebugSteps(), str(tmp_path / "x.png"))
    with pytest.raises(ValueError):
        save_comparison({}, str(tmp_path / "y.png"))


def test_uint8_binary_mask_becomes_visible_and_input_unchanged():
    mask = np.zeros((70, 220), np.uint8)
    mask[20:40, 20:60] = 1
    ref = mask.copy()
    out = draw_char_boxes(mask, [])
    assert np.array_equal(mask, ref)
    assert out.max() == 255 and tuple(out[30, 30]) == (255, 255, 255)
    bout = draw_char_boxes(mask.astype(bool), [])
    assert tuple(bout[30, 30]) == (255, 255, 255)


def test_bgra_input_works_and_not_mutated(tmp_path):
    bgra = np.full((70, 220, 4), 50, np.uint8)
    ref = bgra.copy()
    out = draw_char_boxes(bgra, [CharBox(10, 10, 30, 50)])
    assert out.shape == (70, 220, 3)
    dbg = DebugSteps()
    dbg.images["rgba"] = bgra
    dbg.images["gray"] = np.zeros((70, 220), np.uint8)
    path = str(tmp_path / "g.png")
    save_debug_grid(dbg, path, cols=2)
    assert read_image(path).shape[2] == 3
    assert np.array_equal(bgra, ref)
