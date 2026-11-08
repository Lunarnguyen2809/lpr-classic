# OWNER: A
"""Vẽ ứng viên, hộp ký tự và lưới ảnh debug (SPEC mục 8.12)."""
from __future__ import annotations

import cv2
import numpy as np

from lpr.io_utils import write_image
from lpr.types import BBox, Candidate, CharBox, DebugSteps

RED = (0, 0, 255)       # ứng viên
GREEN = (0, 255, 0)     # bbox thật / hộp ký tự dòng 0
ORANGE = (0, 165, 255)  # hộp ký tự dòng 1 trở đi
TITLE_H = 24            # chiều cao dải tiêu đề trong lưới / so sánh


def _to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Bản sao BGR uint8 của ảnh xám/BGR/BGRA.

    Ảnh bool, float hoặc nhị phân 0/1 (uint8 có max <= 1) được co giãn về 0..255; BGRA bỏ kênh alpha.
    """
    out = img
    if out.dtype == np.bool_:
        out = out.astype(np.uint8) * 255
    elif out.dtype != np.uint8:
        lo, hi = float(out.min()), float(out.max())
        out = ((out.astype(np.float32) - lo) * (255.0 / (hi - lo if hi > lo else 1.0))).astype(np.uint8)
    elif out.size and out.max() <= 1:
        out = out * 255  # mask uint8 0/1 -> 0/255 (tạo bản mới, không sửa input)
    if out.ndim == 2:
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if out.ndim == 3 and out.shape[2] == 4:
        return cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
    return out.copy()


def draw_candidates(img: np.ndarray, cands: list[Candidate], gt: BBox | None = None) -> np.ndarray:
    """Vẽ quad/bbox các ứng viên kèm điểm, và bbox thật (nếu có) lên bản sao ảnh BGR."""
    out = _to_bgr_uint8(img)
    for i, c in enumerate(cands):
        thick = 3 if i == 0 else 1  # ứng viên đầu danh sách (top-1) vẽ đậm hơn
        cv2.polylines(out, [np.round(c.quad).astype(np.int32).reshape(-1, 1, 2)], True, RED, thick)
        x1, y1, x2, y2 = c.bbox
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), RED, 1)  # x2, y2 không bao gồm
        cv2.putText(out, f"{c.score:.2f}", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
    if gt is not None:
        x1, y1, x2, y2 = gt
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), GREEN, 2)
    return out


def draw_char_boxes(plate: np.ndarray, boxes: list[CharBox]) -> np.ndarray:
    """Vẽ hộp ký tự lên bản sao ảnh biển (BGR hoặc xám). Output: ảnh BGR."""
    out = _to_bgr_uint8(plate)
    for b in boxes:
        color = GREEN if b.line == 0 else ORANGE
        cv2.rectangle(out, (b.x1, b.y1), (b.x2 - 1, b.y2 - 1), color, 1)  # x2, y2 không bao gồm
    return out


def _titled_tile(img: np.ndarray, title: str, cell_w: int, cell_h: int) -> np.ndarray:
    """Đặt ảnh (giữ tỉ lệ) vào ô cell_w x cell_h, thêm dải tiêu đề phía trên."""
    img = _to_bgr_uint8(img)
    h, w = img.shape[:2]
    s = min(cell_w / w, cell_h / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    tile = np.full((TITLE_H + cell_h, cell_w, 3), 40, np.uint8)
    tile[TITLE_H:TITLE_H + nh, :nw] = img
    cv2.putText(tile, title, (4, TITLE_H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _save_tiles(images: dict[str, np.ndarray], path: str, cols: int) -> None:
    if not images:
        raise ValueError("Không có ảnh nào để ghép")
    cell_w = max(im.shape[1] for im in images.values())
    cell_h = max(im.shape[0] for im in images.values())
    tiles = [_titled_tile(im, str(name), cell_w, cell_h) for name, im in images.items()]
    cols = max(1, min(cols, len(tiles)))
    blank = np.full_like(tiles[0], 40)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        row += [blank] * (cols - len(row))  # hàng cuối thiếu ô thì đệm ô trống
        rows.append(np.hstack(row))
    write_image(path, np.vstack(rows))


def save_debug_grid(debug: DebugSteps, path: str, cols: int = 3) -> None:
    """Ghép các ảnh trong DebugSteps thành lưới `cols` cột, có tiêu đề từng bước, rồi lưu ra `path`.

    Raise ValueError nếu `debug.images` rỗng.
    """
    _save_tiles(debug.images, path, cols)


def save_comparison(images: dict[str, np.ndarray], path: str) -> None:
    """Đặt nhiều ảnh cạnh nhau, mỗi ảnh có tiêu đề là khóa của dict, rồi lưu ra `path`.

    Raise ValueError nếu `images` rỗng.
    """
    _save_tiles(images, path, cols=len(images))
