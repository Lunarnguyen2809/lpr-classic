# OWNER: SHARED
"""Parse tên file CCPD2019 (xem SPEC mục 5.2 và 8.2)."""
from __future__ import annotations

import os

import numpy as np

PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def decode_plate(indices: list[int]) -> str | None:
    """Giải mã 7 chỉ số ký tự CCPD thành chuỗi biển.

    Input: list 7 số nguyên (tỉnh, chữ cái, 5 ký tự ADS).
    Output: chuỗi 7 ký tự, hoặc None nếu sai độ dài, chỉ số ngoài bảng, hay có ký tự "O" (không có ký tự).
    """
    if len(indices) != 7:
        return None
    tables = [PROVINCES, ALPHABETS] + [ADS] * 5
    chars = []
    for idx, table in zip(indices, tables):
        if not 0 <= idx < len(table):
            return None
        c = table[idx]
        if c == "O":
            return None
        chars.append(c)
    return "".join(chars)


def _point(s: str) -> tuple[int, int]:
    x, y = s.split("&")
    return int(x), int(y)


def parse_ccpd_filename(name: str) -> dict:
    """Parse tên file CCPD (có thể kèm đường dẫn và đuôi .jpg).

    Output: dict {bbox (x1,y1,x2,y2), quad (4×2 float32, TL,TR,BR,BL), text (str | None),
    area, tilt_h, tilt_v, brightness, blur}.
    `text` là None nếu biển chứa "O". Raise ValueError nếu tên sai định dạng.
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    fields = stem.split("-")
    if len(fields) != 7:
        raise ValueError(f"Tên file CCPD phải có 7 trường, nhận {len(fields)}: {name!r}")
    try:
        # Trường area là phần thập phân của tỉ lệ diện tích ("025" = 0.025, CCPD2019 thật dài hơn
        # như "033122605364" = 0.0331...). Đổi về ‰ nguyên.
        if not fields[0].isdigit():
            raise ValueError("area phải là chữ số")
        area = int(round(float("0." + fields[0]) * 1000))
        tilt_h, tilt_v = (int(v) for v in fields[1].split("_"))
        p1, p2 = (_point(p) for p in fields[2].split("_"))
        verts = [_point(p) for p in fields[3].split("_")]
        if len(verts) != 4:
            raise ValueError("cần 4 đỉnh")
        indices = [int(v) for v in fields[4].split("_")]
        brightness = int(fields[5])
        blur = int(fields[6])
    except ValueError as e:
        raise ValueError(f"Tên file CCPD sai định dạng: {name!r} ({e})") from e

    # CCPD lưu đỉnh theo BR, BL, TL, TR -> đổi sang TL, TR, BR, BL.
    br, bl, tl, tr = verts
    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    return {
        "bbox": (p1[0], p1[1], p2[0], p2[1]),
        "quad": quad,
        "text": decode_plate(indices),
        "area": area,
        "tilt_h": tilt_h,
        "tilt_v": tilt_v,
        "brightness": brightness,
        "blur": blur,
    }


def _find_subset_dir(raw_root: str, subset: str) -> str | None:
    """Tìm thư mục `ccpd_<subset>` trong raw_root (cho phép lồng thêm 1–2 cấp, ví dụ CCPD2019/)."""
    target = subset if subset.startswith("ccpd_") else f"ccpd_{subset}"
    for dirpath, dirnames, _ in os.walk(raw_root):
        if target in dirnames:
            return os.path.join(dirpath, target)
        depth = os.path.relpath(dirpath, raw_root).count(os.sep)
        if depth >= 2:
            dirnames[:] = []
    return None


def list_subset_files(raw_root: str, subset: str) -> list[str]:
    """Liệt kê đường dẫn ảnh của một subset (`base` hoặc `ccpd_base`), đã sắp xếp theo tên.

    Raise FileNotFoundError nếu không tìm thấy thư mục subset.
    """
    d = _find_subset_dir(raw_root, subset)
    if d is None:
        raise FileNotFoundError(f"Không thấy thư mục subset {subset!r} trong {raw_root!r}")
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS)]
    return sorted(files)
