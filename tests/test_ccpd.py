import numpy as np
import pytest

from lpr.data.ccpd import decode_plate, parse_ccpd_filename

EXAMPLE = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"


def test_parse_example():
    info = parse_ccpd_filename(EXAMPLE)
    assert info["bbox"] == (154, 383, 386, 473)
    assert info["text"] == "皖AY339S"
    q = info["quad"]
    assert q.dtype == np.float32 and q.shape == (4, 2)
    assert tuple(q[0]) == (154, 383)  # TL
    assert tuple(q[1]) == (363, 402)  # TR
    assert tuple(q[2]) == (386, 473)  # BR
    assert tuple(q[3]) == (177, 454)  # BL
    assert (info["area"], info["tilt_h"], info["tilt_v"]) == (25, 95, 113)
    assert (info["brightness"], info["blur"]) == (37, 15)


def test_parse_long_area_field():
    # Tên file thật trong CCPD2019 có trường area dài (tỉ lệ 0.0331...)
    name = ("033122605364-90_91-215&535_551&643-548&650_202&648_208&531_554&533"
            "-0_0_1_33_29_24_25-177-51.jpg")
    info = parse_ccpd_filename(name)
    assert info["area"] == 33
    assert info["text"] == "皖AB9501"


def test_parse_with_directory():
    info = parse_ccpd_filename(f"data/raw/CCPD2019/ccpd_base/{EXAMPLE}")
    assert info["text"] == "皖AY339S"


@pytest.mark.parametrize("bad", [
    "abc.jpg",
    "025-95_113-154&383_386&473-386&473_177&454_154&383-0_0_22_27_27_33_16-37-15.jpg",  # 3 đỉnh
    "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37.jpg",  # 6 trường
    "025-95_113-154x383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg",
])
def test_parse_bad_name(bad):
    with pytest.raises(ValueError):
        parse_ccpd_filename(bad)


def test_decode_plate_invalid():
    assert decode_plate([0, 0, 22, 27, 27, 33]) is None       # thiếu ký tự
    assert decode_plate([33, 0, 22, 27, 27, 33, 16]) is None  # tỉnh = "O"
    assert decode_plate([0, 24, 22, 27, 27, 33, 16]) is None  # chữ cái = "O"
    assert decode_plate([0, 0, 22, 27, 27, 33, 34]) is None   # ADS = "O"
    assert decode_plate([0, 0, 22, 27, 27, 33, 99]) is None   # ngoài bảng
