# Ghi chú thí nghiệm

Ghi lại mọi tham số đã thử kèm kết quả (ngày, người thử, cấu hình, metric).

## Tiền xử lý

2026-11-05, tune (n=198), cấu hình localize = B + close [25,5] (fill 0.35, dens 0.10), chỉ đổi `preprocess.resize_width`:

| resize_width | P | R | F1 | R@5 |
|---|---|---|---|---|
| 640 | 0.639 | 0.591 | 0.614 | 0.727 |
| 720 (không đổi tỉ lệ) | 0.640 | 0.601 | 0.620 | 0.758 |
| 800 (mặc định, upscale) | 0.647 | 0.621 | 0.634 | 0.778 |

Giữ 800.

## Định vị

Ngày 2026-11-05, phương pháp A (`localize_edge`), tập **tune** (n=198: base 150, blur/challenge/db/tilt 12 mỗi nhóm), IoU ≥ 0.5, top-1 cho P/R/F1, Recall@k với k=5. Script đo nằm ngoài repo (scratchpad). Mặc định chung: sobel_ksize 3, open_kernel [3,3], area_ratio_range [0.002, 0.10], max_angle 30, resize 800 (trừ khi ghi khác). B = fill 0.35, density 0.10.

| # | Cấu hình (khác mặc định cũ) | P | R | F1 | R@5 | ms/ảnh | Ghi chú |
|---|---|---|---|---|---|---|---|
| 0 | mặc định cũ: close [17,3], fill 0.5, dens 0.15 | 0.570 | 0.348 | 0.433 | 0.384 | 146 | baseline |
| 1 | fill 0.35, dens 0.10 (close [17,3]) | 0.545 | 0.515 | 0.530 | 0.682 | 279 | |
| 2 | B + close [25,5] | 0.647 | 0.621 | 0.634 | 0.778 | 271 | |
| 3 | B + close [33,5] | 0.644 | 0.586 | 0.614 | 0.697 | 268 | |
| 4 | B + close [25,9] | 0.647 | 0.601 | 0.623 | 0.702 | 270 | |
| 5 | B + close [25,5] + aspect [1.8,6.5] | 0.641 | 0.621 | 0.631 | 0.783 | 342 | |
| 6 | B + close [25,5] + aspect [2.3,5.0] | 0.676 | 0.621 | 0.647 | 0.753 | 333 | |
| 7 | fill 0.30, dens 0.08 + close [25,5] | 0.617 | 0.611 | 0.614 | 0.788 | 342 | |
| 8 | fill 0.45, dens 0.12 + close [25,5] | 0.682 | 0.586 | 0.630 | 0.707 | 333 | |
| 9 | B + close [21,5] | 0.672 | 0.641 | 0.656 | 0.783 | 347 | |
| 10 | B + close [29,5] | 0.683 | 0.641 | 0.661 | 0.737 | 329 | |
| 11 | B + close [17,5] | 0.593 | 0.566 | 0.579 | 0.687 | 414 | |
| 12 | B + close [21,3] | 0.625 | 0.606 | 0.615 | 0.778 | 320 | |
| 13 | B + close [21,7] | 0.684 | 0.646 | 0.665 | 0.747 | 402 | |
| 14 | B + close [21,5] + aspect [2.3,5.0] | 0.681 | 0.636 | 0.658 | 0.758 | 409 | |
| 15 | B + close [21,5] + open [5,5] | 0.599 | 0.581 | 0.590 | 0.778 | 389 | |
| 16 | B + close [21,5] + max_angle 20 | 0.672 | 0.631 | 0.651 | 0.763 | 312 | |
| **17** | **close [21,5], fill 0.40, dens 0.12 (CHỐT)** | **0.703** | **0.646** | **0.674** | **0.768** | 281 | đã ghi vào default.yaml |

(ms/ảnh dao động 150-420 vì chạy 4 tiến trình song song; không so sánh tốc độ giữa các dòng.)

Kết quả chốt theo nhóm (tune):

| Nhóm | n | P | R | F1 | R@5 |
|---|---|---|---|---|---|
| base | 150 | 0.769 | 0.733 | 0.751 | 0.847 |
| blur | 12 | 0.400 | 0.333 | 0.364 | 0.583 |
| challenge | 12 | 0.636 | 0.583 | 0.609 | 0.750 |
| db | 12 | 0.455 | 0.417 | 0.435 | 0.417 |
| tilt | 12 | 0.286 | 0.167 | 0.211 | 0.333 |
| ALL | 198 | 0.703 | 0.646 | 0.674 | 0.768 |

Nhận xét: nới ngưỡng fill/density là bước tăng lớn nhất (mặt nạ biển thật chỉ lấp ~0.5, density ~0.2). Close ngang 21-29 rộng tốt hơn 17. Nhóm tilt và db yếu: tilt do mask nghiêng làm minAreaRect rộng ra, db do ngược sáng/chói. Chưa tune trên test.

## Nắn

2026-11-05. Đo trên tập tune, chỉ ảnh mà top-1 `localize_edge` có IoU ≥ 0.5 với GT (n=128), corner_error theo pixel ảnh gốc so với quad GT.
Phương pháp: `approxpoly` = mặt nạ HSV xanh dương (H 95-135, S ≥ 70, V ≥ 40) trong bbox nới 15%, đóng/mở morph, giới hạn trong quad ứng viên (chống rò nền xanh), contour lớn nhất chứa tâm, convex hull, approxPolyDP với epsilon tăng 1%..15% chu vi tới đúng 4 đỉnh. Lùi về quad minAreaRect khi: không ra 4 đỉnh, không lồi, diện tích <30% hay >300% bbox.

| subset | n | minAreaRect mean / median | approxpoly mean / median | fallback approxpoly | ms (minAR / approx) |
| --- | --- | --- | --- | --- | --- |
| base | 110 | 18.42 / 11.69 | 16.79 / 11.50 | 0.05 | 0.54 / 2.97 |
| blur | 4 | 14.98 / 10.93 | 15.45 / 11.06 | 0.50 | 0.81 / 3.63 |
| challenge | 7 | 29.47 / 23.25 | 21.82 / 15.52 | 0.00 | 0.64 / 3.36 |
| db | 5 | 32.90 / 30.02 | 27.41 / 26.78 | 0.80 | 0.42 / 5.42 |
| tilt | 2 | 33.96 / 33.96 | 28.84 / 28.84 | 0.50 | 2.95 / 8.34 |
| ALL | 128 | 19.72 / 13.37 | 17.62 / 12.20 | 0.09 | 0.59 / 3.19 |

Chọn `rectify.quad_method: approxpoly` (mean 19.7 -> 17.6 px). Mức lợi chỉ có ở trung bình, không có ý nghĩa theo từng ảnh (45 ảnh tốt hơn / 48 kém hơn / 35 hòa; Wilcoxon p = 0.46; CI 95% của chênh lệch trung bình [0.5, 4.0] px); hằng số đã chỉnh và đo trên cùng 128 ảnh tune; 107/128 ảnh có ít nhất một góc thật nằm ngoài quad ứng viên nên trần sai số do bước định vị quyết định, không phải approxpoly. Cải thiện nhỏ: sai số chủ yếu do bbox/quad ứng viên đã lệch (IoU ≥ 0.5 vẫn có thể lệch 10-20 px), mặt nạ chỉ sửa được trong quad ứng viên. Mặt nạ không giới hạn theo quad ứng viên cho mean 25.6 (tệ hơn minAreaRect) vì rò vào nền xanh/xám xanh. Nhóm tilt/db/blur mẫu quá ít (n ≤ 7), fallback cao, chưa kết luận được. Kiểm tra mắt: warp quad GT cho biển thẳng, lấp đầy, không bị lật; tốt hơn hẳn crop bbox ở ảnh nghiêng (ví dụ tilt_0004, challenge_0004).

## Nhị phân

2026-11-06, B. Thử nhanh trên 5 ảnh tune (mỗi subset 1 ảnh), crop = warp quad GT 220×70, ảnh xám, sau `ensure_text_white` + `clean_binary` open (2,2). Script: `python experiments/check_binarize.py [--C ..] [--block ..]`, ảnh lưới ở `results/figures/binarize/compare.png`.
Chưa có `segment.py` nên chỉ đếm thô "blob giống ký tự" (CCA, lọc theo `segment.char_height_range`, `char_aspect_range`, `min_area`) so với 7 ký tự thật. Đây là kiểm tra bằng mắt, không phải metric chính thức (đợi E5).

Lỗi phát hiện: bản đầu dùng `adaptiveThreshold(THRESH_BINARY, C>0)` thì với biển xanh chữ trắng, nền phẳng thành trắng và ký tự chỉ còn viền (base_0000 chỉ 3/7). Sửa: Otsu đoán cực tính, chữ sáng thì đảo xám, rồi `THRESH_BINARY_INV`. Thêm test nền phẳng phải đen.

Số blob / 7 (sau khi sửa), adaptive block_size 25:

| ảnh | otsu | adaptive C=3 | C=5 | C=7 (mặc định) |
|---|---|---|---|---|
| base_0000 | 7 | 7 | 7 | 7 |
| tilt_0004 | 6 | 6 | 7 | 7 |
| blur_0004 | 3 | 1 | 3 | 1 |
| db_0004 | 4 | 7 | 7 | 6 |
| challenge_0004 | 5 | 7 | 8 | 9 |

Nhận xét bằng mắt: adaptive giữ nét rời, không dính viền biển tốt hơn Otsu ở ảnh rõ/nghiêng/tối (Otsu dính viền dưới, đứt "0", "8"). Ảnh mờ (blur) adaptive vỡ vụn vì tương phản thấp; Otsu ra khối đặc nhưng các ký tự dính nhau. Blob thừa ở challenge là viền/ốc biển, để `segment` lọc. C=5 có vẻ cân bằng nhất nhưng n=5 quá ít; chưa đổi `default.yaml`, chốt method/C bằng E5 trên cả tập tune.

### Otsu vs adaptive trên cả tập tune, 2026-11-08, B

`python experiments/check_segment_all.py` (chỉ split tune, n=198). Cố định segment `cca`, crop warp quad GT, adaptive block 25 C 7, sau `ensure_text_white` + `clean_binary`. Bảng đầy đủ ở `results/tables/check_segment_all.csv`, ví dụ ở `results/figures/segment/compare_binarize.png`. Cột "số 1" = tỉ lệ vị trí chứa số "1" được tính đúng (trong ngoặc là số lần xuất hiện).

| binarize | subset | n | count | pos0 | pos1_6 | all_correct | số 1 |
|---|---|---|---|---|---|---|---|
| otsu | base | 150 | 0.627 | 0.860 | 0.848 | 0.440 | 0.05 (56) |
| otsu | blur | 12 | 0.167 | 0.417 | 0.389 | 0.083 | 0.00 (1) |
| otsu | challenge | 12 | 0.333 | 0.750 | 0.556 | 0.250 | 1.00 (2) |
| otsu | db | 12 | 0.167 | 0.250 | 0.306 | 0.083 | 0.22 (9) |
| otsu | tilt | 12 | 0.417 | 0.583 | 0.694 | 0.083 | 0.14 (7) |
| **otsu** | **ALL** | 198 | 0.540 | **0.773** | 0.760 | 0.364 | 0.11 (75) |
| adaptive | base | 150 | 0.620 | 0.873 | 0.891 | 0.440 | 0.02 (56) |
| adaptive | blur | 12 | 0.417 | 0.500 | 0.625 | 0.167 | 0.00 (1) |
| adaptive | challenge | 12 | 0.583 | 0.583 | 0.792 | 0.333 | 0.50 (2) |
| adaptive | db | 12 | 0.167 | 0.083 | 0.181 | 0.000 | 0.00 (9) |
| adaptive | tilt | 12 | 0.500 | 0.583 | 0.694 | 0.083 | 0.29 (7) |
| **adaptive** | **ALL** | 198 | **0.571** | 0.768 | **0.814** | **0.369** | 0.05 (75) |

Nhận xét:
- Gần như **hòa**: all_correct 0.369 vs 0.364. Tính theo cặp, adaptive đúng hết mà otsu sai ở 21 biển, ngược lại 20 biển (sign test p = 1.0). Không đủ bằng chứng để đổi `binarize.method` trong config; adaptive chỉ được chọn làm nền cho bước 2 vì nhỉnh hơn.
- Adaptive hơn rõ ở pos1_6 (0.814 vs 0.760) và ở blur/challenge: Otsu dùng một ngưỡng cho cả biển nên biển cháy sáng (base_0124) hoặc mờ (base_0057) dính ký tự Hán với "A", dính "FF" thành khối.
- Adaptive thua ở db (pos0 0.083, all 0.000): biển tối/ngược sáng có tương phản thấp hơn C=7 nên nét chữ vỡ vụn; nhiễu dọc nối vào ký tự thành khối rộng (base_0042) hoặc mép viền dọc thành hộp thừa (base_0011, 8/7).
- Trên 137 biển không có số "1", all_correct của hai cách bằng nhau (0.526). Số "1" (75 lần trên 61 biển) gần như luôn tính sai do metric (đã ghi ở mục Tách ký tự), nên trần all_correct thực tế bị chặn ~0.69.

## Tách ký tự

### CCA (`segment_cca` + `merge_split_boxes`), 2026-11-07, B

Cấu hình: crop warp quad GT 220×70, binarize theo config (otsu) + `ensure_text_white` + `clean_binary` (2,2), segment `cca`, tham số `segment.*` mặc định (height [0.4, 0.9], aspect [0.15, 1.0], min_area 30). Script: `python experiments/check_segment.py`, ảnh ở `results/figures/segment/cca.png`.

Hằng số hình học nội bộ trong `segment.py` (không phải tham số thí nghiệm): bỏ thành phần rộng > 30% biển nếu rỗng (fill < 0.35); bỏ mảnh dẹt (cao < 20% biển, rộng > 2× cao); bỏ thành phần có tâm y ngoài [0.25, 0.75] chiều cao (ốc vít, mép); gộp mảnh chồng ≥ 50% theo x; gộp mảnh hẹp (rộng/cao < aspect min) với mảnh kề nếu khe ≤ 6% chiều cao; merge_split: gộp khi có hộp < 0.75× trung vị, khe ≤ 0.3×, hộp gộp ≤ 1.4×; cắt hộp ≥ 1.7× trung vị.

10 biển tune (2 mỗi subset): count_match 3/10, vị trí đúng 45/70.

| image_id | hộp/7 | pos | ghi chú |
|---|---|---|---|
| base_0000 | 7 | 6/7 | trượt vị trí 6 là số "1" (xem dưới) |
| base_0002 | 7 | 7/7 | |
| tilt_0004 | 6 | 5/7 | "8" dính viền dưới, "1" trượt metric |
| tilt_0006 | 5 | 4/7 | nhòe chuyển động, khối dính |
| blur_0004 | 6 | 5/7 | khối "Y996S" dính, cắt đều không đủ chính xác |
| blur_0006 | 1 | 0/7 | Otsu ra một khối đặc, hỏng từ bước nhị phân |
| db_0004 | 7 | 5/7 | |
| db_0006 | 6 | 5/7 | ký tự Hán mảnh bị mất, "1" mảnh bị loại |
| challenge_0004 | 6 | 4/7 | "N256" dính nhau |
| challenge_0006 | 4 | 4/7 | "6MK" dính, biển mờ |

Cả tập tune (n=198), so với bản đầu xóa mọi thành phần rộng > 30% biển:

| subset | n | count (cũ → mới) | pos0 | pos1_6 | all_correct |
|---|---|---|---|---|---|
| base | 150 | 0.587 → 0.627 | 0.833 → 0.860 | 0.830 → 0.848 | 0.407 → 0.440 |
| blur | 12 | 0.167 → 0.167 | 0.500 → 0.417 | 0.319 → 0.389 | 0.083 → 0.083 |
| challenge | 12 | 0.250 → 0.333 | 0.583 → 0.750 | 0.486 → 0.556 | 0.250 → 0.250 |
| db | 12 | 0.167 → 0.167 | 0.167 → 0.250 | 0.306 → 0.306 | 0.083 → 0.083 |
| tilt | 12 | 0.333 → 0.417 | 0.500 → 0.583 | 0.667 → 0.694 | 0.083 → 0.083 |
| ALL | 198 | 0.500 → 0.540 | 0.737 → 0.773 | 0.737 → 0.760 | 0.338 → 0.364 |

Thay đổi: khối đặc rộng (nhiều ký tự dính) không còn bị xóa như viền mà giữ lại để `merge_split_boxes` cắt. Hằng số chọn trên chính tập tune, chưa đo test.

**Vấn đề với số "1" (đã sửa ở mục "Sửa metric position_hits" bên dưới):** trên 150 biển base, vị trí chứa "1" chỉ đúng 3/56 (5%) so với 889/994 (89%) ở ký tự khác. Nguyên nhân chính không phải CCA mà là metric SPEC 9.3: hộp "1" rộng ~9 px, khuôn rộng 22.5 px nên IoU trục x ≈ 0.4 < 0.5 dù hộp nằm đúng giữa khuôn. Phụ: 11% nét "1" có rộng/cao < 0.15 (trung vị 0.21) nên bị `char_aspect_range` loại. Hướng sửa có thể: đổi position_hits sang "tâm hộp nằm trong khoảng khuôn" hoặc IoU so với hộp hẹp hơn (đổi định nghĩa SPEC 9.3), và/hoặc hạ aspect min xuống ~0.1 (config SHARED).

### Sửa metric position_hits cho số "1" (luật tâm), 2026-11-08

Đổi `position_hits` (SPEC 9.3): trúng nếu (a) IoU-x ≥ 0.5, HOẶC (b) hộp hẹp (0 < rộng ≤ 0.6× rộng khuôn) có tâm trong khuôn [x1, x2). Hằng số `evaluate.char_center_max_width_ratio` = 0.6 trong config. Luật (b) chỉ dành cho hộp hẹp như số "1"; hộp rộng bình thường vẫn phải đạt IoU 0.5 (ngưỡng 1.5× ban đầu nới cả hộp bình thường lệch tâm ~11 px và cho mảnh vỡ trúng). Đồng thời hạ `segment.char_aspect_range[0]` 0.15 → 0.10. Đo trên tune n=198, pipeline: nắn quad GT -> otsu -> cca. Chạy lại: `PYTHONPATH=src python experiments/measure_segment_metric.py [--aspect-min 0.12]`.

| cấu hình | all_correct | count | pos0 | pos1_6 | "1" trúng (base) |
|---|---|---|---|---|---|
| trước (main): IoU, aspect 0.15 | 0.364 | 0.540 | 0.773 | 0.760 | 3/56 |
| luật tâm ratio 0.6, aspect 0.15 | 0.490 | 0.540 | 0.788 | 0.812 | 45/56 |
| luật tâm ratio 0.6, aspect 0.12 | 0.505 | 0.545 | 0.788 | 0.820 | 52/56 |
| luật tâm ratio 0.6, aspect 0.10 (chọn) | 0.515 | 0.556 | 0.788 | 0.820 | 53/56 |

Chọn 0.10 vì all_correct cao hơn 0.12 (0.515 vs 0.505). Hằng số chọn trên chính tập tune, chưa đo test. Lưu ý: pos0/pos1_6 là "có hộp nào trúng khuôn" (không ghép 1-1), nên mảnh vỡ cũng có thể được tính trúng; all_correct được chặn bởi count == 7 nên ít bị ảnh hưởng.

**Cho Bích:** `_merge_narrow_neighbors` trong `segment.py` dùng chung `char_aspect_range[0]` làm ngưỡng "mảnh hẹp", nên hạ aspect min cũng đổi ngưỡng gộp mảnh hẹp. Gợi ý tách thành hằng số riêng (reviewer đo chênh lệch không đáng kể: count 0.545 vs 0.551). Em không sửa `segment.py`.

### Kiểm tra khuôn `char_template_mm`

5 biển base cho đúng 7 hộp (base_0000, 0002, 0009, 0011, 0015). Lệch = tâm hộp − tâm khuôn, 1 px = 2 mm:

| vị trí | khuôn (px) | lệch TB px | min..max px | lệch TB mm | rộng hộp TB px | rộng khuôn px |
|---|---|---|---|---|---|---|
| 0 | 7.8–30.2 | −3.0 | −4.5..0.0 | −6.0 | 26.0 | 22.5 |
| 1 | 36.2–58.8 | −1.4 | −2.0..−0.5 | −2.8 | 22.6 | 22.5 |
| 2 | 75.8–98.2 | −2.6 | −3.5..−2.0 | −5.2 | 24.0 | 22.5 |
| 3 | 104.2–126.8 | −2.7 | −4.0..−1.5 | −5.4 | 22.8 | 22.5 |
| 4 | 132.8–155.2 | −3.3 | −4.5..−2.0 | −6.6 | 23.0 | 22.5 |
| 5 | 161.2–183.8 | −3.4 | −4.0..−2.5 | −6.8 | 23.4 | 22.5 |
| 6 | 189.8–212.2 | −2.9 | −5.0..1.0 | −5.8 | 17.4 | 22.5 |

Trung bình |lệch| 2.8 px, lớn nhất 5.0 px. Khuôn khớp về khoảng cách giữa các ký tự (lệch gần như đều nhau), nhưng toàn bộ ký tự thật lệch trái ~3 px (~6 mm) so với khuôn: có thể do quad GT của CCPD ôm cả viền hoặc lệch nhẹ. Với hộp rộng ~23 px, lệch 3 px vẫn cho IoU x ≈ 0.77 nên chưa làm trượt position_hits. **Chưa sửa config** (SHARED); nếu muốn chỉnh thì trừ ~6 mm ở mọi khoảng, cần đo thêm trên nhiều biển.

### Projection + so sánh cca/projection, crop bbox/quad, 2026-11-08, B

`split_lines`: chiếu ngang, hàng > 3% trắng là chữ, > 85% là viền; nối khe ≤ 5% chiều cao, bỏ dòng < 20%. `segment_projection`: bỏ viền/ốc vít (cùng luật CCA) → profile cột trong từng dòng, cột ≥ 10% chiều cao dòng là cột chữ → nối dải cách nhau < `min_gap` (2), nối dải hẹp với dải kề nếu khe ≤ 6% chiều cao (ký tự Hán vỡ) → bỏ dải < `min_width` (4) → dải ≥ 1.7× trung vị cắt tại cột ít trắng nhất trong ± 0.3× trung vị quanh vị trí cắt dự kiến → co hộp theo y → lọc chiều cao, tỉ lệ, diện tích. CCPD toàn biển 1 dòng nên split_lines chỉ kiểm 2 dòng bằng test giả.

Cùng script `experiments/check_segment_all.py`, tập tune n=198.

**cca vs projection** (cố định adaptive, crop quad):

| segment | subset | n | count | pos0 | pos1_6 | all_correct | số 1 |
|---|---|---|---|---|---|---|---|
| cca | base | 150 | 0.620 | 0.873 | 0.891 | 0.440 | 0.02 |
| cca | blur | 12 | 0.417 | 0.500 | 0.625 | 0.167 | 0.00 |
| cca | challenge | 12 | 0.583 | 0.583 | 0.792 | 0.333 | 0.50 |
| cca | db | 12 | 0.167 | 0.083 | 0.181 | 0.000 | 0.00 |
| cca | tilt | 12 | 0.500 | 0.583 | 0.694 | 0.083 | 0.29 |
| **cca** | **ALL** | 198 | 0.571 | 0.768 | **0.814** | 0.369 | 0.05 |
| projection | base | 150 | 0.620 | 0.907 | 0.879 | 0.440 | 0.04 |
| projection | blur | 12 | 0.417 | 0.333 | 0.611 | 0.167 | 0.00 |
| projection | challenge | 12 | 0.583 | 0.583 | 0.778 | 0.333 | 0.50 |
| projection | db | 12 | 0.250 | 0.167 | 0.208 | 0.000 | 0.00 |
| projection | tilt | 12 | 0.583 | 0.667 | 0.681 | 0.083 | 0.00 |
| **projection** | **ALL** | 198 | **0.581** | **0.793** | 0.804 | 0.369 | 0.04 |

**Hòa**: all_correct bằng nhau 0.369; projection đúng hết mà cca sai 13 biển, ngược lại 13 biển (p = 1.0). Script chọn projection chỉ nhờ count_match nhỉnh hơn (0.581 vs 0.571). Mỗi cách thua theo kiểu riêng (`results/figures/segment/compare_segment.png`):
- CCA thua khi ký tự dính với nhiễu dọc hoặc mép viền dọc: base_0042 "06" + chấm nhiễu thành khối, cắt đều ra 3 hộp (9/7); base_0055, base_0094 mép phải thành hộp thừa (8/7); base_0090 chữ Hán vỡ nhiều mảnh nhỏ bị lọc mất (5/7). Projection không bị vì chấm/vệt mảnh không đủ 10% chiều cao cột, và các mảnh chữ Hán vẫn nằm chung một dải cột.
- Projection thua khi ký tự có cột giữa thưa: "H" (base_0033) chỉ có vạch ngang mảnh ở giữa, cột giữa dưới ngưỡng 10% nên bị cắt thành hai nét đứng hẹp và bị loại (6/7); "J" (base_0072) hộp lệch vì phần móc dưới thưa.

**crop bbox vs quad** (cố định adaptive, projection):

| crop | subset | n | count | pos0 | pos1_6 | all_correct |
|---|---|---|---|---|---|---|
| bbox | base | 150 | 0.540 | 0.580 | 0.624 | 0.173 |
| bbox | blur | 12 | 0.417 | 0.250 | 0.500 | 0.083 |
| bbox | challenge | 12 | 0.333 | 0.333 | 0.569 | 0.167 |
| bbox | db | 12 | 0.000 | 0.000 | 0.014 | 0.000 |
| bbox | tilt | 12 | 0.000 | 0.083 | 0.167 | 0.000 |
| **bbox** | **ALL** | 198 | 0.455 | 0.480 | 0.549 | 0.146 |
| **quad** | **ALL** | 198 | **0.581** | **0.793** | **0.804** | **0.369** |

**Quad thắng rõ**: 50 biển chỉ quad đúng hết, 6 biển chỉ bbox đúng (sign test p < 0.001); thắng ở mọi subset, chênh nhất ở tilt và db (bbox count = 0). Lý do: bbox là hình chữ nhật thẳng ôm biển nghiêng nên lẫn nền và viền, chữ bị xiên và lệch khỏi khuôn cố định (base_0002 bbox: đủ 7 hộp nhưng chỉ 3/7 vị trí). Ca bbox thắng (base_0072, 0155) là do quad GT hơi rộng nên mép viền thành hộp thừa.

Tóm lại trên tune: crop **quad** chắc chắn; binarize và segment **chưa phân thắng thua** (chênh ≤ 1 biển). Không đổi config. Hằng số trong segment.py chọn trên chính tune nên số liệu hơi lạc quan; kết luận cuối dựa vào E5/E6 trên test. Điểm nghẽn lớn nhất hiện tại là metric số "1" (trần all_correct ~0.69), tiếp theo là nhóm db (tối, tương phản thấp).

## Đánh giá
