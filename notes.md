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

## Tách ký tự

## Đánh giá
