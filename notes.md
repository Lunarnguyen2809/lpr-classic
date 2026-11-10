# Ghi chú thí nghiệm

Ghi lại mọi tham số đã thử kèm kết quả (ngày, người thử, cấu hình, metric).

## Hệ màu

### E1: độ tách biển / nền theo kênh màu, 2026-11-08, Bích

`python experiments/run_all.py --exp color` (luôn dùng tune n=198, ~20 giây). Độ tách = khoảng cách Bhattacharyya D = sqrt(1 − Σ sqrt(p·q)) giữa histogram 64 bin trong bbox biển và phần ảnh còn lại; 0 = giống hệt, 1 = không chồng lấn. Bảng `results/tables/e1_color.csv`, hình `results/figures/e1_hist.png`.

| hạng | kênh | base | blur | challenge | db | tilt | **ALL** (std) |
|---|---|---|---|---|---|---|---|
| 1 | b (Lab) | 0.687 | 0.629 | 0.670 | 0.481 | 0.466 | **0.657** (0.130) |
| 2 | H (HSV) | 0.681 | 0.621 | 0.697 | 0.500 | 0.462 | **0.654** (0.141) |
| 3 | S | 0.525 | 0.558 | 0.569 | 0.385 | 0.387 | 0.513 (0.135) |
| 4 | V | 0.485 | 0.537 | 0.527 | 0.496 | 0.358 | 0.484 (0.121) |
| 5 | a | 0.502 | 0.414 | 0.451 | 0.328 | 0.347 | 0.474 (0.129) |
| 6 | gray | 0.474 | 0.495 | 0.552 | 0.451 | 0.370 | 0.472 (0.106) |
| 7 | L | 0.465 | 0.491 | 0.548 | 0.468 | 0.357 | 0.465 (0.112) |

Nhận xét:
- **b và H tách tốt nhất (~0.65), S kế tiếp (0.51); gray, L, V, a gần như hòa (~0.47).** Biển CCPD màu xanh dương: trong Lab xanh dương nằm ở b < 128 (phía xanh), trong HSV dồn vào H ≈ 100–120, nên hai kênh này gom gọn vùng biển và lệch khỏi nền. Gray / L chỉ đo độ sáng, mà biển xanh đậm có độ sáng lẫn với thân xe, đường.
- **Ổn định theo số bin:** đo lại với 32 và 128 bin, 3 hạng đầu không đổi (b ≈ H > S). Bốn kênh cuối đổi chỗ cho nhau, chênh < 0.02, coi như hòa.
- **Theo subset:** db và tilt tách kém nhất ở mọi kênh (biển tối hoặc chói, bbox nghiêng ôm cả nền). Trên db, V ≈ H (0.50) vì biển tối làm màu mất bão hòa.
- **Hình histogram:** H ngoài biển có đỉnh lớn ở 0 (pixel xám/trắng, S ≈ 0 nên H không xác định → 0). Gray / L / V có răng cưa đều giữa các bin kề nhau ở cả hai đường: do lượng tử hóa khi giải nén JPEG và đổi hệ màu, không phải do biển; không đổi thứ hạng ở trên.
- **Hệ quả cho pipeline:** `localize_edge` hiện dùng gray (Sobel-x trên ảnh xám). Kênh b hoặc H có thể giúp lọc ứng viên theo màu (biển xanh). Nhưng biển VN nền trắng / vàng, nên đây là đặc trưng riêng của CCPD, cần ghi vào phần hạn chế. Đề xuất cho Châu Anh (#10): thử thêm độ đo "tỉ lệ pixel xanh" (H trong [95, 135], S đủ cao) khi chấm điểm ứng viên. `rectify.approxpoly` đang làm việc tương tự.
- Hạn chế: "ngoài biển" là toàn ảnh còn lại, nên độ tách phản ánh biển so với toàn cảnh, không phải so với vùng ngay sát biển. Trong bbox có cả ký tự trắng và viền nên histogram trong biển có hai đỉnh.

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

### Issue #10: recall ứng viên của `localize_edge`, 2026-11-09

Mục tiêu: biển thật bị bỏ sót vì vùng biển vỡ mảnh sau morph hoặc bị ứng viên nhỏ (logo, chữ) xếp trên. Đo bằng `PYTHONPATH=src python experiments/measure_localize_recall.py [--split test] [--set key=json ...]`. `cand_recall` = có ít nhất một ứng viên (trong top_k) IoU ≥ 0.5 với GT; `top1` = Recall SPEC 9.1; P = TP/(TP+FP) với top-1. Chọn tham số CHỈ trên tune (n=198); test (n=462) đo đúng một lần cho cấu hình chốt.

Quan sát: bộ lọc mới là nút thắt (tăng `top_k` 5 → 10 không đổi recall). Vùng top-1 sai thường là blob rất nhỏ (area_ratio 0.002-0.003) trong khi biển thật 0.015-0.05. Diện tích bbox GT trên tune: min 0.0054, p5 0.0116, p50 0.027, p98 0.092, max 0.153; aspect bbox GT p5 1.43, p95 3.68 (biển nghiêng).

Quét trên tune (ALL, đổi một thứ so với cấu hình cũ trừ khi ghi khác; ms/ảnh dao động do máy tải, không so giữa các dòng):

| cấu hình | cand_recall | top1 | P |
|---|---|---|---|
| cũ (close [21,5], aspect [2,6], area [0.002,0.10]) | 0.768 | 0.646 | 0.703 |
| top_k 10 | 0.768 | 0.646 | 0.703 |
| close thêm [33,7] | 0.768 | 0.646 | 0.688 |
| close thêm [13,3] + [33,7] | 0.828 | 0.652 | 0.665 |
| close thêm [13,3] + [33,7] + [45,9] | 0.823 | 0.636 | 0.649 |
| fill 0.3, density 0.08 | 0.783 | 0.636 | 0.653 |
| aspect [1.5,7.0] | 0.783 | 0.652 | 0.690 |
| area [0.001,0.2] | 0.768 | 0.540 | 0.557 |
| max_angle 45 | 0.768 | 0.621 | 0.668 |
| close [21,9] | 0.697 | 0.616 | 0.705 |
| area [0.005,0.16] | 0.758 | 0.697 | 0.793 |
| area [0.008,0.16] | 0.753 | 0.712 | 0.829 |
| area [0.01,0.16] | 0.722 | 0.687 | 0.834 |
| area [0.008,0.16] + 2 kernel thêm | 0.818 | 0.747 | 0.787 |
| **area [0.008,0.16] + 2 kernel thêm + aspect [1.5,7.0] (CHỐT)** | **0.828** | **0.747** | **0.775** |
| chốt + fill 0.3, density 0.08 (không chọn) | 0.843 | 0.758 | 0.761 |

Thay đổi (config + code, đều có ghi trong SPEC 7 và 8.6):
- `localize.extra_close_kernels: [[13,3],[33,7]]`: đa tỉ lệ. Chạy thêm morph với mỗi kernel, gộp ứng viên, NMS bbox (`nms_iou` 0.5), giữ top_k. Biển vỡ ở kernel này có thể liền ở kernel khác. Kernel đã tính theo ảnh resize 800 px nên không phụ thuộc kích thước ảnh gốc. `[]` tắt (hành vi cũ). `close_kernel` [21,5] vẫn là kernel chính (và là mặt nạ ghi vào debug "morph").
- `area_ratio_range` [0.002,0.10] → [0.008,0.16]: cận dưới theo phân bố GT (loại logo/chữ nhỏ), cận trên theo max GT.
- `aspect_range` [2,6] → [1.5,7]: minAreaRect của vùng cạnh nghiêng hoặc dính viền dài/ngắn hơn biển thật.
- Không đổi fill/density: nới tiếp chỉ được +1.5 pt recall nhưng P giảm (0.775 → 0.761), trong nhiễu của subset 12 ảnh, nên giữ ngưỡng.

Trước → sau, tune (n=198), cấu hình cũ → chốt:

| subset | n | cand_recall | top1 (Recall@1) | P |
|---|---|---|---|---|
| base | 150 | 0.847 → 0.907 | 0.733 → 0.820 | 0.769 → 0.826 |
| blur | 12 | 0.583 → 0.583 | 0.333 → 0.417 | 0.400 → 0.455 |
| challenge | 12 | 0.750 → 0.750 | 0.583 → 0.750 | 0.636 → 0.818 |
| db | 12 | 0.417 → 0.500 | 0.417 → 0.417 | 0.455 → 0.500 |
| tilt | 12 | 0.333 → 0.500 | 0.167 → 0.500 | 0.286 → 0.600 |
| ALL | 198 | 0.768 → 0.828 | 0.646 → 0.747 | 0.703 → 0.775 |

Test (n=462), đo một lần, không chỉnh lại sau đó:

| subset | n | cand_recall | top1 (Recall@1) | P |
|---|---|---|---|---|
| base | 350 | 0.729 → 0.874 | 0.609 → 0.751 | 0.644 → 0.771 |
| blur | 28 | 0.500 → 0.643 | 0.357 → 0.643 | 0.400 → 0.750 |
| challenge | 28 | 0.571 → 0.500 | 0.429 → 0.464 | 0.500 → 0.619 |
| db | 28 | 0.464 → 0.429 | 0.321 → 0.321 | 0.409 → 0.409 |
| tilt | 28 | 0.357 → 0.571 | 0.286 → 0.429 | 0.348 → 0.500 |
| ALL | 462 | 0.667 → 0.792 | 0.545 → 0.682 | 0.593 → 0.729 |

Thời gian (một tiến trình, máy đang tải nên chỉ mang tính tương đối): tune 309 → 426 ms/ảnh, test 269 → 396 ms/ảnh (khoảng +40-50%, do thêm 2 lượt morph + contour + features). Số ứng viên TB mỗi ảnh gần như không đổi (2.0 → 2.05 trên tune).

Trần recall do `area_ratio_range` cận dưới 0.008 (đã kiểm bằng script đếm trên GT bbox): tune 3/198 và test 16/462 bbox GT có area_ratio < 0.008 (test min 0.0032, tune min 0.0054), và 1 bbox GT test > 0.16 (max 0.182). Bộ lọc dùng diện tích minAreaRect của ứng viên (≤ diện tích bbox) nên mất mát thực có thể lớn hơn các con số này.

Còn yếu: db (ngược sáng/chói) và challenge không cải thiện, n mỗi nhóm nhỏ (12 tune, 28 test) nên chênh lệch ±1-2 ảnh. Chưa thử bù nghiêng trước morph (xoay mặt nạ) cho tilt. Cải thiện top-1 chủ yếu nhờ cận dưới area; muốn nâng thêm cần xếp hạng tốt hơn (score hiện chỉ gồm density, fill, aspect).

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

### Chạy lại sau khi merge #6 (luật tâm, aspect min 0.10), 2026-11-08, Bích

Cùng script, tập tune n=198, metric mới. Bảng trên giữ lại để so trước/sau.

| so sánh | cấu hình | count | pos0 | pos1_6 | all_correct | số "1" trúng | đúng riêng |
|---|---|---|---|---|---|---|---|
| binarize (cca, quad) | otsu | 0.556 | 0.788 | 0.820 | 0.515 | 0.88 | 32 |
| | adaptive | 0.566 | 0.818 | **0.896** | 0.515 | 0.88 | 32 |
| segment (adaptive, quad) | **cca** | **0.566** | 0.818 | 0.896 | **0.515** | 0.88 | 18 |
| | projection | 0.545 | **0.828** | **0.906** | 0.490 | 0.89 | 13 |
| crop (adaptive, cca) | bbox | 0.409 | 0.475 | 0.641 | 0.192 | 0.61 | 11 |
| | **quad** | **0.566** | **0.818** | **0.896** | **0.515** | 0.88 | 75 |

Theo subset (adaptive + cca + quad): base 0.613, blur 0.250, challenge 0.167, db 0.083, tilt 0.333.

Nhận xét:
- Luật tâm sửa đúng chỗ: số "1" trúng 0.05 → 0.88, all_correct 0.369 → 0.515.
- **Quad vẫn thắng rõ** (75 vs 11 biển).
- **Otsu vs adaptive vẫn hòa** (0.515 = 0.515, 32 vs 32 biển). Adaptive hơn ở pos1_6 (0.896 vs 0.820) nhưng không thành all_correct vì sai số hộp.
- **CCA nhỉnh hơn projection** (0.515 vs 0.490, 18 vs 13 biển, sign test p ≈ 0.47): chưa đủ để gọi là thắng. Projection trúng vị trí nhiều hơn (pos1_6 0.906) nhưng hay thừa/thiếu hộp (count 0.545) do cắt "H" thành hai nét.
- Không đổi config. Điểm nghẽn còn lại: nhóm db (all 0.083) và count_match (~0.56, tức gần nửa biển vẫn sai số hộp).

### Issue #14: tách ngưỡng gộp mảnh hẹp + 2 gợi ý phụ, 2026-11-08, Bích

Ba thay đổi trong `segment.py`, đo trên tune n=198, crop quad GT, metric luật tâm:
1. **`NARROW_ASPECT = 0.15`** riêng cho luật gộp mảnh hẹp (cả CCA và projection), không dùng chung `char_aspect_range[0]` (đã hạ xuống 0.10 ở #6).
2. **Số phần cắt tính theo khe:** `round((w + khe) / (trung vị + khe))` thay cho `round(w / trung vị)`, khe = trung vị khe giữa các hộp kề. Áp cho `merge_split_boxes` và `_cut_at_valleys`.
3. **Lọc mảnh dẹt sau khi gộp mảnh chồng dọc** (chỉ CCA): nét ngang chữ Hán được gộp vào ký tự trước, mảnh dẹt đứng một mình mới bị bỏ.

all_correct (trong ngoặc: số biển đúng thêm / mất so với chỉ làm bước 1):

| binarize + segment | trước #14 (dùng chung 0.10) | (1) | (1)+(2) | (1)+(3) | (1)+(2)+(3), **chọn** |
|---|---|---|---|---|---|
| otsu + cca | 0.515 | 0.515 | 0.540 (+5/−0) | 0.515 | **0.540** (+5/−0) |
| otsu + projection | 0.566 | 0.596 | 0.596 | 0.596 | **0.596** |
| adaptive + cca | 0.515 | 0.535 | 0.551 (+3/−0) | 0.551 (+4/−1) | **0.566** (+7/−1) |
| adaptive + projection | 0.490 | 0.530 | 0.530 | 0.530 | **0.530** |

- Bước 1 tự nó đã có lợi: hạ min 0.10 trước đây cũng hạ ngưỡng "mảnh hẹp", nên mảnh chữ Hán rộng/cao 0.10–0.15 không còn được gộp. Projection được lợi nhiều nhất (+0.03 / +0.04).
- Bước 2 và 3 chỉ tác động lên CCA. Projection đã cắt ở thung lũng và không có bước gộp mảnh chồng.
- Không tổ hợp nào giảm all_correct. Chênh nhỏ (p ≈ 0.07 cho adaptive + cca, +7/−1), hằng số chọn trên tune, kết luận cuối vẫn dựa vào test.
- Test mới: cả 3 test đều fail với code cũ và pass với code mới.

## Đánh giá

### E0: xuất predictions trên tập test, 2026-11-08, Bích

`python experiments/run_all.py --exp predict --split test` (462 ảnh, ~1.5 phút). Ghi `results/predictions/localization.csv` (4158 dòng) và `segmentation.csv` (7392 dòng = 462 × 16), cột đúng schema SPEC 10.1 / 10.2.

- localization: method `edge` × denoise {none, gaussian, median, bilateral}, mỗi ảnh tối đa top_k = 5 dòng (rank 1..5). Ảnh không có ứng viên vẫn có 1 dòng `rank = 0`, tọa độ trống, `is_tp = False`. `edge_svm` chưa có vì `hog_svm.py` còn TODO. `ms` = thời gian `localize_edge` cả ảnh, lặp lại ở mọi rank.
- segmentation: crop_source {gt, pred} × crop_method {bbox, quad} × binarize {otsu, adaptive} × segmenter {cca, projection}. Crop pred = top-1 của edge + denoise trong config (bilateral), quad pred theo `rectify.quad_method` (approxpoly). Không có ứng viên thì dòng pred có `n_boxes = 0`, mọi cột đúng/sai = False. `ms` = binarize + segment, không gồm crop.
- Lần chạy đầu dùng metric cũ (IoU-x ≥ 0.5). **Đã chạy lại sau khi merge #6** (luật tâm, aspect min 0.10): file CSV hiện tại là bản mới, số mới ở bảng cuối mục này.

Số tổng hợp để Châu Anh đối chiếu khi triage (KHÔNG dùng để chọn tham số, đây là tập test):

| denoise | P | R | F1 | R@5 | không có ứng viên | ms/ảnh |
|---|---|---|---|---|---|---|
| none | 0.376 | 0.314 | 0.342 | 0.370 | 76 | 26 |
| gaussian | 0.466 | 0.435 | 0.450 | 0.535 | 31 | 26 |
| median | 0.441 | 0.394 | 0.416 | 0.496 | 49 | 31 |
| bilateral | 0.593 | 0.545 | 0.568 | 0.667 | 37 | 109 |

Recall top-1 (bilateral) theo subset: base 0.609, blur 0.357, challenge 0.429, db 0.321, tilt 0.286.

Tách ký tự, **metric cũ** (lần chạy đầu, giữ để so):

| crop_source | crop | binarize | segment | count | pos0 | pos1_6 | all_correct |
|---|---|---|---|---|---|---|---|
| gt | quad | otsu | cca | 0.569 | 0.786 | 0.766 | 0.381 |
| gt | quad | otsu | projection | 0.665 | 0.825 | 0.797 | 0.426 |
| gt | quad | adaptive | cca | 0.539 | 0.790 | 0.793 | 0.364 |
| gt | quad | adaptive | projection | 0.576 | 0.816 | 0.793 | 0.361 |
| gt | bbox | adaptive | cca | 0.450 | 0.504 | 0.616 | 0.173 |
| pred | quad | adaptive | projection | 0.290 | 0.411 | 0.353 | 0.095 |
| pred | bbox | otsu | projection | 0.214 | 0.221 | 0.276 | 0.054 |

(Dòng bbox / pred chỉ ghi tổ hợp có all_correct cao nhất; đủ 16 tổ hợp xem trực tiếp trong CSV.)

Khoảng cách gt → pred (0.43 → ~0.09 all_correct) là phần lỗi do định vị (SPEC 9.5): top-1 chỉ đúng 54.5% ảnh, nên phần lớn lỗi tổng thể nằm ở bước định vị chứ không phải tách ký tự.

**Chạy lại sau khi merge #6** (metric luật tâm, `char_aspect_range` min 0.10), cùng lệnh. Định vị không đổi (bilateral top-1 R = 0.545). Tách ký tự:

| crop_source | crop | binarize | segment | count | pos0 | pos1_6 | all_correct |
|---|---|---|---|---|---|---|---|
| gt | quad | otsu | cca | 0.556 | 0.792 | 0.814 | 0.513 |
| gt | quad | otsu | projection | 0.643 | 0.829 | 0.867 | **0.587** |
| gt | quad | adaptive | cca | 0.528 | 0.820 | 0.875 | 0.470 |
| gt | quad | adaptive | projection | 0.524 | 0.870 | 0.910 | 0.474 |
| gt | bbox | adaptive | cca | 0.457 | 0.558 | 0.662 | 0.249 |
| gt | bbox | otsu | cca | 0.444 | 0.556 | 0.597 | 0.249 |
| pred | quad | otsu | projection | 0.281 | 0.418 | 0.370 | 0.141 |
| pred | quad | adaptive | projection | 0.279 | 0.459 | 0.433 | 0.139 |
| pred | bbox | otsu | cca | 0.216 | 0.212 | 0.306 | 0.069 |

all_correct theo subset (gt + quad + adaptive + cca, cấu hình thắng trên tune): base 0.571, blur 0.071, challenge 0.107, db 0.214, tilt 0.214.

Lưu ý: trên **test**, otsu + projection (0.587) cao hơn adaptive + cca (0.470), ngược với **tune** (adaptive = otsu, cca nhỉnh hơn projection). Không chọn lại cấu hình theo test; ghi vào phần thảo luận rằng chênh lệch giữa các tổ hợp nhỏ và không ổn định giữa hai tập. Khoảng cách gt → pred (0.59 → 0.14) vẫn cho thấy định vị là nguồn lỗi chính.

### E5, E6 trên tập test, 2026-11-08, Bích

`python experiments/run_all.py --exp binarize` và `--exp segmentation`. Cả hai gom số từ `results/predictions/segmentation.csv` (E0); thiếu file hoặc khác split thì tự chạy E0 trước. E0 đã chạy lại với code sau #14 (NARROW_ASPECT, cắt theo khe, lọc mảnh dẹt sau khi gộp). Bảng: `results/tables/e5_binarize.csv`, `e6_segmentation.csv`.

**E5: otsu vs adaptive** (crop gt + quad, segment cca), all_correct:

| subset | n | otsu | adaptive |
|---|---|---|---|
| base | 350 | **0.666** | 0.611 |
| blur | 28 | **0.143** | 0.071 |
| challenge | 28 | 0.143 | **0.214** |
| db | 28 | **0.393** | 0.250 |
| tilt | 28 | **0.286** | 0.214 |
| ALL | 462 | **0.563** | 0.509 |

Adaptive trúng vị trí nhiều hơn (pos0 0.859 vs 0.805, pos1_6 0.875 vs 0.815) nhưng sai số hộp nhiều hơn (count 0.567 vs 0.606), nên all_correct thấp hơn. Trên tune hai cách hòa (0.540 vs 0.566 sau #14). Config giữ `otsu`.

**E6: cca vs projection × gt/pred** (binarize theo config = otsu, crop quad), all_correct:

| subset | cca gt | projection gt | cca pred | projection pred |
|---|---|---|---|---|
| base | 0.666 | **0.749** | 0.171 | 0.191 |
| blur | 0.143 | 0.214 | 0.000 | 0.036 |
| challenge | 0.143 | 0.214 | 0.036 | 0.000 |
| db | **0.393** | 0.250 | 0.071 | 0.036 |
| tilt | 0.286 | 0.286 | 0.036 | 0.036 |
| ALL | 0.563 | **0.625** | 0.139 | 0.151 |

- Với otsu, projection hơn cca ở crop gt (0.625 vs 0.563, count 0.680 vs 0.606). Trên tune với otsu cũng vậy (0.596 vs 0.540), nên lần này hai tập nhất quán. Config `segment.method` vẫn là `cca`; đổi hay không thì nhóm quyết (file SHARED).
- Khoảng cách gt → pred: 0.625 → 0.151 (projection). Lỗi tổng thể chủ yếu do định vị (bilateral top-1 R = 0.545), đúng như SPEC 9.5.
- SPEC 9.4 (P/R/F1 theo hộp gán tay): `char_boxes.csv` chưa có biển nào nên cột `char_*` để trống. Gán xong 30 biển thì chạy lại `--exp segmentation`, không cần chạy lại E0.
