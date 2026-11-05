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

## Nhị phân

## Tách ký tự

## Đánh giá
