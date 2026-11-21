# Slide bảo vệ giữa kỳ

Slide Beamer 16:9 chung của nhóm cho đề tài "Phát hiện biển số xe và tách ký tự bằng xử lý ảnh cổ điển" (MAT3562E Computer Vision, HK1 2026-2027). Nội dung theo thứ tự pipeline: bài toán, dữ liệu, thí nghiệm E1-E8, phân tích lỗi, hạn chế, kết luận. Số liệu lấy từ báo cáo (`report/chapters`, `report/tables`, `notes.md`), không có số mới.

## Biên dịch

Dùng **XeLaTeX** (bắt buộc, vì dùng `fontspec`).

- **Overleaf:** tải thư mục `slides/` lên, đặt `slides.tex` làm file chính, vào Menu > Compiler và chọn **XeLaTeX**.
- **Tectonic (cục bộ):** `tectonic -X compile slides.tex` trong thư mục `slides/`.
- **XeLaTeX thường:** `xelatex slides.tex` (chạy 2 lần để có số trang tổng).

Font: TeX Gyre Heros (chữ thường) và TeX Gyre Cursor (đơn cách), có sẵn trong TeX Live và Overleaf, nạp theo tên file `.otf`.

## Cấu trúc

- `slides.tex`: toàn bộ 23 slide.
- `figures/`: `HUS.png` (logo), `e3_subset_r1.png`, `triage_stages_slide.png`, `loc_rank_cands.png`, `e1_hist.png`, `e5_binarize.jpg`, `e6_segment.jpg`, `vn_two_line.jpg`, `vn_loc_fail.jpg`.

File PDF không được commit; tự biên dịch khi cần.
