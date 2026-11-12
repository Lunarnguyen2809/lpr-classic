# Slide bảo vệ giữa kỳ, phần A

Slide Beamer 16:9 cho phần A (Bùi Thuỳ Châu Anh) của đề tài "Phát hiện biển số xe và tách ký tự bằng xử lý ảnh cổ điển" (MAT3562E Computer Vision, HK1 2026-2027). Số liệu lấy từ báo cáo trên nhánh `a/report` (`report/chapters`, `report/tables`, `notes.md`), không có số mới.

## Biên dịch

Dùng **XeLaTeX** (bắt buộc, vì dùng `fontspec`).

- **Overleaf:** tải thư mục `slides/` lên, đặt `slides_a.tex` làm file chính, vào Menu > Compiler và chọn **XeLaTeX**.
- **Tectonic (cục bộ):** `tectonic -X compile slides_a.tex` trong thư mục `slides/`.
- **XeLaTeX thường:** `xelatex slides_a.tex` (chạy 2 lần để có số trang tổng).

Font: TeX Gyre Heros (chữ thường) và TeX Gyre Cursor (đơn cách), có sẵn trong TeX Live và Overleaf, nạp theo tên file `.otf`.

## Cấu trúc

- `slides_a.tex`: toàn bộ 13 slide.
- `figures/`: `HUS.png` (logo), `e3_subset_r1.png`, `triage_stages.png` (từ `report/figures`), `loc_rank_cands.png` (cắt từ `grid_loc_rank.jpg`).

File PDF không được commit; tự biên dịch khi cần.
