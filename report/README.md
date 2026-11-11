# Báo cáo giữa kỳ, phần A (LaTeX)

Cấu trúc:

- `main.tex`: preamble, trang bìa (placeholder), `\input` các section. Phần của Bích để sẵn dòng `% TODO(B)`.
- `sections/a_method.tex`, `a_results.tex`, `a_discussion.tex`: phần A.
- `tables/*.tex`: bảng sinh tự động từ CSV kết quả (chỉ phần `tabular`).
- `figures/`: hình dùng trong báo cáo.
- `make_figures.py`: sinh `tables/*.tex` và `figures/e3_subset_r1.png`.
- `check_tex.py`: kiểm tra nhanh (cân bằng `\begin`/`\end`, ngoặc, đường dẫn `\input`/`\includegraphics`, ký tự `_ % &` chưa escape).

## Biên dịch trên Overleaf

1. Nén cả thư mục `report/` thành zip (hoặc dùng file zip đã có).
2. Overleaf: New Project > Upload Project > chọn file zip.
3. Menu (góc trên trái) > Compiler: chọn **XeLaTeX**.
4. Menu > Main document: `main.tex`. Nhấn Recompile.

Font dùng TeX Gyre Termes / Heros / Cursor (có sẵn trong TeX Live trên Overleaf, hỗ trợ tiếng Việt). Không cần chữ Hán trong văn bản (các chữ Hán nằm trong ảnh).

## Sinh lại bảng và hình

```
python report/make_figures.py <thư mục chứa tables/*.csv> report
python report/check_tex.py report
```

Thư mục CSV là đầu ra của `experiments/run_all.py` và `experiments/triage_failures.py` (các file `e2_filters.csv`, `e3_localization.csv`, `e4_rectify.csv`, `e7_runtime.csv`, `triage_stage_counts.csv`, `triage_gt_vs_pred.csv`). Cần matplotlib. Các hình `triage_stages.png` và `grid_*.jpg` copy từ `results/figures` và `results/failures` (đã thu nhỏ về chiều rộng 900 px).

## Ghi chú số liệu

- Mọi số E2/E3/E4/E7 và triage lấy từ CSV, làm tròn 3 chữ số. Bảng resize, quét tham số, Issue #10 và HOG+SVM lấy từ `notes.md` (không có CSV).
- Thời gian E7 và E2 (ms) đo trên máy ít tải hơn lần ghi trong `notes.md`; báo cáo dùng số từ CSV.
