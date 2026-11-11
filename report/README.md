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

Font dùng TeX Gyre Termes / Heros / Cursor, khai báo theo tên file (`texgyretermes-regular.otf`, ...) để chạy được cả trên Overleaf lẫn Tectonic; hỗ trợ tiếng Việt. Gói `amssymb` đã khai báo trong `main.tex` (cần cho `\mathbb`). Không cần chữ Hán trong văn bản (các chữ Hán nằm trong ảnh).

## Biên dịch bằng Tectonic (không cần cài TeX Live)

```
cd report
tectonic -X compile main.tex
```

Bản PDF kèm theo được biên dịch bằng Tectonic (XeTeX), 15 trang, 0 lỗi, 0 tham chiếu thiếu. `check_tex.py` còn kiểm tra macro cần gói (`\mathbb`, `\toprule`, `subfigure`, ...) mà preamble chưa `\usepackage`.

## Sinh lại bảng và hình

```
python report/make_figures.py <thư mục chứa tables/*.csv> report
python report/check_tex.py report
```

Thư mục CSV là đầu ra của `experiments/run_all.py` và `experiments/triage_failures.py` (các file `e2_filters.csv`, `e3_localization.csv`, `e4_rectify.csv`, `e7_runtime.csv`, `triage_stage_counts.csv`, `triage_gt_vs_pred.csv`). Cần matplotlib. Các hình `triage_stages.png` và `grid_*.jpg` copy từ `results/figures` và `results/failures` (đã thu nhỏ về chiều rộng 900 px).

## Ghi chú số liệu

- Mọi số E2/E3/E4/E7 và triage lấy từ CSV. Quy tắc làm tròn (một lần, half-up): P, R, F1 và các tỉ lệ tính từ số đếm (`tp/fp/fn`, số ảnh) rồi mới làm tròn 3 chữ số, không làm tròn lại từ giá trị 4 chữ số trong CSV (vd. 148/198 = 0.7475 hiển thị 0.747; 1/16 = 0.0625 hiển thị 0.063). Bảng resize, quét tham số, Issue #10 và HOG+SVM lấy từ `notes.md` (không có CSV).
- Thời gian E7 và E2 (ms) đo trên máy ít tải hơn lần ghi trong `notes.md`; báo cáo dùng số từ CSV, số đo lại ghi ở mục "Đo lại thời gian (máy ít tải)" (2026-11-11) trong `notes.md`. Tỉ lệ giữa các bước cũng dịch theo tải máy (tiền xử lý 44% -> 50%) nên chỉ là gần đúng.
