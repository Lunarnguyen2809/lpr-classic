# Báo cáo giữa kỳ, phần A (LaTeX, mẫu báo cáo HUS, pdfLaTeX)

Cấu trúc:

- `main.tex`: preamble theo mẫu HUS (extbook 13pt, T5, babel vietnamese, mathptmx), trang bìa, bảng phân công, xác nhận, mục lục, danh mục hình và bảng, `\include` các chương, tài liệu tham khảo (`thebibliography`). Thông tin bìa (môn, giảng viên, học kỳ, tên và MSSV) khai báo bằng macro ở đầu file; `\TODO{...}` hiện chữ đỏ, chỉ còn dùng cho phần của Bích.
- `chapters/ch1_gioithieu.tex`: bài toán, dữ liệu, chia tập, độ đo.
- `chapters/ch2_phuongphap_a.tex`, `ch3_ketqua_a.tex`, `ch4_phantichloi.tex`: phần A (phương pháp, kết quả, phân tích lỗi).
- `chapters/ch5_phan_b.tex`: chương dành cho Bích (E1, E5, E6, E8), hiện là `\TODO`.
- `chapters/ketluan.tex`: kết luận.
- `tables/*.tex`: bảng sinh tự động từ CSV (mỗi file là một môi trường `table` đầy đủ: caption, label, hàng tiêu đề xanh chữ trắng đậm).
- `figures/`, `image/HUS.png`: hình và logo.
- `make_figures.py`: sinh `tables/*.tex` và `figures/e3_subset_r1.png`.
- `check_tex.py`: kiểm tra nhanh (gói chỉ dành cho XeLaTeX như `fontspec`, ký tự Unicode không an toàn với T5, cân bằng `\begin`/`\end`, ngoặc, đường dẫn `\input`/`\include`/`\includegraphics`, `\ref`/`\cite` có đích, ký tự `_ # % &` chưa escape, macro cần gói).

## Biên dịch trên Overleaf

1. Nén nội dung thư mục `report/` thành zip sao cho `main.tex` nằm ở gốc zip.
2. Overleaf: New Project > Upload Project > chọn file zip.
3. Menu > Compiler: **pdfLaTeX** (mặc định). Menu > Main document: `main.tex`. Nhấn Recompile.

Không dùng `fontspec`/XeLaTeX. Văn bản chỉ chứa chữ Latin và dấu tiếng Việt (UTF-8, T5); ký hiệu đặc biệt viết bằng lệnh LaTeX (`$\geq$`, `$\rightarrow$`, `\textendash`). Chữ Hán chỉ nằm trong ảnh.

Lưu ý: bản này chưa được biên dịch bằng pdfLaTeX trong môi trường phát triển (máy không có đủ gói TeX); đã kiểm tra tĩnh bằng `check_tex.py`. Cần Recompile trên Overleaf để xác nhận, rồi sửa nếu có cảnh báo.

## Sinh lại bảng và hình

```
python report/make_figures.py <thư mục chứa tables/*.csv> report
python report/check_tex.py report
```

Thư mục CSV là đầu ra của `experiments/run_all.py` và `experiments/triage_failures.py` (các file `e2_filters.csv`, `e3_localization.csv`, `e4_rectify.csv`, `e7_runtime.csv`, `triage_stage_counts.csv`, `triage_gt_vs_pred.csv`). Cần matplotlib. Các hình `triage_stages.png` và `grid_*.jpg` copy từ `results/figures` và `results/failures` (đã thu nhỏ về chiều rộng 900 px).

## Ghi chú số liệu

- Mọi số E2/E3/E4/E7 và triage lấy từ CSV. Quy tắc làm tròn (một lần, half-up): P, R, F1 và các tỉ lệ tính từ số đếm (`tp/fp/fn`, số ảnh) rồi mới làm tròn 3 chữ số, không làm tròn lại từ giá trị 4 chữ số trong CSV (vd. 148/198 = 0.7475 hiển thị 0.747; 1/16 = 0.0625 hiển thị 0.063). Bảng resize, quét tham số, Issue #10 và HOG+SVM lấy từ `notes.md` (không có CSV).
- Thời gian E7 và E2 (ms) đo trên máy ít tải hơn lần ghi trong `notes.md`; báo cáo dùng số từ CSV, số đo lại ghi ở mục "Đo lại thời gian (máy ít tải)" (2026-11-11) trong `notes.md`. Tỉ lệ giữa các bước cũng dịch theo tải máy (tiền xử lý 44% -> 50%) nên chỉ là gần đúng.
