# lpr-classic: Phát hiện biển số xe và tách ký tự bằng xử lý ảnh cổ điển

- **Môn học:** MAT3562E – Computer Vision, học kỳ 1 năm học 2026–2027
- **Giảng viên:** TS. Cao Văn Chung
- **Nhóm:** 5 · Đề tài: Dự án 9 (License Plate Detection and Character Segmentation)
- **Mã nguồn:** https://github.com/Lunarnguyen2809/lpr-classic

Bài toán: kiểm soát xe ra vào bãi gửi xe. Từ ảnh xe chụp bằng camera cố định, hệ thống tìm biển số, nắn biển về hình chữ nhật chuẩn 220×70 rồi tách từng ký tự để chuẩn bị cho bước nhận dạng (OCR). Chỉ dùng xử lý ảnh cổ điển, không dùng học sâu.

Pipeline: tiền xử lý → sinh ứng viên bằng cạnh Sobel và hình thái học → lọc ứng viên theo hình học, xếp hạng lại bằng HOG + SVM → nắn phối cảnh → nhị phân hóa (Otsu, adaptive) → tách ký tự (CCA, projection).

## 1. Thành viên và phân công

| Họ tên | MSSV | Vai trò |
|---|---|---|
| Nguyễn Ngọc Bích | 24001977 | Trưởng nhóm |
| Bùi Thuỳ Châu Anh | 24001963 | Thành viên |

**Nguyễn Ngọc Bích**
- Phân tích hệ màu, nhị phân hóa, tách ký tự, các độ đo đánh giá (`color_analysis.py`, `binarize.py`, `segment.py`, `evaluate.py`).
- Khung chạy thí nghiệm `run_all.py`; thí nghiệm E0 (chạy toàn pipeline), E1 (hệ màu), E5 (nhị phân hóa), E6 (tách ký tự); công cụ gán nhãn ký tự `label_chars.py`.
- Báo cáo và slide phần hệ màu, nhị phân hóa, tách ký tự, E8 và kết luận chung.

**Bùi Thuỳ Châu Anh**
- Tiền xử lý, định vị biển bằng cạnh và hình thái học, chấm điểm lại ứng viên bằng HOG + SVM, nắn phối cảnh, trực quan hóa (`preprocess.py`, `localize/`, `rectify.py`, `visualize.py`).
- Thí nghiệm E2 (bộ lọc nhiễu), E3 (định vị), E4 (nắn), E7 (thời gian), E8 (ảnh VN); huấn luyện SVM; phân tích lỗi theo bước; đo recall ứng viên (`train_hog_svm.py`, `triage_failures.py`, `measure_localize_recall.py`).
- Dựng bộ ảnh biển số VN (`crawl_vn_brave.py`, `build_vn_set.py`); notebook demo pipeline.
- Báo cáo và slide phần tiền xử lý, định vị, nắn, thời gian chạy, phân tích lỗi; chuyển báo cáo sang mẫu HUS và ghép bản chung.

**Cùng làm**
- Đặc tả thiết kế (`SPEC.md`).
- Khung chung của thư viện và chuẩn bị dữ liệu CCPD (`types.py`, `config.py`, `io_utils.py`, `src/lpr/data/`, `download_data.py`, `build_subset.py`, `make_splits.py`).
- Gán tay hộp ký tự cho 30 biển test, mỗi người 15 biển.

## 2. Cài đặt

Yêu cầu Python 3.10 trở lên. Mọi lệnh chạy từ thư mục gốc repo.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` cài thư viện `lpr` trong `src/` để các script import được.

## 3. Dữ liệu

### 3.1 Tải dữ liệu đã dựng sẵn (khuyến nghị)

Link Drive: **<DÁN LINK DRIVE VÀO ĐÂY>** (chỉ chia sẻ cho giảng viên, xem mục 7).

Tải thư mục `data/` trên Drive và chép vào thư mục gốc repo (gộp với `data/` có sẵn). Sau khi chép cần có:

```
data/ccpd_subset/{base,tilt,blur,db,challenge}/   660 ảnh CCPD đã lấy mẫu
data/vn_custom/images/                            ảnh biển số VN cho E8
```

Các file nhãn đã có sẵn trong repo (mục 3.3), nên không cần chạy lại các bước ở mục 3.2.

### 3.2 Nguồn dữ liệu gốc và cách dựng lại

**CCPD2019** (Xu và cộng sự, ECCV 2018), hơn 300 nghìn ảnh xe biển Trung Quốc:
- Repo gốc: https://github.com/detectRecog/CCPD
- Google Drive: https://drive.google.com/file/d/1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc/view
- BaiduYun: https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw (mã: `hm0u`)

```bash
# 1. Tải và giải nén 5 nhóm base, tilt, blur, db, challenge vào data/raw
python scripts/download_data.py --source gdrive --out data/raw
#    Nếu Google Drive báo hết lượt tải (script thoát mã 2): tải tay CCPD2019.tar.xz từ một link ở trên, rồi chạy
python scripts/download_data.py --source local --archive <đường dẫn>/CCPD2019.tar.xz --out data/raw

# 2. Lấy mẫu 660 ảnh (seed 42) và đọc nhãn từ tên file
python scripts/build_subset.py --raw data/raw --out data/ccpd_subset --annotations data/annotations.csv

# 3. Chia tune 30% / test 70% theo từng nhóm
python scripts/make_splits.py --annotations data/annotations.csv --vn data/vn_custom/annotations.csv --out data/splits.csv
```

Tiêu chí lấy mẫu: nhóm base 500 ảnh, mỗi nhóm khó (tilt, blur, db, challenge) 40 ảnh; chỉ lấy ảnh có chuỗi biển hợp lệ đủ 7 ký tự. Ảnh được đổi tên thành `<nhóm>_<số thứ tự>.jpg` (ví dụ `base_0000.jpg`), tên gốc giữ ở cột `orig_name`.

**Ảnh biển số Việt Nam** (chỉ dùng cho E8, đánh giá định tính): 139 ảnh có hộp biển (44 biển một dòng, 95 biển hai dòng), lấy từ 2 bộ trên Kaggle:
- https://www.kaggle.com/datasets/tanhphp/vietnamese-license-plates (75 ảnh, giấy phép Apache-2.0)
- https://www.kaggle.com/datasets/duydieunguyen/licenseplates (64 ảnh, chưa rõ giấy phép)

Danh sách ảnh đã chọn và nguồn của từng ảnh nằm trong `data/vn_custom/selection.csv` và `data/vn_custom/sources.csv`. Ngoài ra có 48 ảnh tìm qua Brave Image Search chưa gán hộp biển nên chưa dùng.

### 3.3 Nhãn có sẵn trong repo

- `data/annotations.csv`: 660 ảnh CCPD. Gồm bbox, 4 góc biển (thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái), chuỗi biển, độ nghiêng, độ sáng, độ mờ, đọc từ tên file gốc của CCPD.
- `data/splits.csv`: tune 198 ảnh, test 462 ảnh, qual 139 ảnh (ảnh VN).
- `data/char_labels/char_boxes.csv`: hộp ký tự gán tay cho 30 biển test (210 hộp), tọa độ trên ảnh biển đã nắn 220×70.
- `data/vn_custom/annotations.csv`: hộp biển của 139 ảnh VN.

## 4. Cấu trúc thư mục

```
lpr-classic/
├── README.md
├── SPEC.md                      # đặc tả thiết kế: chữ ký hàm, định dạng dữ liệu, độ đo
├── notes.md                     # sổ ghi các tham số đã thử và kết quả
├── requirements.txt
├── pyproject.toml               # cài thư viện lpr bằng pip install -e .
├── config/
│   ├── default.yaml             # toàn bộ tham số pipeline và đường dẫn
│   └── vn_crawl.yaml            # cấu hình tìm ảnh VN qua Brave
├── data/
│   ├── annotations.csv          # nhãn 660 ảnh CCPD
│   ├── splits.csv               # chia tune / test / qual
│   ├── char_labels/
│   │   └── char_boxes.csv       # hộp ký tự gán tay, 30 biển test
│   ├── vn_custom/
│   │   ├── annotations.csv      # hộp biển của 139 ảnh VN
│   │   ├── selection.csv        # danh sách ảnh VN đã chọn
│   │   ├── sources.csv          # nguồn, giấy phép, đặc điểm từng ảnh VN
│   │   ├── todo_cvat.txt        # 48 ảnh chưa gán hộp biển
│   │   └── images/              # (*) ảnh VN
│   ├── ccpd_subset/             # (*) 660 ảnh: base/ tilt/ blur/ db/ challenge/
│   └── raw/                     # (*) CCPD2019 gốc, chỉ có khi tự dựng lại
├── src/lpr/                     # thư viện chính
│   ├── types.py, config.py, io_utils.py
│   ├── data/                    # đọc nhãn CCPD, nạp tập dữ liệu
│   ├── preprocess.py            # resize, ảnh xám, CLAHE, lọc nhiễu
│   ├── color_analysis.py        # phân tích hệ màu (E1)
│   ├── localize/
│   │   ├── edge.py              # sinh và lọc ứng viên: Sobel-x, Otsu, hình thái học
│   │   └── hog_svm.py           # chấm điểm lại ứng viên bằng HOG + SVM
│   ├── rectify.py               # nắn phối cảnh về 220×70
│   ├── binarize.py              # nhị phân hóa Otsu, adaptive
│   ├── segment.py               # tách ký tự CCA, projection
│   ├── evaluate.py              # IoU, P/R/F1, độ đo tách ký tự
│   └── visualize.py             # vẽ hộp, lưới các bước xử lý
├── scripts/
│   ├── download_data.py         # tải và giải nén CCPD2019
│   ├── build_subset.py          # lấy mẫu 660 ảnh, đọc nhãn
│   ├── make_splits.py           # chia tune / test
│   ├── label_chars.py           # công cụ gán tay hộp ký tự
│   ├── crawl_vn_brave.py        # tìm ảnh VN qua Brave
│   └── build_vn_set.py          # dựng bộ ảnh VN từ selection.csv
├── experiments/
│   ├── run_all.py               # chạy thí nghiệm E0–E8
│   ├── train_hog_svm.py         # huấn luyện SVM trên tune
│   ├── triage_failures.py       # phân tích lỗi theo bước
│   └── measure_*.py, check_*.py # script phụ dùng khi tinh chỉnh trên tune
├── notebooks/
│   └── 02_pipeline_demo.ipynb   # demo từng bước trên 4 ảnh test
├── tests/                       # 16 file test (pytest)
├── report/                      # mã LaTeX báo cáo, hình, bảng
├── slides/                      # mã LaTeX slide
├── models/                      # (*) svm.joblib
└── results/                     # (*) predictions/ tables/ figures/ failures/
```

(*) Không có trên GitHub: tải từ Drive (mục 3.1) hoặc được tạo ra khi chạy.

## 5. Chạy lại thí nghiệm

Tham số đọc từ `config/default.yaml`. Mọi tham số và lựa chọn phương pháp đều chọn trên tập tune; tập test chỉ dùng để báo cáo.

```bash
# 1. Kiểm tra cài đặt (212 test, không cần dữ liệu)
pytest -q
python experiments/run_all.py --list

# 2. Huấn luyện SVM trên tập tune, lưu models/svm.joblib (cần trước mọi thí nghiệm trừ E1, E2)
python experiments/train_hog_svm.py

# 3. Chạy thử nhanh trên 20 ảnh (không bắt buộc)
python experiments/run_all.py --exp predict --limit 20

# 4. Chạy toàn bộ E0–E8 trên tập test
python experiments/run_all.py --exp all

# 5. Phân tích lỗi theo bước pipeline (cần kết quả E0)
python experiments/triage_failures.py
```

Chạy riêng một thí nghiệm: `python experiments/run_all.py --exp <tên>`, thêm `--split tune` hoặc `--limit N` nếu cần.

| Mã | `--exp` | Nội dung | Kết quả (trong `results/`) |
|---|---|---|---|
| E0 | `predict` | Chạy toàn pipeline, ghi dự đoán từng ảnh | `predictions/localization.csv`, `predictions/segmentation.csv` |
| E1 | `color` | Độ tách biển/nền của 7 kênh màu (xám, H, S, V, L, a, b), trên tune | `tables/e1_color.csv`, `figures/e1_hist.png` |
| E2 | `filters` | Bộ lọc nhiễu: none, gaussian, median, bilateral | `tables/e2_filters.csv` |
| E3 | `localization` | Định vị: điểm heuristic (`edge`) và HOG + SVM (`edge_svm`) | `tables/e3_localization.csv` |
| E4 | `rectify` | Cắt theo bbox và nắn theo 4 góc | `tables/e4_rectify.csv` |
| E5 | `binarize` | Nhị phân hóa: Otsu và adaptive | `tables/e5_binarize.csv` |
| E6 | `segmentation` | Tách ký tự: CCA và projection | `tables/e6_segmentation.csv` |
| E7 | `runtime` | Thời gian từng bước | `tables/e7_runtime.csv` |
| E8 | `vn_qualitative` | Ảnh biển số VN, đánh giá định tính (cần `data/vn_custom/images/`) | `figures/vn/*.png`, `tables/e8_vn.csv` |

Thí nghiệm sau tự đọc lựa chọn tốt nhất của thí nghiệm trước: E3 dùng bộ lọc E2 chọn; E0, E4, E7, E8 dùng bộ lọc và phương pháp định vị mà E2, E3 chọn; thiếu bảng thì tự chạy lại. E5, E6 đọc kết quả của E0. Bước phân tích lỗi ghi `tables/triage_*.csv`, `failures/<bước lỗi>/` và `figures/triage_stages.png`.

**Notebook demo:** `notebooks/02_pipeline_demo.ipynb` minh họa từng bước trên 4 ảnh test cố định và đã lưu sẵn output. Muốn chạy lại cần có `data/ccpd_subset/`, `models/svm.joblib` và Jupyter (`pip install jupyter`):

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_pipeline_demo.ipynb
```

## 6. Ảnh kết quả minh họa

- `notebooks/02_pipeline_demo.ipynb`: từng bước trên 4 ảnh test (tiền xử lý, bản đồ cạnh và hình thái học, ứng viên, xếp hạng bằng SVM, nắn, nhị phân hóa, hộp ký tự).
- `report/figures/vn_success.jpg`: biển VN một dòng; hộp thật (xanh lá) và ứng viên top-1 (đỏ), biển sau nắn, ảnh nhị phân và hộp ký tự.
- `report/figures/e5_binarize.jpg`: Otsu và adaptive trên ảnh biển, kèm hộp ký tự và khuôn 7 vị trí.
- `report/figures/e6_segment.jpg`: tách ký tự bằng CCA và projection.
- `report/figures/grid_loc_no_candidate.jpg`, `grid_loc_rank.jpg`, `grid_rectify.jpg`, `grid_segment_count.jpg`, `grid_segment_position.jpg`: các bước xử lý của 5 ảnh test, mỗi ảnh lỗi ở một bước.
- `report/figures/vn_two_line.jpg`, `vn_loc_fail.jpg`: lỗi trên biển VN hai dòng và các ca định vị sai.

Sau khi chạy, ảnh mới nằm trong `results/figures/` và `results/failures/<bước lỗi>/`.

## 7. Giấy phép và quyền riêng tư

- Ảnh dữ liệu không đưa lên GitHub. Link Drive ở mục 3.1 chỉ chia sẻ cho giảng viên để kiểm tra bài.
- Ảnh VN chụp xe có biển số thật, và bộ `duydieunguyen/licenseplates` chưa rõ giấy phép, nên chỉ dùng trong phạm vi bài tập, không công bố lại.
