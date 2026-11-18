# lpr-classic

Project 9 (giữa kỳ): định vị biển số và tách ký tự bằng xử lý ảnh cổ điển, không học sâu. Thiết kế chi tiết ở [SPEC.md](SPEC.md).

## Cài đặt

Python 3.10 trở lên. Mọi lệnh chạy từ thư mục gốc repo.

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

## Tải dữ liệu

Thử Google Drive trước:

```bash
python scripts/download_data.py --source gdrive --out data/raw
```

Nếu báo lỗi quota (thoát mã 2), tải tay `CCPD2019.tar.xz` từ link được in ra rồi chạy:

```bash
python scripts/download_data.py --source local --archive /đường/dẫn/CCPD2019.tar.xz --out data/raw
```

Hoặc dùng thư mục đã giải nén sẵn (`--source local --extracted <thư mục>`), hoặc Kaggle (`--source kaggle --kaggle-slug <owner>/<dataset>`, cần `pip install kaggle` và `~/.kaggle/kaggle.json`).

Script chỉ giải nén 5 subset `base, tilt, blur, db, challenge` (đổi bằng `--subsets`), bỏ giải nén bằng `--skip-extract`, và ghi `data/raw/DOWNLOAD_INFO.txt`.

## Dựng subset

```bash
python scripts/build_subset.py --raw data/raw --out data/ccpd_subset --annotations data/annotations.csv
```

Lấy mẫu seed 42: base 500, mỗi subset khó 40 (tổng 660), chỉ lấy ảnh có chuỗi biển hợp lệ. Ảnh được copy thành `data/ccpd_subset/<subset>/<image_id>.jpg`.

## Chia tập

```bash
python scripts/make_splits.py --annotations data/annotations.csv --vn data/vn_custom/annotations.csv --out data/splits.csv
```

Tune 30% / test 70%, phân tầng theo subset. Ảnh VN luôn là `qual`.

## Ảnh VN (Kaggle công khai + crawl Brave)

~150–200 ảnh biển số VN sau lọc chặt (hiện 187), đa dạng màu biển, loại xe, 1/2 dòng, điều kiện chụp: 75 ảnh `tanhphp/vietnamese-license-plates` (Apache-2.0, chỉ ảnh gốc không `_aug`), 64 ảnh `duydieunguyen/licenseplates` (license không rõ, chỉ dùng nội bộ), 48 ảnh crawl Brave duyệt tay (internal_only). Mọi ảnh là split `qual`, chỉ test định tính. Nguồn, license và thẻ từng ảnh ở `data/vn_custom/sources.csv`.

**Ảnh `data/vn_custom/images/` không nằm trong git** (license: crawl `internal_only`, `duydieunguyen` unknown); repo chỉ giữ metadata. Dựng lại cục bộ: tải 2 dataset Kaggle về `data/raw/vn_kaggle/` (kaggle CLI, cần token), chạy crawl để có `data/vn_custom/candidates/`, rồi chạy `build_vn_set.py` từ `selection.csv` như bên dưới.

```bash
# (một lần, cần BRAVE_API_KEY) ứng viên crawl
python scripts/crawl_vn_brave.py --dry-run                 # xem query, không gọi mạng
python scripts/crawl_vn_brave.py                           # crawl ~400 ứng viên (resume được; --categories để chọn)
python scripts/crawl_vn_brave.py --contact-sheet           # lưới duyệt ở data/vn_custom/candidates/_sheets/
# dựng bộ ảnh cuối từ danh sách chọn
python scripts/build_vn_set.py --selection data/vn_custom/selection.csv [--force]
python scripts/build_vn_set.py --sheet data/vn_custom/_sheet_final.jpg    # lưới soát bbox (gitignore)
python scripts/make_splits.py                              # gán split qual
```

`build_vn_set.py` ghi `data/vn_custom/images/vn_NNNN.jpg`, `sources.csv`, `annotations.csv` (bbox ảnh Kaggle: bbox YOLO hoặc hình chữ nhật bao của polygon, đổi sang pixel) và `todo_cvat.txt` (ảnh crawl chưa có bbox, gán bằng CVAT rồi nối vào `annotations.csv`). Ứng viên crawl ở `data/vn_custom/candidates/` (gitignore).

## Chạy test

```bash
pytest -q
```

## Chạy thí nghiệm

```bash
python experiments/run_all.py --list
python experiments/run_all.py --exp predict --split test --limit 20
python experiments/run_all.py --exp all

# huấn luyện SVM cho localize/hog_svm (chỉ tune) và đo top-1 edge vs edge_svm
PYTHONPATH=src python experiments/train_hog_svm.py --eval-splits tune test

# E2, E3, E4, E7 (E3, E4, E7 cần models/svm.joblib ở trên; E3 đọc denoise đã chọn từ e2_filters.csv,
# E4 đọc method đã chọn từ e3_localization.csv, thiếu thì tự chạy E2/E3 trước)
PYTHONPATH=src python experiments/run_all.py --exp filters
PYTHONPATH=src python experiments/run_all.py --exp localization
PYTHONPATH=src python experiments/run_all.py --exp rectify
PYTHONPATH=src python experiments/run_all.py --exp runtime

# E8: định tính trên ảnh VN (cần data/vn_custom/images/ cục bộ); ghi results/figures/vn/vn_*.png và tables/e8_vn.csv
PYTHONPATH=src python experiments/run_all.py --exp vn_qualitative
```

Đo recall ứng viên của `localize_edge` theo subset (issue #10, mặc định tune; `--split test` chỉ dùng một lần cho cấu hình đã chốt):

```bash
PYTHONPATH=src python experiments/measure_localize_recall.py
```

Triage lỗi theo bước trên test (SPEC 2.3; cần `results/predictions/*.csv` của `--exp predict`, thiếu bảng E2/E3 thì tự chạy trên tune):

```bash
PYTHONPATH=src python experiments/run_all.py --exp predict --split test
PYTHONPATH=src python experiments/triage_failures.py            # bảng tables/triage_*.csv, failures/<bước>/, figures/triage_stages.png
PYTHONPATH=src python experiments/triage_failures.py --segmenter projection   # chỉ bảng, hậu tố _otsu_projection
# Lưu ý: triage_failures.py import `_chosen`, `_load_svm`, `segmentation_rows` từ run_all.py (phần của B); đổi tên các hàm này thì triage hỏng.
```

Kết quả nằm trong `results/` (predictions, tables, figures, failures).

## Notebook demo pipeline

`notebooks/02_pipeline_demo.ipynb` minh hoạ từng bước (tiền xử lý, Sobel + morph, ứng viên + NMS, SVM rerank, warp, nhị phân, tách ký tự, so với GT) trên 4 ảnh test cố định. Cần `data/ccpd_subset/` và `models/svm.joblib` (train bằng `experiments/train_hog_svm.py` ở trên). Chạy lại cần cài thêm Jupyter (`pip install jupyter`, có `nbconvert` và `ipykernel`; không nằm trong `requirements.txt`). Chạy lại và lưu kèm output:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_pipeline_demo.ipynb
```

## Phân công file

| File | Chủ |
|---|---|
| `src/lpr/types.py`, `config.py`, `io_utils.py`, `data/*`, `scripts/*` | SHARED |
| `src/lpr/preprocess.py` | A |
| `src/lpr/localize/edge.py`, `localize/hog_svm.py` | A |
| `src/lpr/rectify.py` | A |
| `src/lpr/visualize.py` | A |
| `notebooks/02_pipeline_demo.ipynb`, `tests/test_rectify.py` | A |
| `src/lpr/color_analysis.py` | B |
| `src/lpr/binarize.py` | B |
| `src/lpr/segment.py` | B |
| `src/lpr/evaluate.py`, `tests/test_evaluate.py` | B |
| `experiments/run_all.py`, `notebooks/01_color_spaces.ipynb` | B |
| `experiments/train_hog_svm.py`, `tests/test_hog_svm.py` | A |
| `experiments/measure_localize_recall.py` | synaamin |
| `experiments/triage_failures.py`, `tests/test_triage.py` | A (synaamin) |

Chỉ sửa file mình sở hữu. Muốn đổi file SHARED hoặc chữ ký hàm thì sửa SPEC trước và báo người kia. Tham số đã thử ghi vào [notes.md](notes.md).
