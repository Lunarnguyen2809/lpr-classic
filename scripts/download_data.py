# OWNER: SHARED
"""Tải CCPD2019 và giải nén chỉ các subset cần dùng (xem SPEC mục 5.1 và 5.6).

Ví dụ:
    python scripts/download_data.py --source gdrive --out data/raw
    python scripts/download_data.py --source local --archive D:/Downloads/CCPD2019.tar.xz --out data/raw
    python scripts/download_data.py --source local --extracted D:/CCPD2019 --out data/raw
    python scripts/download_data.py --source kaggle --kaggle-slug <owner>/<dataset> --out data/raw
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import shutil
import subprocess
import sys
import tarfile

GDRIVE_ID = "1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"
ARCHIVE_NAME = "CCPD2019.tar.xz"
DEFAULT_SUBSETS = ["base", "tilt", "blur", "db", "challenge"]
XZ_MAGIC = b"\xfd7zXZ\x00"

MANUAL_HELP = f"""
==================================================================
Không tải được CCPD2019 từ Google Drive (có thể do lỗi quota).
Tải tay theo một trong các cách sau:

  1. Mở trình duyệt (đã đăng nhập Google):
       https://drive.google.com/file/d/{GDRIVE_ID}/view
     Tải file {ARCHIVE_NAME}, sau đó chạy:
       python scripts/download_data.py --source local --archive <đường/dẫn>/{ARCHIVE_NAME} --out data/raw

  2. BaiduYun: https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw  (mã: hm0u)

  3. Kaggle (cần ~/.kaggle/kaggle.json và `pip install kaggle`):
       python scripts/download_data.py --source kaggle --kaggle-slug <owner>/<dataset> --out data/raw

  Repo gốc: https://github.com/detectRecog/CCPD
==================================================================
"""

for _s in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, không in được tiếng Việt
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


class QuotaError(RuntimeError):
    """Google Drive từ chối tải (quota hoặc trả về trang HTML)."""


def _is_xz(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(len(XZ_MAGIC)) == XZ_MAGIC


def download_gdrive(out_dir: str) -> str:
    """Tải file qua gdown, đúng một lần. Raise QuotaError nếu lỗi quota hoặc nhận về HTML."""
    import gdown

    dest = os.path.join(out_dir, ARCHIVE_NAME)
    if os.path.isfile(dest) and _is_xz(dest):
        print(f"Đã có {dest}, bỏ qua bước tải.")
        return dest
    try:
        result = gdown.download(id=GDRIVE_ID, output=dest, quiet=False, resume=True)
    except Exception as e:  # gdown ném FileURLRetrievalError khi quota/không quyền
        raise QuotaError(str(e)) from e
    if not result or not os.path.isfile(dest):
        raise QuotaError("gdown không trả về file.")
    if not _is_xz(dest):
        bad = dest + ".html"
        shutil.move(dest, bad)
        raise QuotaError(f"File tải về không phải .tar.xz (có thể là trang HTML báo quota), đã lưu ở {bad}")
    return dest


def download_kaggle(slug: str, out_dir: str) -> str:
    """Tải dataset Kaggle theo slug bằng Kaggle CLI. Trả về .tar.xz nếu có, ngược lại trả về thư mục."""
    exe = shutil.which("kaggle")
    if exe is None:
        sys.exit("Không tìm thấy lệnh `kaggle`. Cài bằng `pip install kaggle` và đặt ~/.kaggle/kaggle.json.")
    dest = os.path.join(out_dir, "kaggle")
    os.makedirs(dest, exist_ok=True)
    subprocess.run([exe, "datasets", "download", "-d", slug, "-p", dest, "--unzip"], check=True)
    archives = glob.glob(os.path.join(dest, "**", "*.tar.xz"), recursive=True)
    return archives[0] if archives else dest


def _wanted_dirs(subsets: list[str]) -> set[str]:
    return {s if s.startswith("ccpd_") else f"ccpd_{s}" for s in subsets}


def extract_subsets(archive: str, out_dir: str, subsets: list[str]) -> dict[str, int]:
    """Giải nén streaming (`r|xz`), chỉ lấy file nằm trong thư mục của các subset cần dùng.

    Trả về số file đã trích theo từng thư mục subset.
    """
    wanted = _wanted_dirs(subsets)
    counts = {w: 0 for w in wanted}
    out_abs = os.path.abspath(out_dir)
    with tarfile.open(archive, mode="r|xz") as tar:
        for m in tar:
            parts = m.name.replace("\\", "/").split("/")
            hit = next((p for p in parts[:-1] if p in wanted), None)
            if hit is None or not m.isfile():
                continue
            target = os.path.abspath(os.path.join(out_dir, m.name))
            if not target.startswith(out_abs + os.sep):
                continue  # bỏ qua đường dẫn thoát ra ngoài out_dir
            try:
                tar.extract(m, out_dir, filter="data")
            except TypeError:  # Python < 3.12 không có tham số filter
                tar.extract(m, out_dir)
            counts[hit] += 1
            if counts[hit] % 5000 == 0:
                print(f"  {hit}: {counts[hit]} file")
    return counts


def write_info(out_dir: str, source: str, archive: str | None, extra: dict) -> None:
    size = os.path.getsize(archive) if archive and os.path.isfile(archive) else None
    lines = [
        f"source: {source}",
        f"date: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"archive: {archive or '-'}",
        f"archive_size_bytes: {size if size is not None else '-'}",
    ] + [f"{k}: {v}" for k, v in extra.items()]
    with open(os.path.join(out_dir, "DOWNLOAD_INFO.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["gdrive", "kaggle", "local"], required=True)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--archive", help="file .tar.xz có sẵn (source=local)")
    ap.add_argument("--extracted", help="thư mục CCPD đã giải nén sẵn (source=local)")
    ap.add_argument("--kaggle-slug", help="<owner>/<dataset> (source=kaggle)")
    ap.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    archive: str | None = None
    extra: dict = {"subsets": " ".join(args.subsets)}

    if args.source == "gdrive":
        try:
            archive = download_gdrive(args.out)
        except QuotaError as e:
            print(f"Lỗi: {e}", file=sys.stderr)
            print(MANUAL_HELP, file=sys.stderr)
            sys.exit(2)
    elif args.source == "kaggle":
        if not args.kaggle_slug:
            ap.error("--source kaggle cần --kaggle-slug <owner>/<dataset>")
        got = download_kaggle(args.kaggle_slug, args.out)
        extra["kaggle_slug"] = args.kaggle_slug
        if os.path.isdir(got):
            extra["extracted_dir"] = got
            write_info(args.out, args.source, None, extra)
            print(f"Dữ liệu Kaggle đã giải nén ở {got}. Dùng: build_subset.py --raw {got}")
            return
        archive = got
    else:
        if args.extracted:
            if not os.path.isdir(args.extracted):
                sys.exit(f"Không thấy thư mục {args.extracted}")
            extra["extracted_dir"] = os.path.abspath(args.extracted)
            write_info(args.out, args.source, None, extra)
            print(f"Dùng thư mục đã giải nén. Chạy: build_subset.py --raw {args.extracted}")
            return
        if not args.archive or not os.path.isfile(args.archive):
            ap.error("--source local cần --archive <file.tar.xz> hoặc --extracted <thư mục>")
        archive = args.archive

    if not args.skip_extract:
        print(f"Giải nén {args.subsets} từ {archive} ...")
        counts = extract_subsets(archive, args.out, args.subsets)
        for k, v in sorted(counts.items()):
            print(f"  {k}: {v} file")
            extra[f"n_{k}"] = v
    write_info(args.out, args.source, archive, extra)
    print("Xong.")


if __name__ == "__main__":
    main()
