# OWNER: SHARED
"""Crawl ảnh biển số VN từ Brave Image Search làm ứng viên, duyệt tay (SPEC mục 5.4).

Cần biến môi trường BRAVE_API_KEY (không bao giờ in ra). Quy trình:
    1. crawl:         python scripts/crawl_vn_brave.py [--categories night cctv] [--dry-run]
    2. contact sheet: python scripts/crawl_vn_brave.py --contact-sheet
    3. duyệt tay các sheet, ghi candidate_id giữ lại (vd candidates/keep_suggested.txt)
    4. bộ ảnh cuối (~150–200 sau lọc) do build_vn_set.py dựng: chọn ảnh vào data/vn_custom/selection.csv rồi chạy scripts/build_vn_set.py
       (thay cho `--promote` cũ; build_vn_set.py là nơi DUY NHẤT ghi images/, sources.csv, annotations.csv)

Ứng viên nằm ở data/vn_custom/candidates/<category>/vnc_<sha12>.jpg|png (gitignore) cùng manifest.csv
(status luôn là hold; ảnh được chọn do selection.csv quyết định). Dedupe sha256 + dHash xuyên mọi category, kể cả ứng viên đã lưu
(chạy lại được). Không đụng annotations.csv: bbox gán sau bằng CVAT.

manifest.csv: cột `file` là đường dẫn file đã lưu, tương đối so với candidates/ (webp được đổi sang .png).
Cột `sha256`, `dhash`, `mime` mô tả BYTE GỐC tải về (trước khi đổi webp -> png), không phải file đã lưu.
Manifest cũ thiếu cột `file` được tự backfill (glob candidates/<category>/<candidate_id>.*) khi crawl/contact sheet.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

for _s in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, không in được tiếng Việt
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

API = "https://api.search.brave.com/res/v1/images/search"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
MANIFEST_FIELDS = ["candidate_id", "category", "search_query", "source_page_url", "original_image_url",
                   "title", "width", "height", "mime", "sha256", "dhash", "accessed_at", "status", "file"]
REASONS = ["no_url", "filtered_title", "fetch_fail", "too_small", "dup_sha", "dup_dhash"]
ROOT = Path(__file__).resolve().parent.parent


# ---------- logic thuần (không mạng) ----------

def dhash(im: Image.Image) -> int:
    """dHash 64 bit: xám 9x8, so pixel kề ngang."""
    g = im.convert("L").resize((9, 8), Image.LANCZOS)
    px = g.tobytes()
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (1 if px[r * 9 + c] > px[r * 9 + c + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class Dedupe:
    """Tập sha256 + dHash đã thấy (toàn bộ category)."""

    def __init__(self, max_hamming: int = 6):
        self.max_hamming = max_hamming
        self.shas: set[str] = set()
        self.hashes: list[int] = []

    def add(self, sha: str, dh: int) -> None:
        self.shas.add(sha.lower())
        self.hashes.append(dh)

    def check(self, sha: str, dh: int) -> str | None:
        """Trả 'dup_sha' / 'dup_dhash' nếu trùng, None nếu mới."""
        if sha.lower() in self.shas:
            return "dup_sha"
        if any(hamming(dh, h) <= self.max_hamming for h in self.hashes):
            return "dup_dhash"
        return None


def candidate_quota(cat_cfg: dict, multiplier: float) -> int:
    if cat_cfg.get("candidate_quota"):
        return int(cat_cfg["candidate_quota"])
    return int(round(cat_cfg["final_quota"] * multiplier))


def title_filtered(title: str, words: list[str]) -> bool:
    t = (title or "").lower()
    return any(w.lower() in t for w in words)


def candidate_id(sha: str) -> str:
    return f"vnc_{sha[:12]}"


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh), [])


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def load_state(cand_dir: Path, max_hamming: int) -> tuple[Dedupe, dict[str, int]]:
    """Dựng dedupe + số ứng viên mỗi category từ manifest (resume)."""
    dd = Dedupe(max_hamming)
    counts: dict[str, int] = {}
    for r in read_csv(cand_dir / "manifest.csv"):
        dd.add(r["sha256"], int(r["dhash"], 16))
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    return dd, counts


def find_candidate_file(cand_dir: Path, category: str, cid: str, rel: str | None = None) -> Path | None:
    """File ứng viên: ưu tiên cột `file` của manifest, thiếu thì dò theo category + candidate_id."""
    if rel:
        p = cand_dir / rel
        if p.exists():
            return p
    for ext in (".jpg", ".png"):
        p = cand_dir / category / (cid + ext)
        if p.exists():
            return p
    return None


def migrate_manifest(cand_dir: Path) -> int:
    """Backfill cột `file` cho manifest cũ (tmp + os.replace). Trả số dòng đã ghi, 0 nếu không cần."""
    path = cand_dir / "manifest.csv"
    if not path.exists() or "file" in read_header(path):
        return 0
    rows = read_csv(path)
    for r in rows:
        p = find_candidate_file(cand_dir, r["category"], r["candidate_id"])
        r["file"] = p.relative_to(cand_dir).as_posix() if p else ""
    write_csv_atomic(path, MANIFEST_FIELDS, rows)
    return len(rows)


def contact_sheets(root: Path, thumb=(200, 150), cols=6) -> list[Path]:
    """Một JPEG lưới mỗi category, nhãn candidate_id vẽ trên thumbnail."""
    cand_dir = root / "candidates"
    out_dir = cand_dir / "_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    cats: dict[str, list[dict]] = {}
    for r in read_csv(cand_dir / "manifest.csv"):
        cats.setdefault(r["category"], []).append(r)
    paths = []
    tw, th = thumb
    for cat, rows in cats.items():
        n = len(rows)
        nrows = (n + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, nrows * th), (40, 40, 48))
        for i, r in enumerate(rows):
            p = find_candidate_file(cand_dir, cat, r["candidate_id"], r.get("file"))
            if p is None:
                continue
            x, y = (i % cols) * tw, (i // cols) * th
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((tw - 4, th - 4))
                sheet.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
            d = ImageDraw.Draw(sheet)
            label = r["candidate_id"]
            d.rectangle([x, y, x + 6 * len(label) + 4, y + 12], fill=(0, 0, 0))
            d.text((x + 2, y + 1), label, fill=(255, 255, 255))
        out = out_dir / f"{cat}.jpg"
        sheet.save(out, "JPEG", quality=88)
        paths.append(out)
    return paths


# ---------- mạng ----------

def search(key: str, query: str, cfg: dict, retries: int = 4) -> list[dict]:
    import requests
    params = {"q": query, "count": cfg.get("count", 100), "safesearch": cfg.get("safesearch", "off")}
    if cfg.get("search_lang"):
        params["search_lang"] = cfg["search_lang"]
    for attempt in range(retries + 1):
        r = requests.get(API, params=params, timeout=30,
                         headers={"X-Subscription-Token": key, "Accept": "application/json"})
        if r.status_code == 429 and attempt < retries:
            time.sleep(2 ** attempt * 2)
            continue
        r.raise_for_status()
        return r.json().get("results", [])
    return []


def fetch_image(url: str, max_bytes: int, timeout: float):
    import requests
    buf = bytearray()
    with requests.get(url, headers={"User-Agent": UA}, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(64 * 1024):
            buf += chunk
            if len(buf) > max_bytes:
                raise ValueError("too large")
    data = bytes(buf)
    im = Image.open(io.BytesIO(data))
    im.verify()
    im = Image.open(io.BytesIO(data))
    fmt = (im.format or "").lower()
    if fmt not in ("jpeg", "png", "webp"):
        raise ValueError(f"format {fmt}")
    im.load()
    return data, im, fmt


def safe_fetch(url, max_bytes, timeout):
    try:
        return fetch_image(url, max_bytes, timeout)
    except Exception:
        return None


def save_file(data: bytes, im: Image.Image, fmt: str, dest_base: Path) -> Path:
    if fmt == "webp":  # lưu png để mọi công cụ đọc được
        out = dest_base.with_suffix(".png")
        im.convert("RGB").save(out)
    else:
        out = dest_base.with_suffix(".jpg" if fmt == "jpeg" else ".png")
        out.write_bytes(data)
    return out


def crawl(args, cfg: dict, root: Path) -> None:
    cand_dir = root / "candidates"
    cats = cfg["categories"]
    names = args.categories or list(cats)
    bad = [n for n in names if n not in cats]
    if bad:
        sys.exit(f"category không có trong config: {bad}")
    if args.dry_run:
        for name in names:
            cc = cats[name]
            print(f"[{name}] quota ứng viên {candidate_quota(cc, cfg.get('candidate_multiplier', 2))}"
                  f" (final {cc['final_quota']})")
            for q in cc["queries"]:
                print(f"   q: {q}")
        return
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        sys.exit("Thiếu biến môi trường BRAVE_API_KEY")
    cand_dir.mkdir(parents=True, exist_ok=True)
    n_mig = migrate_manifest(cand_dir)
    if n_mig:
        print(f"manifest.csv cũ: backfill cột file cho {n_mig} dòng", flush=True)
    manifest = cand_dir / "manifest.csv"
    qdone_path = cand_dir / "queries_done.csv"
    qdone = {(r["category"], r["query"]) for r in read_csv(qdone_path)}
    dd, counts = load_state(cand_dir, args.max_hamming)
    excl = list(cfg.get("exclude_title", []))
    stats = {n: {k: 0 for k in REASONS + ["results", "saved"]} for n in names}
    api_calls = 0
    new_file = not manifest.exists()
    new_q = not qdone_path.exists()
    with manifest.open("a", newline="", encoding="utf-8") as mf, \
            qdone_path.open("a", newline="", encoding="utf-8") as qf:
        w = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        qw = csv.DictWriter(qf, fieldnames=["category", "query"])
        if new_file:
            w.writeheader()
        if new_q:
            qw.writeheader()
        for name in names:
            cc = cats[name]
            quota = candidate_quota(cc, cfg.get("candidate_multiplier", 2))
            if counts.get(name, 0) >= quota:
                print(f"[{name}] đã đủ {counts[name]}/{quota}, bỏ qua", flush=True)
                continue
            st = stats[name]
            cat_excl = excl + cc.get("exclude_title", [])
            (cand_dir / name).mkdir(exist_ok=True)
            for q in cc["queries"]:
                if counts.get(name, 0) >= quota:
                    break
                if (name, q) in qdone:
                    continue
                api_calls += 1
                try:
                    results = search(key, q, cfg)
                except Exception as e:
                    print(f"[{name}] search lỗi ({type(e).__name__}): {q}", flush=True)
                    time.sleep(args.delay)
                    continue
                time.sleep(args.delay)
                st["results"] += len(results)
                todo = []
                for res in results:
                    url = (res.get("properties") or {}).get("url") or ""
                    if not url.startswith("http"):
                        st["no_url"] += 1
                    elif title_filtered(res.get("title", ""), cat_excl):
                        st["filtered_title"] += 1
                    else:
                        todo.append((res, url))
                step = args.workers * 2
                for i in range(0, len(todo), step):
                    if counts.get(name, 0) >= quota:
                        break
                    chunk = todo[i:i + step]
                    with ThreadPoolExecutor(args.workers) as ex:
                        fetched = list(ex.map(
                            lambda t: safe_fetch(t[1], int(args.max_mb * 1024 * 1024), args.timeout), chunk))
                    for (res, url), got in zip(chunk, fetched):
                        if counts.get(name, 0) >= quota:
                            break
                        if got is None:
                            st["fetch_fail"] += 1
                            continue
                        data, im, fmt = got
                        if min(im.size) < args.min_side:
                            st["too_small"] += 1
                            continue
                        sha = hashlib.sha256(data).hexdigest()
                        dh = dhash(im)
                        why = dd.check(sha, dh)
                        if why:
                            st[why] += 1
                            continue
                        dd.add(sha, dh)
                        cid = candidate_id(sha)
                        out = save_file(data, im, fmt, cand_dir / name / cid)
                        w.writerow({
                            "candidate_id": cid, "category": name, "search_query": q,
                            "source_page_url": res.get("url", ""), "original_image_url": url,
                            "title": res.get("title", ""), "width": im.size[0], "height": im.size[1],
                            "mime": f"image/{fmt}", "sha256": sha, "dhash": f"{dh:016x}",
                            "accessed_at": datetime.now(timezone.utc).isoformat(), "status": "hold",
                            "file": out.relative_to(cand_dir).as_posix()})
                        mf.flush()
                        counts[name] = counts.get(name, 0) + 1
                        st["saved"] += 1
                # chỉ đánh dấu xong khi vòng kết quả chạy hết (quota dừng giữa chừng vẫn tính xong);
                # crash giữa vòng thì không ghi, lần sau chạy lại query này.
                qw.writerow({"category": name, "query": q})
                qf.flush()
                qdone.add((name, q))
                print(f"[{name}] {counts.get(name, 0)}/{quota} sau query: {q}", flush=True)
    print(f"\nAPI calls trong lần chạy này: {api_calls}")
    print("category".ljust(18) + " ".join(k.rjust(8) for k in ["total", "results", "saved"] + REASONS))
    for n in names:
        s = stats[n]
        print(n.ljust(18) + " ".join(str(v).rjust(8) for v in
                                    [counts.get(n, 0), s["results"], s["saved"]] + [s[k] for k in REASONS]))
    print(f"Tổng ứng viên: {sum(counts.values())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "vn_crawl.yaml"))
    ap.add_argument("--root", default=str(ROOT / "data" / "vn_custom"))
    ap.add_argument("--categories", nargs="+")
    ap.add_argument("--dry-run", action="store_true", help="in query, không gọi mạng")
    ap.add_argument("--contact-sheet", action="store_true", help="dựng lưới ảnh mỗi category rồi thoát")
    ap.add_argument("--min-side", type=int, default=320)
    ap.add_argument("--max-mb", type=float, default=15)
    ap.add_argument("--max-hamming", type=int, default=6, help="dHash hamming <= giá trị này coi là trùng")
    ap.add_argument("--delay", type=float, default=1.1, help="giây giữa hai lần gọi API (tối thiểu 1.1)")
    ap.add_argument("--timeout", type=float, default=15)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.delay = max(args.delay, 1.1)
    root = Path(args.root)

    if args.contact_sheet:
        for p in contact_sheets(root):
            print(p)
        return
    crawl(args, load_cfg(Path(args.config)), root)


if __name__ == "__main__":
    main()
