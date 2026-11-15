import csv
import importlib.util
import os

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("crawl_vn_brave", os.path.join(ROOT, "scripts", "crawl_vn_brave.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


def _gradient(w=64, h=64, flip=False):
    im = Image.new("L", (w, h))
    im.putdata([(255 - x * 255 // (w - 1)) if flip else (x * 255 // (w - 1)) for _ in range(h) for x in range(w)])
    return im.convert("RGB")


def test_dhash_near_duplicate_vs_different():
    a = _gradient()
    b = a.resize((200, 200))  # cùng nội dung khác kích thước -> gần như cùng hash
    c = _gradient(flip=True)  # nội dung khác hẳn
    assert M.hamming(M.dhash(a), M.dhash(b)) <= 6
    assert M.hamming(M.dhash(a), M.dhash(c)) > 6


def test_dedupe_sha_and_dhash():
    dd = M.Dedupe(max_hamming=6)
    dd.add("ab" * 32, 0b1010)
    assert dd.check("ab" * 32, 0xFFFF_FFFF_FFFF_FFFF) == "dup_sha"
    assert dd.check("cd" * 32, 0b1010 ^ 0b11) == "dup_dhash"  # hamming 2
    assert dd.check("cd" * 32, 0xFFFF_FFFF_FFFF_FFFF) is None


def test_quota_and_title_filter():
    assert M.candidate_quota({"final_quota": 14}, 2) == 28
    assert M.candidate_quota({"final_quota": 14, "candidate_quota": 10}, 2) == 10
    assert M.title_filtered("Mẫu biển số VECTOR 2025", ["vector"])
    assert not M.title_filtered("Xe taxi biển vàng", ["vector", "logo"])


def _make_root(tmp_path, n=3):
    cand = tmp_path / "candidates"
    rows = []
    for i in range(n):
        sha = f"{i:02d}" * 32
        cid = M.candidate_id(sha)
        (cand / "night").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), (i * 40, 10, 10)).save(cand / "night" / f"{cid}.png")
        rows.append({"candidate_id": cid, "category": "night", "search_query": "q",
                     "source_page_url": f"http://p/{i}", "original_image_url": f"http://i/{i}.png",
                     "title": "t", "width": 40, "height": 30, "mime": "image/png", "sha256": sha,
                     "dhash": f"{i:016x}", "accessed_at": "x", "status": "hold"})
    M.write_csv_atomic(cand / "manifest.csv", M.MANIFEST_FIELDS, rows)
    return [r["candidate_id"] for r in rows]


def test_load_state_counts_resume(tmp_path):
    _make_root(tmp_path, 3)
    dd, counts = M.load_state(tmp_path / "candidates", 6)
    assert counts == {"night": 3}
    assert len(dd.shas) == 3 and len(dd.hashes) == 3


def test_contact_sheet(tmp_path):
    _make_root(tmp_path, 4)
    out = M.contact_sheets(tmp_path)
    assert len(out) == 1 and out[0].name == "night.jpg" and out[0].exists()


# ---------- mạng giả (mock requests.get, không mạng thật) ----------

import argparse
import io
import random

import pytest


class _Resp:
    def __init__(self, status=200, payload=None, body=b""):
        self.status_code, self._payload, self._body = status, payload, body
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True


def _png_bytes(seed, size=(64, 64)):
    im = Image.frombytes("L", size, random.Random(seed).randbytes(size[0] * size[1])).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def test_save_file_webp_to_png(tmp_path):
    im = Image.new("RGB", (40, 30), (200, 10, 10))
    buf = io.BytesIO()
    im.save(buf, "WEBP")
    out = M.save_file(buf.getvalue(), im, "webp", tmp_path / "vnc_x")
    assert out.suffix == ".png" and out.exists()
    with Image.open(out) as r:
        assert r.format == "PNG"


def test_fetch_image_size_cap(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(body=b"x" * 5000))
    with pytest.raises(ValueError, match="too large"):
        M.fetch_image("http://i/a.png", 1000, 1)
    ok = _png_bytes(1)
    resp = _Resp(body=ok)
    monkeypatch.setattr("requests.get", lambda *a, **k: resp)
    data, im, fmt = M.fetch_image("http://i/a.png", 10**6, 1)
    assert data == ok and fmt == "png" and resp.closed


def test_search_retries_on_429(monkeypatch):
    seq = [_Resp(429), _Resp(429), _Resp(200, {"results": [{"title": "a"}]})]
    calls = []
    monkeypatch.setattr("requests.get", lambda *a, **k: calls.append(1) or seq.pop(0))
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    assert M.search("k", "q", {}) == [{"title": "a"}]
    assert len(calls) == 3 and sleeps == [2, 4]


def _crawl_env(monkeypatch, tmp_path, per_query, quota):
    """per_query: {query: [seed,...]}. Trả (cfg, args, api_log)."""
    api_log = []

    def fake_get(url, params=None, **k):
        if url == M.API:
            api_log.append(params["q"])
            res = [{"title": "t", "url": f"http://p/{s}", "properties": {"url": f"http://i/{s}.png"}}
                   for s in per_query[params["q"]]]
            return _Resp(200, {"results": res})
        seed = int(url.rsplit("/", 1)[1].split(".")[0])
        return _Resp(200, body=_png_bytes(seed))

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("BRAVE_API_KEY", "dummy")
    cfg = {"categories": {"night": {"final_quota": 1, "candidate_quota": quota, "queries": list(per_query)}}}
    args = argparse.Namespace(categories=None, dry_run=False, delay=0, max_hamming=6, workers=2,
                              max_mb=1, timeout=1, min_side=10)
    return cfg, args, api_log


def test_crawl_quota_stop_and_resume(monkeypatch, tmp_path, capsys):
    per = {"q1": [1, 2], "q2": [3, 4, 5], "q3": [6, 7]}
    cfg, args, api_log = _crawl_env(monkeypatch, tmp_path, per, quota=3)
    M.crawl(args, cfg, tmp_path)
    assert api_log == ["q1", "q2"]  # đủ quota giữa q2: q3 không gọi
    rows = M.read_csv(tmp_path / "candidates" / "manifest.csv")
    assert len(rows) == 3 and all(r["file"].endswith(".png") for r in rows)
    assert all((tmp_path / "candidates" / r["file"]).exists() for r in rows)
    done = {r["query"] for r in M.read_csv(tmp_path / "candidates" / "queries_done.csv")}
    assert done == {"q1", "q2"}  # q2 bị quota cắt giữa chừng vẫn tính xong
    # resume cùng quota: không gọi API nào, không thêm dòng
    api_log.clear()
    M.crawl(args, cfg, tmp_path)
    assert api_log == [] and len(M.read_csv(tmp_path / "candidates" / "manifest.csv")) == 3
    # nâng quota lên 4: chỉ q3 chạy, dừng ở 4
    cfg["categories"]["night"]["candidate_quota"] = 4
    M.crawl(args, cfg, tmp_path)
    assert api_log == ["q3"] and len(M.read_csv(tmp_path / "candidates" / "manifest.csv")) == 4


def test_crawl_crash_mid_results_not_marked(monkeypatch, tmp_path):
    cfg, args, api_log = _crawl_env(monkeypatch, tmp_path, {"q1": [1, 2]}, quota=5)

    def boom(im):
        raise KeyboardInterrupt

    monkeypatch.setattr(M, "dhash", boom)
    with pytest.raises(KeyboardInterrupt):
        M.crawl(args, cfg, tmp_path)
    assert M.read_csv(tmp_path / "candidates" / "queries_done.csv") == []


def test_manifest_without_file_column(tmp_path):
    ids = _make_root(tmp_path, 2)
    cand = tmp_path / "candidates"
    old_fields = [f for f in M.MANIFEST_FIELDS if f != "file"]
    strip = lambda rs: [{k: v for k, v in r.items() if k != "file"} for r in rs]
    M.write_csv_atomic(cand / "manifest.csv", old_fields, strip(M.read_csv(cand / "manifest.csv")))
    assert "file" not in M.read_header(cand / "manifest.csv")
    dd, counts = M.load_state(cand, 6)  # đọc được khi thiếu cột
    assert counts == {"night": 2}
    assert len(M.contact_sheets(tmp_path)) == 1
    assert M.migrate_manifest(cand) == 2
    rows = M.read_csv(cand / "manifest.csv")
    assert [r["file"] for r in rows] == [f"night/{c}.png" for c in ids]
    assert M.migrate_manifest(cand) == 0  # idempotent
