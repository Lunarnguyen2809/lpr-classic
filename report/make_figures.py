"""Sinh bảng LaTeX (report/tables/*.tex) và hình E3 (report/figures/e3_subset_r1.png) từ CSV kết quả.

Cách dùng: python report/make_figures.py <thư mục assets chứa tables/*.csv> [thư mục report]
Quy tắc làm tròn (một lần, half-up, dùng chung cho mọi bảng): P, R, F1 và các tỉ lệ lấy theo số đếm
(tp/fp/fn, count nguyên của n) rồi mới làm tròn 3 chữ số, không làm tròn lại từ giá trị 4 chữ số trong CSV
(vd. 148/198 = 0.7475 -> 0.747, 1/16 = 0.0625 -> 0.063). Chỉ khi không suy ra được số đếm mới dùng giá trị CSV.
ms và corner: làm tròn half-up từ giá trị CSV (1 chữ số; E7: 2 chữ số).
"""
import csv
import sys
from decimal import ROUND_HALF_UP, Decimal, getcontext
from fractions import Fraction
from pathlib import Path

getcontext().prec = 40

SUBSETS = ["base", "blur", "challenge", "db", "tilt", "ALL"]
STAGES = ["loc_no_candidate", "loc_rank", "rectify", "segment_count", "segment_position", "ok"]


def esc(s):
    return str(s).replace("_", r"\_")


def rd(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def hu(x, nd):
    """Làm tròn half-up đúng một lần; x là Fraction hoặc chuỗi/số thập phân."""
    if isinstance(x, Fraction):
        d = Decimal(x.numerator) / Decimal(x.denominator)
    else:
        d = Decimal(str(x))
    return str(d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def f3(x):
    return hu(x, 3)


def f1(x):
    return hu(x, 1)


def frac(num, den, fallback):
    """Tỉ lệ num/den làm tròn 3 chữ số; den = 0 thì dùng giá trị CSV."""
    return f3(Fraction(int(num), int(den))) if int(den) else f3(fallback)


def from_rate(rate, den):
    """Đổi tỉ lệ CSV (4 chữ số) về số đếm nguyên trên mẫu số den rồi làm tròn lại một lần.

    Nếu rate*den không gần số nguyên (mẫu số khác) thì trả về giá trị CSV làm tròn."""
    r = float(rate)
    k = round(r * den)
    if abs(r * den - k) < 0.5e-4 * den + 1e-9 and den:
        return f3(Fraction(k, den))
    return f3(rate)


def prf(r):
    tp, fp, fn = int(r["tp"]), int(r["fp"]), int(r["fn"])
    return (frac(tp, tp + fp, r["precision"]), frac(tp, tp + fn, r["recall"]),
            frac(2 * tp, 2 * tp + fp + fn, r["f1"]))


def tab(cols, header, rows, out):
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        if r == "MID":
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    Path(out).write_text("\n".join(lines), encoding="utf-8")


def e4rates(r):
    """count_match, pos0, pos1_6, all_correct từ số đếm (n ảnh; pos1_6 trên 6n ký tự)."""
    n = int(r["n"])
    return (from_rate(r["count_match"], n), from_rate(r["pos0_acc"], n), from_rate(r["pos1_6_acc"], 6 * n),
            from_rate(r["all_correct"], n))


def E4ROW(r, cs, cm, ce):
    cmatch, p0, p16, allc = e4rates(r)
    return [cs, cm, r["n"], ce, cmatch, p0, p16, allc]


def main():
    src = Path(sys.argv[1]) / "tables"
    rep = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent
    tdir = rep / "tables"
    tdir.mkdir(exist_ok=True)

    # E2: bộ lọc (dòng ALL)
    rows = []
    e2 = rd(src / "e2_filters.csv")
    for sp in ("tune", "test"):
        for r in e2:
            if r["split"] == sp and r["subset"] == "ALL":
                name = esc(r["denoise"])
                if r["chosen"] == "True":
                    name = r"\textbf{" + name + "}"
                P, R, F = prf(r)
                rows.append([sp, name, r["n"], P, R, F, from_rate(r["recall_at_5"], int(r["n"])),
                             f1(r["ms_mean"])])
        if sp == "tune":
            rows.append("MID")
    tab("llrrrrrr", ["Tập", "Bộ lọc", "$n$", "P", "R", "F1", "R@5", "ms/ảnh"], rows, tdir / "t_e2.tex")

    # E3: định vị theo subset, tập test
    e3 = rd(src / "e3_localization.csv")
    has_split = "split" in e3[0]
    rows = []
    for m in ("edge", "edge_svm"):
        for sb in SUBSETS:
            for r in e3:
                if has_split and r["split"] != "test":
                    continue
                if r["method"] == m and r["subset"] == sb:
                    n = int(r["n"])
                    P, R, F = prf(r)
                    rows.append([esc(m), sb, r["n"], P, R, F, from_rate(r["recall_at_1"], n),
                                 from_rate(r["recall_at_3"], n), from_rate(r["recall_at_5"], n)])
        if m == "edge":
            rows.append("MID")
    tab("llrrrrrrr", ["Phương pháp", "Nhóm", "$n$", "P", "R", "F1", "R@1", "R@3", "R@5"], rows,
        tdir / "t_e3.tex")

    # E4: ALL theo crop_source x crop_method
    rows = []
    e4 = rd(src / "e4_rectify.csv")
    for cs in ("gt", "pred", "pred_tp"):
        for cm in ("bbox", "quad"):
            for r in e4:
                if r["crop_source"] == cs and r["crop_method"] == cm and r["subset"] == "ALL":
                    ce = "--" if not r["corner_mean"] else f"{f1(r['corner_mean'])} / {f1(r['corner_median'])}"
                    rows.append(E4ROW(r, esc(cs), cm, ce))
        if cs != "pred_tp":
            rows.append("MID")
    tab("llrrrrrr", ["Nguồn crop", "Crop", "$n$", "corner (tb / trung vị)", "count\\_match", "pos0", "pos1--6",
                     "all\\_correct"], rows, tdir / "t_e4.tex")

    # E4: pred_tp theo subset
    rows = []
    for sb in SUBSETS:
        d = {}
        for r in e4:
            if r["crop_source"] == "pred_tp" and r["subset"] == sb:
                d[r["crop_method"]] = r
        b, q = d["bbox"], d["quad"]
        rows.append([sb, b["n"], f1(b["corner_mean"]), f1(q["corner_mean"]), e4rates(b)[0],
                     e4rates(q)[0], e4rates(b)[3], e4rates(q)[3]])
    tab("lrrrrrrr", ["Nhóm", "$n$", "corner bbox", "corner quad", "count bbox", "count quad", "all bbox",
                     "all quad"], rows, tdir / "t_e4_subset.tex")

    # E7: ALL
    names = {"preprocess": "tiền xử lý (resize, xám, CLAHE, lọc)",
             "localize_other": "Sobel, morph đa tỉ lệ, lọc, NMS (ước lượng)",
             "hog_svm": "HOG + SVM", "quad": "ước lượng quad", "rectify": "warp 220$\\times$70",
             "binarize": "nhị phân hóa (B)", "segment": "tách ký tự (B)",
             "total_edge": "tổng (không SVM)", "total": "tổng"}
    rows = []
    e7 = rd(src / "e7_runtime.csv")
    for st in names:
        for r in e7:
            if r["step"] == st and r["subset"] == "ALL":
                nm = names[st]
                if st.startswith("total"):
                    nm = r"\textbf{" + nm + "}"
                if st == "total_edge":
                    rows.append("MID")
                rows.append([nm, r["n"], hu(r["ms_mean"], 2), hu(r["ms_p95"], 2)])
    tab("lrrr", ["Bước", "$n$", "tb (ms)", "p95 (ms)"], rows, tdir / "t_e7.tex")

    # triage: bước x subset
    cnt = {}
    for r in rd(src / "triage_stage_counts.csv"):
        cnt[(r["stage"], r["subset"])] = (r["n"], hu(Fraction(100 * int(r["n"]), int(r["n_subset"])), 1),
                                          r["n_subset"])
    rows = []
    for st in STAGES:
        row = [esc(st)]
        for sb in SUBSETS:
            row.append(cnt[(st, sb)][0])
        row.append(cnt[(st, "ALL")][1])
        rows.append(row)
    hdr = ["Bước"] + [f"{sb} ({cnt[('ok', sb)][2]})" for sb in SUBSETS] + ["\\% ALL"]
    tab("lrrrrrrr", hdr, rows, tdir / "t_triage.tex")

    # triage: bảng chéo (cộng các nhóm base..tilt, bỏ dòng ALL để không đếm đôi)
    agg = {}
    for r in rd(src / "triage_gt_vs_pred.csv"):
        if r["subset"] == "ALL":
            continue
        k = (r["gt_status"], r["pred_stage"])
        agg[k] = agg.get(k, 0) + int(r["n"])
    rows = []
    for g in ("ok", "segment_count", "segment_position"):
        tot = sum(agg.get((g, p), 0) for p in STAGES)
        rows.append([f"{esc(g)} ({tot})"] + [str(agg.get((g, p), 0)) for p in STAGES])
    tab("lrrrrrr", ["crop gt $\\backslash$ bước pred"] + [esc(s) for s in STAGES], rows,
        tdir / "t_triage_cross.tex")

    # hình E3
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    sel = {}
    for r in e3:
        if has_split and r["split"] != "test":
            continue
        sel[(r["method"], r["subset"])] = r
    x = np.arange(len(SUBSETS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    v1 = [float(sel[("edge", s)]["recall_at_1"]) for s in SUBSETS]
    v2 = [float(sel[("edge_svm", s)]["recall_at_1"]) for s in SUBSETS]
    v5 = [float(sel[("edge", s)]["recall_at_5"]) for s in SUBSETS]
    ax.bar(x - w / 2, v1, w, color="#a7c7ee", label="edge, R@1")
    ax.bar(x + w / 2, v2, w, color="#c3a6e0", label="edge_svm, R@1")
    ax.scatter(x, v5, marker="_", s=900, color="#444444", zorder=3, label="R@5 (trần xếp hạng)")
    for xi, a, b in zip(x, v1, v2):  # nhãn giá trị đặt trong thân cột để không đè vạch R@5
        ax.text(xi - w / 2, a - 0.05, f"{a:.2f}", ha="center", va="center", fontsize=7, color="#222222")
        ax.text(xi + w / 2, b - 0.05, f"{b:.2f}", ha="center", va="center", fontsize=7, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\nn={sel[('edge', s)]['n']}" for s in SUBSETS], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall (IoU >= 0.5), tập test")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    (rep / "figures").mkdir(exist_ok=True)
    fig.savefig(rep / "figures" / "e3_subset_r1.png")
    print("ok")


if __name__ == "__main__":
    main()
