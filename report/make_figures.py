"""Sinh bảng LaTeX (report/tables/*.tex) và hình E3 (report/figures/e3_subset_r1.png) từ CSV kết quả.

Cách dùng: python report/make_figures.py <thư mục assets chứa tables/*.csv> [thư mục report]
Số liệu lấy nguyên từ CSV, làm tròn 3 chữ số (ms: 1 chữ số; corner: 1 chữ số).
"""
import csv
import sys
from pathlib import Path

SUBSETS = ["base", "blur", "challenge", "db", "tilt", "ALL"]
STAGES = ["loc_no_candidate", "loc_rank", "rectify", "segment_count", "segment_position", "ok"]


def esc(s):
    return str(s).replace("_", r"\_")


def rd(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f3(x):
    return f"{float(x):.3f}"


def f1(x):
    return f"{float(x):.1f}"


def tab(cols, header, rows, out):
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        if r == "MID":
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    Path(out).write_text("\n".join(lines), encoding="utf-8")


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
                rows.append([sp, name, r["n"], f3(r["precision"]), f3(r["recall"]), f3(r["f1"]),
                             f3(r["recall_at_5"]), f1(r["ms_mean"])])
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
                    rows.append([esc(m), sb, r["n"], f3(r["precision"]), f3(r["recall"]), f3(r["f1"]),
                                 f3(r["recall_at_1"]), f3(r["recall_at_3"]), f3(r["recall_at_5"])])
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
                    ce = "--" if not r["corner_mean"] else f"{float(r['corner_mean']):.1f} / {float(r['corner_median']):.1f}"
                    rows.append([esc(cs), cm, r["n"], ce, f3(r["count_match"]), f3(r["pos0_acc"]),
                                 f3(r["pos1_6_acc"]), f3(r["all_correct"])])
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
        rows.append([sb, b["n"], f1(b["corner_mean"]), f1(q["corner_mean"]), f3(b["count_match"]),
                     f3(q["count_match"]), f3(b["all_correct"]), f3(q["all_correct"])])
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
                rows.append([nm, r["n"], f"{float(r['ms_mean']):.2f}", f"{float(r['ms_p95']):.2f}"])
    tab("lrrr", ["Bước", "$n$", "tb (ms)", "p95 (ms)"], rows, tdir / "t_e7.tex")

    # triage: bước x subset
    cnt = {}
    for r in rd(src / "triage_stage_counts.csv"):
        cnt[(r["stage"], r["subset"])] = (r["n"], r["pct"], r["n_subset"])
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
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    v1 = [float(sel[("edge", s)]["recall_at_1"]) for s in SUBSETS]
    v2 = [float(sel[("edge_svm", s)]["recall_at_1"]) for s in SUBSETS]
    v5 = [float(sel[("edge", s)]["recall_at_5"]) for s in SUBSETS]
    ax.bar(x - w / 2, v1, w, color="#a7c7ee", label="edge, R@1")
    ax.bar(x + w / 2, v2, w, color="#c3a6e0", label="edge_svm, R@1")
    ax.scatter(x, v5, marker="_", s=900, color="#444444", zorder=3, label="R@5 (trần xếp hạng)")
    for xi, a, b in zip(x, v1, v2):
        ax.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\nn={sel[('edge', s)]['n']}" for s in SUBSETS], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall (IoU >= 0.5), tập test")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    (rep / "figures").mkdir(exist_ok=True)
    fig.savefig(rep / "figures" / "e3_subset_r1.png")
    print("ok")


if __name__ == "__main__":
    main()
