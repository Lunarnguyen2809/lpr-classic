r"""Kiểm tra nhanh các file .tex (không thay thế biên dịch thật).

Dùng: python report/check_tex.py <thư mục report>
Kiểm tra: cân bằng \begin/\end, ngoặc nhọn, đường dẫn \input và \includegraphics,
ký tự _ # chưa escape ngoài math, ký tự % chưa escape ngoài comment, & ngoài bảng,
macro cần gói (\mathbb, \toprule, subfigure, ...) mà preamble chưa \usepackage.
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
files = ([root / "main.tex"] + sorted((root / "sections").glob("*.tex"))
         + sorted((root / "tables").glob("*.tex")))
errors = []

TABLE_ENVS = ("tabular", "align", "array")


def strip_comment(line):
    out, i = [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if c == "%":
            break
        out.append(c)
        i += 1
    return "".join(out)


def strip_math(s):
    s = re.sub(r"\\\[.*?\\\]", "", s, flags=re.S)
    s = re.sub(r"\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", "", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", "", s)
    return s


for f in files:
    raw = f.read_text(encoding="utf-8").splitlines()
    lines = [strip_comment(line) for line in raw]
    text = "\n".join(lines)

    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", text):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        elif not stack or stack[-1] != m.group(2):
            errors.append(f"{f}: end {m.group(2)} không khớp (stack={stack[-3:]})")
        else:
            stack.pop()
    if stack:
        errors.append(f"{f}: thiếu end cho {stack}")

    t2 = re.sub(r"\\[{}]", "", text)
    if t2.count("{") != t2.count("}"):
        errors.append(f"{f}: ngoặc nhọn lệch {t2.count('{')} vs {t2.count('}')}")

    for m in re.finditer(r"\\input\{([^}]*)\}", text):
        p = root / m.group(1)
        if not (p.exists() or Path(str(p) + ".tex").exists()):
            errors.append(f"{f}: input không tồn tại: {m.group(1)}")
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text):
        if not (root / m.group(1)).exists():
            errors.append(f"{f}: includegraphics không tồn tại: {m.group(1)}")

    # ký tự đặc biệt (bỏ math, bỏ tham số của lệnh không in ra)
    plain = strip_math(text)
    plain = re.sub(r"\\(ref|label|input|includegraphics|usepackage|usetikzlibrary|documentclass|"
                   r"setmainfont|setsansfont|setmonofont|hypersetup)(\[[^\]]*\])?\{[^}]*\}(\[[^\]]*\])?", "", plain)
    in_tab = f.parent.name == "tables"
    depth = 0
    for n, line in enumerate(plain.splitlines(), 1):
        for m in re.finditer(r"\\begin\{(%s)\}" % "|".join(TABLE_ENVS), line):
            depth += 1
        for ch in ("_", "#"):
            if ch == "#" and "newcommand" in line:
                continue
            for m in re.finditer(re.escape(ch), line):
                if m.start() > 0 and line[m.start() - 1] == "\\":
                    continue
                errors.append(f"{f}: '{ch}' chưa escape: {line.strip()[:80]}")
        for m in re.finditer("&", line):
            if m.start() > 0 and line[m.start() - 1] == "\\":
                continue
            if depth <= 0 and not in_tab:
                errors.append(f"{f}: '&' ngoài bảng: {line.strip()[:80]}")
        for m in re.finditer(r"\\end\{(%s)\}" % "|".join(TABLE_ENVS), line):
            depth -= 1
    # dòng gốc: % không phải comment được strip_comment xử lý; kiểm tra '%' lẻ trong text đã strip
    if re.search(r"(?<!\\)%", text):
        errors.append(f"{f}: còn ký tự % chưa escape")

# macro -> gói cần có (kiểm tra danh sách nhỏ, mở rộng khi cần)
NEEDS = [(r"\\math(?:bb|frak)\b|\\checkmark\b|\\varnothing\b", "amssymb"),
         (r"\\begin\{(?:align|gather|cases)\*?\}|\\text\{", "amsmath"),
         (r"\\(?:top|mid|bottom)rule\b", "booktabs"), (r"\\begin\{subfigure\}", "subcaption"),
         (r"\\(?:resizebox|includegraphics)\b", "graphicx"), (r"\\(?:textcolor|color)\b", "xcolor"),
         (r"\\begin\{tikzpicture\}", "tikz")]
body = "\n".join(strip_comment(line) for f in files for line in f.read_text(encoding="utf-8").splitlines())
pre = (root / "main.tex").read_text(encoding="utf-8")
pkgs = set()
for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", pre):
    pkgs.update(x.strip() for x in m.group(1).split(","))
for pat, pkg in NEEDS:
    if re.search(pat, body) and pkg not in pkgs:
        errors.append(f"main.tex: dùng macro khớp /{pat}/ nhưng thiếu " + "\\usepackage{" + pkg + "}")

print("\n".join(errors) if errors else "OK: không phát hiện lỗi")
sys.exit(1 if errors else 0)
