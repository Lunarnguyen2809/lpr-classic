r"""Kiểm tra nhanh các file .tex (không thay thế biên dịch thật).

Dùng: python report/check_tex.py <thư mục report>
Kiểm tra:
1. gói chỉ dành cho XeLaTeX/LuaLaTeX (fontspec, xeCJK, ...) bị cấm vì báo cáo biên dịch bằng pdfLaTeX;
2. ký tự Unicode không an toàn với pdfLaTeX (T5 + inputenc utf8): chỉ cho phép chữ Latin (gồm dấu tiếng Việt);
   ký tự như dấu lớn hơn hoặc bằng, mũi tên, gạch dài, chữ Hán phải viết bằng lệnh LaTeX;
3. cân bằng \begin/\end, ngoặc nhọn, đường dẫn \input/\include/\includegraphics, ký tự _ # chưa escape ngoài math,
   ký tự % chưa escape ngoài comment, & ngoài bảng, macro cần gói (\mathbb, \toprule, subfigure, ...)
   mà preamble chưa \usepackage.
"""
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
files = ([root / "main.tex"] + sorted((root / "chapters").glob("*.tex"))
         + sorted((root / "tables").glob("*.tex")))
errors = []

TABLE_ENVS = ("tabular", "tabularx", "align", "array")
FORBIDDEN_PKGS = {"fontspec", "xeCJK", "xecjk", "unicode-math", "polyglossia", "luatexja", "luacode",
                  "xltxtra", "xunicode", "mathspec", "xepersian", "ctex"}


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

    # (1) gói chỉ có trên XeLaTeX/LuaLaTeX và lệnh font của fontspec
    for n, line in enumerate(raw, 1):
        code = strip_comment(line)
        for m in re.finditer(r"\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{([^}]*)\}", code):
            for p in (x.strip() for x in m.group(1).split(",")):
                if p in FORBIDDEN_PKGS:
                    errors.append(f"{f}:{n}: gói '{p}' chỉ chạy trên XeLaTeX/LuaLaTeX, không dùng với pdfLaTeX")
        if re.search(r"\\(?:setmainfont|setsansfont|setmonofont|newfontfamily|setCJKmainfont)\b", code):
            errors.append(f"{f}:{n}: lệnh fontspec không dùng được với pdfLaTeX")

    # (2) ký tự Unicode ngoài chữ Latin
    for n, line in enumerate(raw, 1):
        for ch in line:
            if ord(ch) < 128:
                continue
            name = unicodedata.name(ch, "?")
            if not name.startswith("LATIN"):
                errors.append(f"{f}:{n}: ký tự không an toàn với pdfLaTeX/T5: {ch!r} U+{ord(ch):04X} ({name}); "
                              f"thay bằng lệnh LaTeX (vd. $\\geq$, $\\rightarrow$, \\textendash)")

    # (3) các kiểm tra cấu trúc
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

    for m in re.finditer(r"\\(?:input|include)\{([^}]*)\}", text):
        p = root / m.group(1)
        if not (p.exists() or Path(str(p) + ".tex").exists()):
            errors.append(f"{f}: input/include không tồn tại: {m.group(1)}")
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text):
        if not (root / m.group(1)).exists():
            errors.append(f"{f}: includegraphics không tồn tại: {m.group(1)}")

    # ký tự đặc biệt (bỏ math, bỏ tham số của lệnh không in ra)
    plain = strip_math(text)
    plain = re.sub(r"\\(ref|label|cite|input|include|includegraphics|usepackage|usetikzlibrary|documentclass|"
                   r"PassOptionsToPackage|font|hypersetup|bibitem)(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})?(\[[^\]]*\])?",
                   "", plain)
    in_tab = f.parent.name == "tables"
    depth = 0
    for n, line in enumerate(plain.splitlines(), 1):
        for m in re.finditer(r"\\begin\{(%s)\}" % "|".join(TABLE_ENVS), line):
            depth += 1
        for ch in ("_", "#"):
            if ch == "#" and f.name == "main.tex" and re.search(r"#[1-9]", line):
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
    if re.search(r"(?<!\\)%", text):
        errors.append(f"{f}: còn ký tự % chưa escape")

# macro -> gói cần có (kiểm tra danh sách nhỏ, mở rộng khi cần)
NEEDS = [(r"\\math(?:bb|frak)\b|\\checkmark\b|\\varnothing\b", "amssymb"),
         (r"\\begin\{(?:align|gather|cases)\*?\}|\\text\{", "amsmath"),
         (r"\\(?:top|mid|bottom)rule\b", "booktabs"), (r"\\begin\{subfigure\}", "subcaption"),
         (r"\\(?:resizebox|includegraphics)\b", "graphicx"), (r"\\(?:textcolor|color)\b", "xcolor"),
         (r"\\(?:rowcolor|cellcolor)\b", "colortbl"), (r"\\rowcolors\b", "xcolor"),
         (r"\\begin\{algorithmic\}", "algpseudocode"), (r"\\begin\{algorithm\}", "algorithm"),
         (r"\\begin\{tikzpicture\}", "tikz")]
body = "\n".join(strip_comment(line) for f in files for line in f.read_text(encoding="utf-8").splitlines())
pre = (root / "main.tex").read_text(encoding="utf-8")
pkgs = set()
for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", pre):
    pkgs.update(x.strip() for x in m.group(1).split(","))
for pat, pkg in NEEDS:
    if re.search(pat, body) and pkg not in pkgs:
        errors.append(f"main.tex: dùng macro khớp /{pat}/ nhưng thiếu " + "\\usepackage{" + pkg + "}")

# nhãn: mọi \ref/\cite có đích
labels = set(re.findall(r"\\label\{([^}]*)\}", body))
bibs = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", body))
for m in re.finditer(r"\\ref\{([^}]*)\}", body):
    if m.group(1) not in labels:
        errors.append(f"\\ref{{{m.group(1)}}} không có \\label")
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    for k in m.group(1).split(","):
        if k.strip() not in bibs:
            errors.append(f"\\cite{{{k.strip()}}} không có \\bibitem")

print("\n".join(errors) if errors else "OK: không phát hiện lỗi")
sys.exit(1 if errors else 0)
