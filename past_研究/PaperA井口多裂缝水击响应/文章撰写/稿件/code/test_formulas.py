import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import render_equations_for_docx as m
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

text = (Path(__file__).resolve().parents[1] / "PaperA_SPE_Journal_初版中文稿.md").read_text(encoding="utf-8")
for i, f in enumerate(re.findall(r"\\\[(.*?)\\\]", text, re.S), 1):
    s = m.clean_formula(f)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, f"${s}$", ha="center")
    try:
        fig.canvas.draw()
        print(i, "OK", repr(s[:140]))
    except Exception as exc:
        print(i, "FAIL", repr(s[:140]), str(exc).splitlines()[-1])
    plt.close(fig)
