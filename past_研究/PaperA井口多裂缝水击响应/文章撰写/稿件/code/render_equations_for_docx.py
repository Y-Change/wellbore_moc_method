from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def clean_formula(tex: str) -> str:
    """Make the manuscript's display TeX compatible with matplotlib mathtext."""
    tex = " ".join(tex.split())
    tex = re.sub(r"\\tag\{[^}]*\}", "", tex)
    tex = tex.replace("\\begin{aligned}", "").replace("\\end{aligned}", "")
    tex = tex.replace("\\begin{array}{ll}", "").replace("\\end{array}", "")
    tex = tex.replace(r"\operatorname{clip}", r"\mathrm{clip}")
    tex = re.sub(r"\\text\{([^{}]*)\}", lambda m: r"\mathrm{" + m.group(1) + "}", tex)
    tex = re.sub(r"\\substack\{([^{}]*)\}", lambda m: r"\mathrm{" + m.group(1) + "}", tex)
    tex = tex.replace("\\left", "").replace("\\right", "")
    tex = tex.replace("\\displaystyle", "")
    tex = tex.replace(r"\,", r"\, ").replace(r"\;", r"\; ")
    tex = tex.replace(r"\operatorname", r"\mathrm")
    tex = tex.replace(r"\mathop", "")
    tex = re.sub(r"\\mathcal\s+([A-Za-z])", lambda m: r"\mathrm{" + m.group(1) + "}", tex)
    tex = tex.replace(r"\le", r"\leq").replace(r"\ge", r"\geq")
    tex = re.sub(r"\\mathrm\s+([A-Za-z]+)", r"\1", tex)
    # Mathtext does not need line-break commands for the compact equations used here.
    tex = tex.replace("\\\\", r"\quad")
    return tex


def fallback_text(tex: str) -> str:
    text = " ".join(tex.split())
    text = re.sub(r"\\(?:mathrm|operatorname|text)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    return text


def render(tex: str, out: Path, number: int) -> None:
    formula = clean_formula(tex)
    fig = plt.figure(figsize=(10.5, 0.95), dpi=300)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    try:
        ax.text(0.5, 0.52, f"${formula}$", ha="center", va="center", fontsize=18, color="black")
        fig.canvas.draw()
    except Exception:
        ax.clear()
        ax.axis("off")
        # The fallback remains mathematical notation without exposing raw TeX in the DOCX.
        ax.text(0.5, 0.52, fallback_text(tex), ha="center", va="center", fontsize=14, color="black")
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: render_equations_for_docx.py manuscript.md output_dir", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    formulas = re.findall(r"\\\[(.*?)\\\]", text, flags=re.S)
    for i, formula in enumerate(formulas, 1):
        render(formula, out_dir / f"equation_{i:02d}.png", i)
    print(f"rendered {len(formulas)} display equations to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
