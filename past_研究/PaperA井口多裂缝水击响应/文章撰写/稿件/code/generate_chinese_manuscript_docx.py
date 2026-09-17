from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
MANUSCRIPT = ROOT / "PaperA井口多裂缝水击响应/文章撰写/稿件/PaperA_SPE_Journal_初版中文稿.md"
EQ_DIR = ROOT / "PaperA井口多裂缝水击响应/文章撰写/稿件/docx_equations"
OUT = ROOT / "PaperA井口多裂缝水击响应/文章撰写/稿件/PaperA_SPE_Journal_中文版初稿_含图题.docx"


FIGURES = [
    ("图 1", ROOT / "PaperA井口多裂缝水击响应/文章撰写/稿件/图表数据/Figure_1_Local_vs_Network.png",
     "局部标签与网络观测。左图为被本文前向证据否定的朴素局部标签解释；右图表示井口观测是经过多个离散界面、下游反馈和二维倒谱算子后的网络条件化响应。"),
    ("图 2", ROOT / "PaperA井口多裂缝水击响应/文章撰写/稿件/图表数据/Figure_2_Forward_and_Operator.png",
     "前向模型和观测算子。(a) 集总裂缝节点的流量守恒与柔度/滤失关系；(b) 单簇基线与四簇网络的累积倒谱对比；(c) 名义射孔窗口、局部峰提取半径和网络条件化峰值。面板 (c) 使用模拟几何邻域，是基准峰值提取而非盲重建。"),
    ("图 3", ROOT / "PaperA井口多裂缝水击响应/文章撰写/Section3_可信度指数模型与方法论/figures/Figure_3_3_Physics_Informed_Dealiasing_Workflow.png",
     "物理信息去混叠算子示意。(a) 原始连续倒谱及名义射孔和非射孔路径候选；(b) 仅由前向波场定义的候选路径凹口与有限传输增益；(c) 有限增益和 0.80 m 超高斯聚焦候选核；(d) 先验/传输敏感性代理量 M_sens。图中采用固定的示意值 T_eff = 0.784 并对增益归一化显示，且未包含 eta(S)；该值不是由 426 个峰值表反演得到。该图是算子模板和代理量示意，不是盲恢复率验证或真实性概率。"),
    ("图 4", ROOT / "PaperA井口多裂缝水击响应/文章撰写/Section3_可信度指数模型与方法论/figures/Figure_3_4_MultiCase_Physics_Correction_Matrix.png",
     "四个极端工况的前向证据矩阵。(a) S = 30 m、n = 4 的低响应和 3120/3150 m 路径候选；(b) S = 10 m 的窗口重叠；(c) S = 20 m、n = 8 的深级联耗散；(d) S = 80 m 的高间距原始基线。绿色、黄色和灰色分别表示随簇序号变化的较高、歧义和低信息敏感度代理，未包含 eta(S)；酒红色表示非射孔路径候选。色带不表示真实状态概率。"),
    ("图 5", ROOT / "PaperA井口多裂缝水击响应/文章撰写/5.1_间距S主效应分析/figures/Figure_5_1_Spacing_Main_Effect.png",
     "间距主效应。(a) n = 4 的连续倒谱剖面随 S 的变化；(b) 各簇表观峰值 P_i(S)；(c) 相对响应 alpha_i = P_i/P_1。"),
    ("图 6", ROOT / "PaperA井口多裂缝水击响应/文章撰写/5.2_裂缝总数n主效应分析/figures/Figure_5_2_Multiplicity_Main_Effect.png",
     "簇数主效应和末端衰减。汇总不同 n 下的峰值、末首峰比和累计能量代理量，用于展示级联界面数的作用。"),
    ("图 7", ROOT / "PaperA井口多裂缝水击响应/文章撰写/5.3_首缝深度X1主效应分析/figures/Figure_5_3_Depth_Main_Effect.png",
     "首簇深度效应。比较 X_1 改变时的绝对峰值和相对响应，检验深度能否被一个独立标量增益表示。"),
    ("图 8", ROOT / "PaperA井口多裂缝水击响应/文章撰写/稿件/图表数据/Figure_8_Interaction_Phase_Maps.png",
     "间距、簇数与深度的交互。(a) P_1(S,n) 相图；(b) R_end(S,n) 相图；(c) 按簇数分组的 R_end-L_span 关系；(d) P_1(S,X_1) 相图；(e) alpha_2(S,X_1) 相图；(f) 不同 X_1 下的 alpha_2(S) 曲线。灰点表示经认证的离散网格，连续色面仅作插值可视化。"),
    ("图 9", ROOT / "PaperA井口多裂缝水击响应/文章撰写/5.6_空间衰减包络与模型拟合/figures/Figure_5_6_Spatial_vs_Topological_Decay.png",
     "固定几何衰减包络。展示 P_i = A exp[-gamma(i - 1)] 在固定几何内的描述性拟合；gamma 是工况内经验参数，不解释为普适透射系数。"),
    ("图 10", ROOT / "PaperA井口多裂缝水击响应/文章撰写/5.7_可辨识度边界与可行域评估/figures/Figure_5_7_Operational_Envelope.png",
     "可辨识度与操作包络。给出不同门限下的可识别簇数、表观信噪比和间距可行域。"),
]


def set_font(run, name="宋体", size=10.5, bold=False, italic=False, color=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_text(cell, text, bold=False, size=9.0):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(text)
    set_font(r, size=size, bold=bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_page_field(paragraph):
    run = paragraph.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.addnext(fld)


def inline_math_to_text(text: str) -> str:
    def clean(s: str) -> str:
        s = s.strip()
        s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\operatorname\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\(?:mathrm|operatorname|text)\s+", "", s)
        s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
        s = s.replace(r"\left", "").replace(r"\right", "")
        replacements = {
            r"\qquad": " ", r"\quad": " ", r"\ldots": "…", r"\Rightarrow": "⇒", r"\Delta": "Δ",
            r"\pi": "π", r"\in": "∈", r"\le": "≤", r"\ge": "≥", r"\approx": "≈", r"\times": "×",
            r"\sum": "Σ", r"\partial": "∂", r"\sqrt": "√", r"\exp": "exp", r"\min": "min", r"\max": "max",
            r"\eta": "η", r"\lambda": "λ", r"\rho": "ρ", r"\gamma": "γ", r"\kappa": "κ",
            r"\tau": "τ", r"\alpha": "α", r"\mathrm": "", r"\mathcal": "", r"\tilde": "~", r"\hat": "^", r"\bar": "¯",
        }
        for k, v in replacements.items():
            s = s.replace(k, v)
        s = s.replace("\\", "")
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\^\{([^{}])\}", r"^\1", s)
        s = re.sub(r"_\{([^{}])\}", r"_\1", s)
        s = s.replace("{", "").replace("}", "")
        return s

    text = re.sub(r"\\\((.*?)\\\)", lambda m: clean(m.group(1)), text)
    text = re.sub(r"\$([^$]+)\$", lambda m: clean(m.group(1)), text)
    text = text.replace("`", "")
    return text


def add_text_runs(paragraph, text, size=10.5):
    # Keep simple italic journal titles and bold labels readable.
    parts = re.split(r"(\*[^*]+\*)", text)
    for part in parts:
        if not part:
            continue
        italic = part.startswith("*") and part.endswith("*")
        value = part[1:-1] if italic else part
        value = inline_math_to_text(value)
        r = paragraph.add_run(value)
        set_font(r, size=size, italic=italic)


def add_body_paragraph(doc, text, style="Normal", size=10.5, first_line=True):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    p.paragraph_format.space_after = Pt(5)
    if first_line and style == "Normal":
        p.paragraph_format.first_line_indent = Cm(0.74)
    add_text_runs(p, text, size=size)
    return p


def add_markdown_table(doc, rows):
    values = [[inline_math_to_text(c.strip()) for c in row] for row in rows]
    table = doc.add_table(rows=len(values), cols=len(values[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            set_cell_text(table.cell(i, j), value, bold=(i == 0), size=8.8)
            if i == 0:
                set_cell_shading(table.cell(i, j), "D9EAF7")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def parse_and_add_body(doc, source_text):
    lines = source_text.splitlines()
    i = 1  # skip title
    eq_counter = 0
    references_mode = False
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("## 图件清单与图注"):
            break
        if not line:
            i += 1
            continue
        if line.startswith("## "):
            title = inline_math_to_text(line[3:].strip())
            references_mode = (title == "参考文献")
            if title == "摘要":
                p = doc.add_paragraph()
                p.style = "Heading 1"
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                r = p.add_run("摘要")
                set_font(r, name="黑体", size=14, bold=True)
            else:
                p = doc.add_paragraph(style="Heading 1")
                r = p.add_run(title)
                set_font(r, name="黑体", size=14, bold=True)
            i += 1
            continue
        if line.startswith("### "):
            p = doc.add_paragraph(style="Heading 2")
            r = p.add_run(inline_math_to_text(line[4:].strip()))
            set_font(r, name="黑体", size=12, bold=True)
            i += 1
            continue
        if line == "\\[":
            formula = []
            i += 1
            while i < len(lines) and lines[i].strip() != "\\]":
                formula.append(lines[i])
                i += 1
            eq_counter += 1
            img = EQ_DIR / f"equation_{eq_counter:02d}.png"
            if img.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.space_after = Pt(3)
                p.add_run().add_picture(str(img), width=Inches(5.8))
            else:
                add_body_paragraph(doc, inline_math_to_text(" ".join(formula)), first_line=False)
            i += 1
            continue
        if line.startswith("| ") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-{3,}", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|?\s*:?-{3,}", lines[i]):
                    rows.append([c for c in lines[i].strip().strip("|").split("|")])
                i += 1
            add_markdown_table(doc, rows)
            continue
        if re.match(r"^\d+\.\s+", line):
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            p.paragraph_format.space_after = Pt(4)
            add_text_runs(p, re.sub(r"^\d+\.\s+", "", line), size=10.5)
            i += 1
            continue
        if line.startswith("**关键词：**"):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(10)
            r = p.add_run("关键词：")
            set_font(r, name="黑体", size=10.5, bold=True)
            add_text_runs(p, line.replace("**关键词：**", "").strip(), size=10.5)
            i += 1
            continue
        if references_mode and re.match(r"^\[\d+\]", line):
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.08
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.first_line_indent = Cm(0.0)
            add_text_runs(p, line, size=9.2)
            i += 1
            continue
        if line.startswith("**") and line.endswith("**"):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            r = p.add_run(line.strip("*"))
            set_font(r, size=10.5, bold=True)
            i += 1
            continue
        add_body_paragraph(doc, line)
        i += 1


def add_figures(doc):
    doc.add_page_break()
    p = doc.add_paragraph(style="Heading 1")
    r = p.add_run("图件与题注")
    set_font(r, name="黑体", size=14, bold=True)
    add_body_paragraph(doc, "以下主图均为本文正文使用的 400 dpi PNG 版本，图题与正文中的图号保持一致。图件源文件同时保留可编辑 SVG 和 PDF 版本。", first_line=False)
    for number, path, caption in FIGURES:
        if not path.exists():
            add_body_paragraph(doc, f"{number}（图件缺失：{path.name}）", first_line=False)
            continue
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        with Image.open(path) as im:
            w, h = im.size
        max_w, max_h = 6.35, 7.0
        width = max_w
        height = width * h / w
        if height > max_h:
            height = max_h
            width = height * w / h
        p.add_run().add_picture(str(path), width=Inches(width), height=Inches(height))
        cap = doc.add_paragraph(style="Caption")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_before = Pt(2)
        cap.paragraph_format.space_after = Pt(10)
        rr = cap.add_run(f"{number}  {caption}")
        set_font(rr, name="宋体", size=9.5)


def configure_doc(doc):
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    for style_name, size in [("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11)]:
        st = styles[style_name]
        st.font.name = "黑体"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
    if "Caption" in styles:
        styles["Caption"].font.name = "宋体"
        styles["Caption"]._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        styles["Caption"].font.size = Pt(9.5)
        styles["Caption"].font.color.rgb = RGBColor(0, 0, 0)
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hr = header.add_run("多簇井筒非局部水击倒谱响应")
    set_font(hr, name="宋体", size=8.5, color=(100, 100, 100))
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = footer.add_run("第 ")
    set_font(fr, name="宋体", size=8.5, color=(100, 100, 100))
    add_page_field(footer)
    fr2 = footer.add_run(" 页")
    set_font(fr2, name="宋体", size=8.5, color=(100, 100, 100))


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else MANUSCRIPT
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_doc(doc)
    text = source.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("# ").strip()
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(title)
    set_font(r, name="黑体", size=18, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("中文版初稿（含主图与图片题注）")
    set_font(r, name="宋体", size=12, color=(80, 80, 80))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("2026 年 9 月")
    set_font(r, name="宋体", size=10, color=(100, 100, 100))
    doc.add_paragraph()
    parse_and_add_body(doc, text)
    add_figures(doc)
    doc.core_properties.title = title
    doc.core_properties.subject = "多簇井筒水击倒谱响应与物理信息重建"
    doc.core_properties.author = ""
    doc.save(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
