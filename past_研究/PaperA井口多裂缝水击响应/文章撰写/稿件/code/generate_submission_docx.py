#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication-ready submission DOCX manuscripts for Paper A:
1. Chinese submission version: PaperA_SPE_Journal_中文投稿版_可编辑公式含图.docx
2. English submission version: PaperA_SPE_Journal_English_Submission_EditableMath.docx

Features:
- Native editable Word Office Math (OMML) for display equations (1)-(12)
- In-line figure placement directly following first citation in text
- Three-line tables for data and case comparisons
- Complete physical Nomenclature table
- Single-column submission layout with standard margins, 1.5 line spacing, headers/footers
"""

from __future__ import annotations

import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
import latex2mathml.converter


# ---------------------------------------------------------------------------
# OMML (Office Math) Generator from LaTeX
# ---------------------------------------------------------------------------
MATHML_NS = "{http://www.w3.org/1998/Math/MathML}"


def _m_elem(name: str):
    return OxmlElement(f"m:{name}")


def _m_text_run(text: str):
    run = _m_elem("r")
    properties = _m_elem("rPr")
    style = _m_elem("sty")
    style.set(qn("m:val"), "p")
    properties.append(style)
    run.append(properties)
    value = _m_elem("t")
    value.text = text
    run.append(value)
    return run


def _append_children(target, source) -> None:
    if source.text and source.text.strip():
        target.append(_m_text_run(source.text.strip()))
    for child in source:
        _append_mathml(target, child)
        if child.tail and child.tail.strip():
            target.append(_m_text_run(child.tail.strip()))


def _script(target, node, kind: str) -> None:
    result = _m_elem(kind)
    expression = _m_elem("e")
    sub = _m_elem("sub")
    sup = _m_elem("sup")
    children = list(node)
    if children:
        _append_mathml(expression, children[0])
    if len(children) > 1:
        _append_mathml(sub, children[1])
    if len(children) > 2:
        _append_mathml(sup, children[2])
    result.append(expression)
    if kind in {"sSub", "sSubSup"}:
        result.append(sub)
    if kind in {"sSup", "sSubSup"}:
        result.append(sup if kind == "sSubSup" else sub)
    target.append(result)


def _append_mathml(target, node) -> None:
    tag = node.tag.removeprefix(MATHML_NS)
    children = list(node)

    if tag in {"math", "mrow", "mstyle", "semantics", "annotation"}:
        _append_children(target, node)
    elif tag in {"mi", "mn", "mo", "mtext"}:
        target.append(_m_text_run("".join(node.itertext())))
    elif tag == "mfrac":
        fraction = _m_elem("f")
        numerator = _m_elem("num")
        denominator = _m_elem("den")
        if children:
            _append_mathml(numerator, children[0])
        if len(children) > 1:
            _append_mathml(denominator, children[1])
        fraction.extend((numerator, denominator))
        target.append(fraction)
    elif tag == "msub":
        _script(target, node, "sSub")
    elif tag == "msup":
        _script(target, node, "sSup")
    elif tag in {"msubsup", "munderover"}:
        _script(target, node, "sSubSup")
    elif tag == "munder":
        _script(target, node, "sSub")
    elif tag == "mover":
        _script(target, node, "sSup")
    elif tag == "msqrt":
        radical = _m_elem("rad")
        properties = _m_elem("radPr")
        hide_degree = _m_elem("degHide")
        hide_degree.set(qn("m:val"), "1")
        properties.append(hide_degree)
        degree = _m_elem("deg")
        expression = _m_elem("e")
        _append_children(expression, node)
        radical.extend((properties, degree, expression))
        target.append(radical)
    elif tag == "mroot":
        radical = _m_elem("rad")
        degree = _m_elem("deg")
        expression = _m_elem("e")
        if children:
            _append_mathml(expression, children[0])
        if len(children) > 1:
            _append_mathml(degree, children[1])
        radical.extend((degree, expression))
        target.append(radical)
    elif tag == "mfenced":
        delimiter = _m_elem("d")
        properties = _m_elem("dPr")
        begin = _m_elem("begChr")
        begin.set(qn("m:val"), node.attrib.get("open", "("))
        end = _m_elem("endChr")
        end.set(qn("m:val"), node.attrib.get("close", ")"))
        properties.extend((begin, end))
        expression = _m_elem("e")
        _append_children(expression, node)
        delimiter.extend((properties, expression))
        target.append(delimiter)
    elif tag == "mtable":
        matrix = _m_elem("m")
        for row_node in children:
            row = _m_elem("mr")
            for cell_node in list(row_node):
                cell = _m_elem("e")
                _append_children(cell, cell_node)
                row.append(cell)
            matrix.append(row)
        target.append(matrix)
    elif tag in {"mtr", "mtd"}:
        _append_children(target, node)
    elif tag == "mspace":
        target.append(_m_text_run(" "))
    else:
        _append_children(target, node)


def latex_to_omml_math(latex: str):
    """Convert LaTeX string to m:oMath element."""
    clean_tex = latex.strip()
    clean_tex = re.sub(r"\\tag\{.*?\}", "", clean_tex).strip()
    clean_tex = clean_tex.replace(r"\operatorname{clip}", r"\mathrm{clip}")
    clean_tex = clean_tex.replace(r"\operatorname", r"\mathrm")
    clean_tex = clean_tex.replace(r"\mathrm d", r"\mathrm{d}")
    clean_tex = clean_tex.replace(r"\mathrm dt", r"\mathrm{d}t")
    
    mathml_str = latex2mathml.converter.convert(clean_tex)
    mathml_xml = ElementTree.fromstring(mathml_str)
    
    math = _m_elem("oMath")
    _append_mathml(math, mathml_xml)
    return math


# ---------------------------------------------------------------------------
# Word Formatting & Typography Helpers
# ---------------------------------------------------------------------------
def set_font(run, name="Times New Roman", east_asia="宋体", size=10.5, bold=False, italic=False, color=None):
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:eastAsia"), east_asia)
    rFonts.set(qn("w:cs"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def clean_inline_math(text: str) -> list[tuple[str, bool, bool, bool]]:
    """
    Parse inline text with \(...\) or $...$ into runs of:
    (content_text, is_italic, is_subscript, is_superscript)
    """
    pattern = r"(\\\(.*?\\\)|\$[^\$]+\$)"
    parts = re.split(pattern, text)
    result = []

    for part in parts:
        if not part:
            continue
        if (part.startswith(r"\(") and part.endswith(r"\)")) or (part.startswith("$") and part.endswith("$")):
            raw_math = part[2:-2] if part.startswith(r"\(") else part[1:-1]
            raw_math = raw_math.strip()
            tokens = _parse_math_tokens(raw_math)
            result.extend(tokens)
        else:
            result.append((part, False, False, False))
    return result


def _parse_math_tokens(math_str: str) -> list[tuple[str, bool, bool, bool]]:
    s = math_str
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "").replace(r"\quad", " ")
    s = s.replace(r"\le", "≤").replace(r"\ge", "≥").replace(r"\approx", "≈").replace(r"\times", "×")
    s = s.replace(r"\in", "∈").replace(r"\pm", "±").replace(r"\to", "→").replace(r"\ldots", "…")
    s = s.replace(r"\alpha", "α").replace(r"\beta", "β").replace(r"\gamma", "γ").replace(r"\Delta", "Δ")
    s = s.replace(r"\delta", "δ").replace(r"\lambda", "λ").replace(r"\eta", "η").replace(r"\kappa", "κ")
    s = s.replace(r"\rho", "ρ").replace(r"\tau", "τ").replace(r"\pi", "π").replace(r"\sigma", "σ")
    s = s.replace(r"\phi", "φ").replace(r"\psi", "ψ").replace(r"\omega", "ω").replace(r"\mu", "μ")
    
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1/\2)", s)
    
    runs = []
    token_pat = r"(\\mathrm\{[^{}]*\}|\\mathcal\{[^{}]*\}|\\tilde\{[^{}]*\}|\\hat\{[^{}]*\}|_[0-9A-Za-z]|_\{[^{}]+\}|\^[0-9A-Za-z]|\^\{[^{}]+\}|[A-Za-z]+|[0-9]+|[^A-Za-z0-9_\^]+)"
    
    for m in re.finditer(token_pat, s):
        tok = m.group(0)
        if tok.startswith(r"\mathrm{") and tok.endswith("}"):
            runs.append((tok[8:-1], False, False, False))
        elif tok.startswith(r"\mathcal{") and tok.endswith("}"):
            runs.append((tok[9:-1], True, False, False))
        elif tok.startswith(r"\tilde{") and tok.endswith("}"):
            runs.append((tok[7:-1] + "~", True, False, False))
        elif tok.startswith(r"\hat{") and tok.endswith("}"):
            runs.append((tok[5:-1] + "^", True, False, False))
        elif tok.startswith("_"):
            sub_val = tok[2:-1] if tok.startswith("_{") else tok[1:]
            sub_val = sub_val.replace(r"\mathrm{", "").replace("}", "")
            runs.append((sub_val, False, True, False))
        elif tok.startswith("^"):
            sup_val = tok[2:-1] if tok.startswith("^{") else tok[1:]
            sup_val = sup_val.replace(r"\mathrm{", "").replace("}", "")
            runs.append((sup_val, False, False, True))
        elif re.match(r"^[A-Za-zα-ω]$", tok):
            runs.append((tok, True, False, False))
        elif re.match(r"^[A-Za-z]{2,}$", tok):
            runs.append((tok, False, False, False))
        else:
            runs.append((tok, False, False, False))
            
    return runs if runs else [(s, False, False, False)]


def add_formatted_runs(paragraph, text: str, default_font="Times New Roman", east_asia="宋体", size=10.5, default_bold=False):
    tokens = clean_inline_math(text)
    for content, italic, is_sub, is_sup in tokens:
        if not content:
            continue
        run = paragraph.add_run(content)
        font_name = "Cambria Math" if (italic or is_sub or is_sup) else default_font
        set_font(run, name=font_name, east_asia=east_asia, size=size, bold=default_bold, italic=italic)
        if is_sub:
            run.font.subscript = True
        if is_sup:
            run.font.superscript = True


def add_equation_block(doc, latex_str: str, eq_num_str: str):
    """
    Insert a display formula as a borderless 1-row x 2-col table:
    Left cell: centered OMML equation
    Right cell: right-aligned equation number e.g. (1)
    """
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement("w:tblBorders")
    for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        b_node = OxmlElement(f"w:{border_name}")
        b_node.set(qn("w:val"), "none")
        tblBorders.append(b_node)
    tblPr.append(tblBorders)

    cell_eq = table.cell(0, 0)
    cell_num = table.cell(0, 1)

    cell_eq.width = Inches(5.8)
    cell_num.width = Inches(0.7)

    p_eq = cell_eq.paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.space_before = Pt(4)
    p_eq.paragraph_format.space_after = Pt(4)

    math_elem = latex_to_omml_math(latex_str)
    p_eq._p.append(math_elem)

    p_num = cell_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(4)
    p_num.paragraph_format.space_after = Pt(4)
    r_num = p_num.add_run(eq_num_str)
    set_font(r_num, name="Times New Roman", size=10.5)

    cell_eq.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    cell_num.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_figure_block(doc, img_path: Path, caption_label: str, caption_title: str, caption_text: str, lang="zh"):
    """
    Insert a figure and its caption into the document.
    """
    if not img_path.exists():
        print(f"Warning: image path does not exist: {img_path}")
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    run_img.add_picture(str(img_path), width=Inches(6.0))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    p_cap.paragraph_format.line_spacing = 1.15

    r_lbl = p_cap.add_run(caption_label)
    set_font(r_lbl, name="Times New Roman" if lang == "en" else "黑体",
             east_asia="黑体", size=9.5, bold=True)

    if caption_title:
        r_title = p_cap.add_run(f" {caption_title} ")
        set_font(r_title, name="Times New Roman" if lang == "en" else "黑体",
                 east_asia="黑体", size=9.5, bold=True)

    add_formatted_runs(p_cap, caption_text,
                       default_font="Times New Roman",
                       east_asia="宋体",
                       size=9.5)


def add_three_line_table(doc, headers: list[str], rows: list[list[str]], caption_label: str, caption_title: str, lang="zh", col_widths=None):
    """
    Create an academic three-line table (top line 1.5pt, header bottom 0.75pt, table bottom 1.5pt).
    """
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(4)
    r_lbl = p_cap.add_run(caption_label)
    set_font(r_lbl, name="Times New Roman" if lang == "en" else "黑体", east_asia="黑体", size=10, bold=True)
    if caption_title:
        r_tit = p_cap.add_run(f" {caption_title}")
        set_font(r_tit, name="Times New Roman" if lang == "en" else "黑体", east_asia="黑体", size=10, bold=True)

    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement("w:tblBorders")
    
    top = OxmlElement("w:top")
    top.set(qn("w:val"), "single")
    top.set(qn("w:sz"), "12")  # 1.5 pt
    top.set(qn("w:space"), "0")
    top.set(qn("w:color"), "000000")
    tblBorders.append(top)

    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "12")  # 1.5 pt
    bottom.set(qn("w:space"), "0")
    bottom.set(qn("w:color"), "000000")
    tblBorders.append(bottom)

    for b in ["left", "right", "insideH", "insideV"]:
        node = OxmlElement(f"w:{b}")
        node.set(qn("w:val"), "none")
        tblBorders.append(node)
    tblPr.append(tblBorders)

    hdr_cells = table.rows[0].cells
    for j, h in enumerate(headers):
        hdr_cells[j].text = ""
        p = hdr_cells[j].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        add_formatted_runs(p, h, default_font="Times New Roman", east_asia="宋体", size=9.5, default_bold=True)
        
        tcPr = hdr_cells[j]._tc.get_or_add_tcPr()
        tcBorders = OxmlElement("w:tcBorders")
        b_node = OxmlElement("w:bottom")
        b_node.set(qn("w:val"), "single")
        b_node.set(qn("w:sz"), "6")  # 0.75 pt
        b_node.set(qn("w:color"), "000000")
        tcBorders.append(b_node)
        tcPr.append(tcBorders)

        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "F2F4F7")
        tcPr.append(shd)

    for i, row in enumerate(rows):
        row_cells = table.rows[i + 1].cells
        for j, val in enumerate(row):
            row_cells[j].text = ""
            p = row_cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            add_formatted_runs(p, val, default_font="Times New Roman", east_asia="宋体", size=9.0)

    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Inches(w)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def setup_document_page_layout(doc):
    for sec in doc.sections:
        sec.top_margin = Cm(2.54)
        sec.bottom_margin = Cm(2.54)
        sec.left_margin = Cm(2.54)
        sec.right_margin = Cm(2.54)
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(11.0)
        
        header = sec.header
        p_hdr = header.paragraphs[0]
        p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_hdr.paragraph_format.space_after = Pt(0)
        r_hdr = p_hdr.add_run("Paper A | Multi-Cluster Wellbore Water Hammer Response")
        set_font(r_hdr, name="Times New Roman", size=8.5, color=(128, 128, 128))

        footer = sec.footer
        p_ftr = footer.paragraphs[0]
        p_ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_ftr.paragraph_format.space_after = Pt(0)
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), "PAGE")
        p_ftr._p.append(fld)


# ---------------------------------------------------------------------------
# Metadata Tables (Nomenclature)
# ---------------------------------------------------------------------------
NOMENCLATURE_ZH = [
    (r"H(x,t)", "井筒水头（测压管水头）", "m"),
    (r"Q(x,t)", "井筒瞬态体积流量", r"m³/s"),
    (r"a", "流体压力水击波速", "m/s"),
    (r"D", "井筒套管内径", "m"),
    (r"A", r"井筒过流截面积 (\pi D^2/4)", "m²"),
    (r"g", "重力加速度 (9.8 m/s²)", "m/s²"),
    (r"f", "Darcy-Weisbach 沿程摩阻系数", "无量纲"),
    (r"Q_{f,k}", r"第 k 簇水力裂缝节点进液流量", r"m³/s"),
    (r"C_{f,k}", r"第 k 簇集总裂缝柔度", "m²"),
    (r"k_{\mathrm{leak},k}", r"第 k 簇压裂液滤失系数", r"m^{5/2}/s"),
    (r"H_{f,k}", r"第 k 簇裂缝内流体水头", "m"),
    (r"H_{\mathrm{ext}}", "储层远场外部基底水头", "m"),
    (r"X_1", "首簇射孔名义深度", "m"),
    (r"S", "射孔簇均匀空间间距", "m"),
    (r"n", "压裂段内设计射孔簇总数", "无量纲"),
    (r"x_{\mathrm{perf},k}", r"第 k 簇名义射孔设计深度", "m"),
    (r"C(x,t)", "二维时间-深度边际倒谱场", "a.u."),
    (r"P_i", r"第 i 簇名义位置处的提取表观倒谱峰值", "a.u."),
    (r"\alpha_i", r"归一化相对幅值比 (P_i/P_1)", "无量纲"),
    (r"R_{\mathrm{end}}", r"末首峰比 (P_n/P_1)", "无量纲"),
    (r"L_{\mathrm{span}}", r"射孔簇总跨度 (n-1)S", "m"),
    (r"\gamma", "固定几何下的经验空间衰减指数参数", "1/簇"),
    (r"z_k", r"第 k 簇非负真实进液活跃度（状态未知）", "无量纲"),
    (r"h_k(x)", r"第 k 簇名义深度单位前向脉冲响应模板", "a.u."),
    (r"g_m(x)", "预测非射孔多路径伪峰模板", "a.u."),
    (r"c_m", "预测多路径伪峰幅值系数", "无量纲"),
    (r"G_{\mathrm{peel}}", "物理信息多路径剥离算子", "无量纲"),
    (r"K_{\mathrm{comp}}", "传输与多界面相位物理补偿算子", "无量纲"),
    (r"T_{\mathrm{eff}}", "簇间有效透射系数", "无量纲"),
    (r"\eta(S)", "间距依赖的相消相干调制因子", "无量纲"),
    (r"F_{\mathrm{focus}}", "拟合活跃簇的超高斯空间聚焦算子", "无量纲"),
    (r"w_{\mathrm{core}}", "超高斯聚焦核特征宽度 (0.80 m)", "m"),
    (r"M(x)", "连续证据置信度与敏感度掩膜", "无量纲"),
    (r"\tilde P(x)", "物理信息非循环重建后的最终倒谱剖面", "a.u."),
]

NOMENCLATURE_EN = [
    (r"H(x,t)", "Hydraulic piezometric head", "m"),
    (r"Q(x,t)", "Volumetric flow rate", r"m³/s"),
    (r"a", "Acoustic water-hammer wave speed", "m/s"),
    (r"D", "Internal casing diameter", "m"),
    (r"A", r"Wellbore cross-sectional area (\pi D^2/4)", "m²"),
    (r"g", "Gravitational acceleration", "m/s²"),
    (r"f", "Darcy--Weisbach friction factor", "dimensionless"),
    (r"Q_{f,k}", r"Volumetric fluid-entry rate into fracture cluster k", r"m³/s"),
    (r"C_{f,k}", r"Lumped fracture compliance for cluster k", "m²"),
    (r"k_{\mathrm{leak},k}", r"Lumped leak-off coefficient for cluster k", r"m^{5/2}/s"),
    (r"H_{f,k}", r"Hydraulic head inside fracture at cluster k", "m"),
    (r"H_{\mathrm{ext}}", "External reservoir baseline head", "m"),
    (r"X_1", "Nominal depth of first perforation cluster", "m"),
    (r"S", "Uniform perforation cluster spacing", "m"),
    (r"n", "Total number of designed perforation clusters", "dimensionless"),
    (r"x_{\mathrm{perf},k}", r"Nominal designed perforation depth for cluster k", "m"),
    (r"C(x,t)", "2D time--depth marginal cepstrum field", "a.u."),
    (r"P_i", r"Apparent cepstral peak extracted at cluster i", "a.u."),
    (r"\alpha_i", r"Normalized apparent peak ratio (P_i/P_1)", "dimensionless"),
    (r"R_{\mathrm{end}}", r"Terminal-to-first peak amplitude ratio (P_n/P_1)", "dimensionless"),
    (r"L_{\mathrm{span}}", r"Total perforation interval span (n-1)S", "m"),
    (r"\gamma", "Empirical spatial decay parameter", r"cluster^{-1}"),
    (r"z_k", r"Non-negative fluid-entry activity factor for cluster k", "dimensionless"),
    (r"h_k(x)", r"Forward unit impulse response template for cluster k", "a.u."),
    (r"g_m(x)", "Forward multipath reverberation ghost template", "a.u."),
    (r"c_m", "Ghost peak amplitude coefficient", "dimensionless"),
    (r"G_{\mathrm{peel}}", "Physics-informed multipath peeling operator", "dimensionless"),
    (r"K_{\mathrm{comp}}", "Forward transmission and phase compensation operator", "dimensionless"),
    (r"T_{\mathrm{eff}}", "Inter-cluster effective transmission factor", "dimensionless"),
    (r"\eta(S)", "Spacing-dependent wave interference factor", "dimensionless"),
    (r"F_{\mathrm{focus}}", "Super-Gaussian spatial focusing operator", "dimensionless"),
    (r"w_{\mathrm{core}}", "Characteristic width of super-Gaussian core (0.80 m)", "m"),
    (r"M(x)", "Continuous evidence and reliability uncertainty mask", "dimensionless"),
    (r"\tilde P(x)", "Physics-informed reconstructed cepstral profile", "a.u."),
]


# ---------------------------------------------------------------------------
# Chinese Submission Manuscript Builder
# ---------------------------------------------------------------------------
def build_chinese_docx(base_dir: Path, out_path: Path):
    print(f"Building Chinese submission manuscript: {out_path.name}...")
    doc = Document()
    setup_document_page_layout(doc)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(8)
    r_title = p_title.add_run("多簇井筒非局部水击倒谱响应：机理证伪与物理信息非循环重建框架")
    set_font(r_title, name="黑体", east_asia="黑体", size=18, bold=True)

    # Authors Placeholder
    p_author = doc.add_paragraph()
    p_author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_author.paragraph_format.space_after = Pt(3)
    r_author = p_author.add_run("作者姓名 1,2*，合作作者 1，通讯作者 2")
    set_font(r_author, name="宋体", east_asia="宋体", size=11, bold=True)

    # Affiliations Placeholder
    p_affil = doc.add_paragraph()
    p_affil.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_affil.paragraph_format.space_after = Pt(12)
    r_affil = p_affil.add_run("（1. 油气藏地质及开发工程全国重点实验室，四川 成都 610500；\n2. 石油工程学院，北京 100083）\n* 通讯作者邮箱：corresponding_author@domain.edu.cn")
    set_font(r_affil, name="宋体", east_asia="宋体", size=9.0, italic=True)

    # Abstract Box / Paragraph
    p_abs_title = doc.add_paragraph()
    p_abs_title.paragraph_format.space_before = Pt(6)
    p_abs_title.paragraph_format.space_after = Pt(2)
    r_abs_t = p_abs_title.add_run("【摘要】")
    set_font(r_abs_t, name="黑体", east_asia="黑体", size=10.5, bold=True)

    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    p_abs.paragraph_format.space_after = Pt(6)
    p_abs.paragraph_format.first_line_indent = Cm(0.74)
    abs_text = (
        "井口水击瞬态压力记录能够在无需下入井下阵列仪器的条件下，用于评估多簇压裂阶段实际进液破裂的射孔簇，"
        "但多簇井筒中的波场响应并非简单的局部量。下游水力裂缝界面会强烈改变上游表观峰；中等簇间距可能因界面波干涉使后序峰的衰减显著强于更小或更大的间距；"
        "多次内反射还会在未设计射孔的深度产生显著的多路径伪峰。本文采用一维特征线法（method of characteristics，MOC）和二维时间-深度边际倒谱，"
        "对覆盖 426 个经过认证的独立拓扑网格进行系统正演与定量分析。结果表明：固定首簇深度 X_1=3000 m、簇间距 S=20 m 时，增加第二簇使首簇表观倒谱峰由 5.157 a.u. "
        "骤降至 2.705 a.u.，降幅高达 47.5%，直接证伪了无网络条件修正的单簇标定假设；在 n=4 时，S=30 m 的第三、第四峰因界面相消作用分别降至 0.536 和 0.324 a.u.，"
        "而在 S=100 m 时恢复至 1.21 和 0.92 a.u.；总跨度同为 60 m 时，n=2, 3, 4, 7 的末首峰比 R_end 分别为 0.472、0.188、0.158 和 0.072，表明总跨度并非充分描述变量；"
        "连续倒谱剖面在约 3120 m 和 3150 m 处清晰检测到与多路径时延严格一致的次级伪峰。\n"
        "基于上述物理证伪，本文提出了由多路径剥离算子 G_peel、有限物理增益传输补偿算子 K_comp、超高斯空间聚焦算子 F_focus 以及连续置信度掩膜 M(x) "
        "构成的物理信息非循环重建框架。将名义射孔深度界定为已知工程先验，而将实际进液状态定义为非负未知量。非负活动量约束与有限增益设计确保了严格的“零保持性”，"
        "彻底避免了传统局部峰值除法凭空生成假裂缝的问题。四个极端几何工况验证了前向失真特征与算子边界。本文结果为压裂后水击监测提供了严谨的物理证据链与标准化反演接口。"
    )
    add_formatted_runs(p_abs, abs_text, default_font="宋体", east_asia="宋体", size=10.0)

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_after = Pt(12)
    r_kwt = p_kw.add_run("【关键词】")
    set_font(r_kwt, name="黑体", east_asia="黑体", size=10.5, bold=True)
    r_kw = p_kw.add_run("水击；多簇压裂；特征线法；倒谱；多路径混响；物理信息反问题；不确定性掩膜；零保持性")
    set_font(r_kw, name="宋体", east_asia="宋体", size=10.0)

    # Nomenclature Table
    p_nom_h = doc.add_paragraph()
    p_nom_h.paragraph_format.space_before = Pt(8)
    p_nom_h.paragraph_format.space_after = Pt(4)
    r_nh = p_nom_h.add_run("符号说明表 (Nomenclature)")
    set_font(r_nh, name="黑体", east_asia="黑体", size=12, bold=True)

    nom_rows = [[s, d, u] for s, d, u in NOMENCLATURE_ZH]
    add_three_line_table(doc, ["符号", "物理含义与定义", "SI 计量单位"], nom_rows,
                         caption_label="表 0", caption_title="文中所用主要物理变量与算子符号说明",
                         lang="zh", col_widths=[1.5, 4.0, 1.0])

    # Section 1
    _add_heading(doc, "1 引言", level=1, lang="zh")
    _add_para(doc, "多簇压裂是现代非常规低渗透油气藏水平井高效开发的核心工艺。压裂施工效果直接取决于设计射孔簇中究竟有哪些簇真正破裂并有效接收了压裂支撑剂与压裂液。井下微地震成像和分布式光纤传感（DTS/DAS）能够提供直观的进液与应变信息；已有管波研究表明，裂缝水力柔度、Krauklis 共振波和界面反射波也能从井筒波场中得到有效约束 [1,2]。然而，受限于高昂的作业成本与恶劣的井下井温井压约束，施工现场迫切需要基于井口地面监测的无侵入式诊断手段。关井或停泵引发的水击（Water Hammer）水力瞬变波在井筒内沿程传播，连续采样沿程阻抗与流动截面变化，并可变换到时延或深度表示，因而展现出重大的工程应用前景 [3,4]。近期研究已进一步推进到射孔簇进液深度检测、簇级压裂事件识别以及超高频井口监测装备研发 [5–7]。", lang="zh")

    _add_para(doc, "传统水击信号解释方法通常建立在同态倒谱（Homomorphic Cepstrum）与局部反射假设基础之上 [8,9]：通过将某一特定时延峰直接对应于某一个射孔簇，并将相应倒谱峰幅值视为该簇裂缝进液程度的局部测量代理量。然而，这种解释仅在裂缝与井筒网络耦合极弱时才近似成立。实际的多簇水平井筒是由多个离散裂缝节点组成的强连通阻抗界面网络：波在任一簇界面的透射和反射不仅会改变入射到所有下游簇的波场能量与相位，更会反向叠加到返回井口的复合信号中。因此，在总跨度相同而界面数不同的网络中，末端响应存在巨大差异；簇间距改变会同时改变波形幅值而绝非仅有到时平移 [10–13]。这一“局部标签与网络观测”的根本差异如图 1 所示。", lang="zh")

    # In-line Figure 1
    fig1_path = base_dir / "图表数据/Figure_1_Local_vs_Network.png"
    add_figure_block(doc, fig1_path, "图 1｜", "局部标签与网络观测示意图。",
                     "左图为传统文献中被本文前向证据证伪的朴素局部标签解释（假设每个倒谱峰幅值仅反映该处裂缝能量）；"
                     "右图为本文揭示的物理真实：井口观测是经过多个离散阻抗界面、上游-下游往复反馈和二维倒谱算子变换后的连通网络条件化响应。",
                     lang="zh")

    _add_para(doc, "针对上述矛盾，本文着重回答两个核心物理问题：第一，基于严格受控的井筒 MOC 网格前向模拟，能够证伪哪些关于局部性、间距效应和总跨度的传统默认假设？第二，在已知射孔枪设计位置的前提下，如何去除可预测的多路径混响伪峰，而不预先假设哪些设计簇已经实际破裂？第二个问题在数学上被严格表述为一个受约束的物理信息反问题：名义工程射孔深度为已知先验，而实际进液活化状态为非负未知量。若将峰值直接强行贴合到所有名义位置，将陷入循环论证的逻辑陷阱。", lang="zh")

    _add_para(doc, "本文基于项目中经过认证的峰值数据库（01_几何网格/峰值表/decay_table.csv）以及 5.1 至 5.7 节的系列分析。全部分析的独立物理单元是拓扑 MOC 网格，而非单个时间采样或倒谱箱。本研究前向设计空间扫描是确定性数值解，不包含现场随机噪声或经验重复实验，因此严格不报告无统计意义的 p 值。本文的贡献被严格界定为一组可证伪的网络前向物理证据，以及一套可在未来盲基准上检验的非循环算子规范；近期贝叶斯诊断、裂缝尺寸反演、天然裂缝模型、连续小波衰减分析与自动参数反演研究提供了问题背景，但不替代本研究的前向证据 [14–18]。", lang="zh")

    # Section 2
    _add_heading(doc, "2 前向模型与观测算子", level=1, lang="zh")
    _add_heading(doc, "2.1 控制方程与节点边界", level=2, lang="zh")
    _add_para(doc, "水平井筒被模化为具有恒定横截面积、弱可压缩流体的一维分布参数管道。瞬态压力波场采用测压管水头 H(x,t) 与体积流量 Q(x,t) 表示。连续性方程与动量守恒方程写为：", lang="zh")

    # Equation 1
    eq1_tex = r"\frac{\partial H}{\partial t}+\frac{a^2}{gA}\frac{\partial Q}{\partial x}=0, \qquad \frac{\partial Q}{\partial t}+gA\frac{\partial H}{\partial x}+\frac{f}{2DA}Q|Q|=0 ."
    add_equation_block(doc, eq1_tex, "(1)")

    _add_para(doc, "式中：a 为水中声速，A=\\pi D^2/4 为井筒截面积，D 为内径，g 为重力加速度，f 为 Darcy--Weisbach 沿程摩阻系数。频变与工程非定常摩阻模型详见文献 [19–21]。本文主基准采用定常摩阻假定，f 由初始流动雷诺数决定，不随瞬时速度脉动动态更新；非定常 Brunone 摩阻不属于本文核心结论边界。", lang="zh")
    _add_para(doc, "在各个射孔簇节点 x_k 处，质量守恒以及集总裂缝柔度与压裂液滤失关系满足（图 2(a)）：", lang="zh")

    # Equation 2
    eq2_tex = r"Q_{L,k}-Q_{R,k}=Q_{f,k},\qquad Q_{f,k}=C_{f,k}\frac{\mathrm d H_{f,k}}{\mathrm dt}+k_{\mathrm{leak},k}\sqrt{H_{f,k}-H_{\mathrm{ext}}} ."
    add_equation_block(doc, eq2_tex, "(2)")

    _add_para(doc, "式中：C_{f,k} 为第 k 簇的集总水力裂缝柔度，k_{\\mathrm{leak},k} 为储层基质滤失系数，H_{\\mathrm{ext}} 为外部远场储层水头。基准计算中各簇均赋予相同的标称 C_f 和 k_\\mathrm{leak}，以剥离裂缝异质性干扰，专注考察网络拓扑与几何间距效应。MOC 沿特征线积分，Courant 数精确取 1。标准仿真参数为：D=0.1397 m，a=1450 m/s，流体密度 1000 kg/m³，运动黏度 10⁻⁶ m²/s，管壁粗糙度 4.5×10⁻⁵ m，初始稳态流速 1.0 m/s，初始井口水头 300 m，时间步长 \\Delta t=10⁻³ s，C_f=10⁻⁵ m²，k_\\mathrm{leak}=10⁻⁴ m^{5/2}/s，泵切断停泵时间为 1.0 s。", lang="zh")

    # In-line Figure 2
    fig2_path = base_dir / "图表数据/Figure_2_Forward_and_Operator.png"
    add_figure_block(doc, fig2_path, "图 2｜", "前向动力学模型与倒谱观测算子。",
                     "(a) 离散射孔簇节点的流量守恒与集总柔度/滤失本构边界示意；"
                     "(b) 单簇基线与四簇网络在井口接收到的时间-深度累积倒谱剖面响应对比；"
                     "(c) 名义射孔窗口、局部峰值提取半径 r 及网络条件化表观峰值提取逻辑。"
                     "注意：面板 (c) 中的窗口搜索基于真实几何邻域，仅用于前向基准特征提取，并不构成反问题的盲定位验证。",
                     lang="zh")

    _add_heading(doc, "2.2 设计空间与 426 个独立拓扑网格", level=2, lang="zh")
    _add_para(doc, "压裂段设计空间由名义射孔深度完全确定：", lang="zh")

    # Equation 3
    eq3_tex = r"x_{\mathrm{perf},k}=X_1+(k-1)S,\qquad k=1,\ldots,n ."
    add_equation_block(doc, eq3_tex, "(3)")

    _add_para(doc, "参数空间覆盖：首簇深度 X_1 \\in \\{2000, 2500, 3000, 3500, 4000, 4500\\} m，簇间距 S \\in \\{10, 20, \\ldots, 100\\} m，簇数 n \\in \\{1, \\ldots, 8\\}。在数据库 decay_table.csv 中，定常摩阻下 n=2~8 构成了 6×7×10=420 个多簇拓扑网格，单簇对照基线在 6 个不同深度提供了 6 个独立网格，累计形成 426 个独立拓扑仿真实例。", lang="zh")

    _add_heading(doc, "2.3 边际倒谱观测与局部表观峰提取", level=2, lang="zh")
    _add_para(doc, "井口瞬态水头时间序列经过 30 s 的 Hamming 窗以 5 s 的滑动步长进行同态变换 [22]，得到二维时间-深度边际倒谱场 C(x,t)。在基准物理分析中，对应于第 i 个名义簇设计深度的表观峰值提取算子定义为：", lang="zh")

    # Equation 4
    eq4_tex = r"P_i=\max_{|x-x_i|\le r}\left[-\sum_t C(x,t)\right],\qquad r=\min(15\,\mathrm m,0.49S) ."
    add_equation_block(doc, eq4_tex, "(4)")

    _add_para(doc, "式中：x_i 为数值模拟中的真实设定深度。P_i 本质上是网络输出在局部的投影表观峰，绝非该簇吸收能量的直接纯净测度。本文进一步定义归一化相对幅值 \\alpha_i=P_i/P_1、末首峰比 R_{\\mathrm{end}}=P_n/P_1、总压裂跨度 L_{\\mathrm{span}}=(n-1)S，以及用于定性描述的经验空间包络 P_i=A\\exp[-\\gamma(i-1)]。衰减参数 \\gamma 仅是特定工况下的表观拟合斜率，不能等同于单界面的物理透射率。", lang="zh")

    # Section 3
    _add_heading(doc, "3 机理证伪试验", level=1, lang="zh")
    _add_heading(doc, "3.1 下游反馈否定局部首峰标签", level=2, lang="zh")
    _add_para(doc, "传统观念认为首簇距离井口最近，其反射波最先返回且未经历下游界面的穿透，因而其峰值应保持不变。然而数值实验彻底否定了该假设。如图 6 所示，在 X_1=3000 m、S=20 m 的基准工况下，单簇 (n=1) 时的首峰幅值为 5.157 a.u.；当在下游 20 m 处增加第二簇 (n=2) 时，首簇表观峰直接骤降至 2.705 a.u.，降幅高达 47.545%！在全部多簇网格中，首峰在 2.45 至 3.55 a.u. 之间随下游拓扑剧烈震荡。这充分证实下游簇界面的往复波场反馈会反向重构首峰形态，否定了单簇标定在多簇环境下的直接外推有效性。", lang="zh")

    # In-line Figure 6
    fig6_path = base_dir / "../5.2_裂缝总数n主效应分析/figures/Figure_5_2_Multiplicity_Main_Effect.png"
    add_figure_block(doc, fig6_path, "图 6｜", "射孔簇总数 n 的主效应与末端衰减特征。",
                     "汇总不同簇数 n 下各簇表观峰值 P_i、末首峰比 R_end 以及累计衰减能量随界面级联数量的变化规律，"
                     "直观展现了下游界面对上游响应的强反馈与能量递增性耗散。",
                     lang="zh")

    _add_heading(doc, "3.2 间距同时改变幅值与时延（界面相消陷阱）", level=2, lang="zh")
    _add_para(doc, "传统时延模型认为簇间距 S 仅仅改变波沿程传播的双程旅行时（Δt=2S/a），不改变波形自身幅值。表 1 与图 5 汇总了 X_1=3000 m, n=4 定常摩阻下的峰值响应。数据显示，随着间距从 10 m 增加至 100 m，后序簇幅值呈现非单调的“V 型谷”剧烈调制：在 S=30 m 时，第三、第四簇表观峰跌入极深低谷（分别仅为 0.536 和 0.324 a.u.），而在更紧密的 S=10 m 下峰值反而较高（0.964 和 0.634 a.u.），当 S 扩大至 100 m 时又恢复至 1.209 和 0.920 a.u.。双簇对照组的响应在各间距下相对平坦（P_1≈2.70~2.74 a.u.，P_2≈1.24~1.30 a.u.）。这一现象强力证明：在 n≥3 的级联网络中，特定的间距尺度会激发强烈的多界面相消干涉（Destructive Interference），使得后序真实裂缝表观信号几近湮灭。", lang="zh")

    # Table 1
    t1_headers = ["间距 S (m)", "首簇峰 P1 (a.u.)", "第二簇峰 P2 (a.u.)", "第三簇峰 P3 (a.u.)", "第四簇峰 P4 (a.u.)", "末首峰比 Rend"]
    t1_rows = [
        ["10", "2.738", "1.421", "0.964", "0.634", "0.2316"],
        ["20", "2.844", "1.317", "0.614", "0.450", "0.1582"],
        ["30", "3.122", "1.442", "0.536", "0.324", "0.1038"],
        ["40", "3.050", "1.514", "0.596", "0.327", "0.1072"],
        ["50", "3.063", "1.541", "0.609", "0.436", "0.1423"],
        ["60", "3.088", "1.464", "0.592", "0.437", "0.1415"],
        ["70", "3.040", "1.527", "1.064", "0.693", "0.2280"],
        ["80", "3.090", "1.581", "1.158", "0.878", "0.2841"],
        ["100", "3.159", "1.705", "1.209", "0.920", "0.2912"],
    ]
    add_three_line_table(doc, t1_headers, t1_rows, caption_label="表 1",
                         caption_title="X1=3000 m, n=4 定常摩阻下不同簇间距对表观倒谱峰值的非单调调制",
                         lang="zh", col_widths=[1.0, 1.1, 1.1, 1.1, 1.1, 1.1])

    # In-line Figure 5
    fig5_path = base_dir / "../5.1_间距S主效应分析/figures/Figure_5_1_Spacing_Main_Effect.png"
    add_figure_block(doc, fig5_path, "图 5｜", "簇间距 S 的主效应分析。",
                     "(a) n=4 时连续倒谱空间剖面随间距 S 的演化；"
                     "(b) 各簇绝对表观峰值 P_i(S) 在 S=30~50 m 出现的剧烈谷值陷阱；"
                     "(c) 归一化相对响应 \\alpha_i=P_i/P_1 对间距调制的灵敏响应。",
                     lang="zh")

    _add_heading(doc, "3.3 总跨度不是充分描述变量", level=2, lang="zh")
    _add_para(doc, "工程上常使用射孔总跨度 L_{\\mathrm{span}}=(n-1)S 作为表征压裂段规模的综合指标。然而如图 8(a–c) 所示，当总跨度严格固定为 60 m 时，不同网格配置呈现出截然不同的信号响应：对于 n=2, S=60 m，末首比 R_{\\mathrm{end}}=0.4722；对于 n=3, S=30 m，R_{\\mathrm{end}}=0.1877；对于 n=4, S=20 m，R_{\\mathrm{end}}=0.1582；而对于 n=7, S=10 m，R_{\\mathrm{end}} 骤降至 0.0716。这表明相同的物理跨度下，界面数量的增加呈几何级数加剧了透射损失与多次反射。总跨度不能作为反演的充分变量，界面拓扑数 n 与间距 S 必须被独立保留。", lang="zh")

    # In-line Figure 8
    fig8_path = base_dir / "图表数据/Figure_8_Interaction_Phase_Maps.png"
    add_figure_block(doc, fig8_path, "图 8｜", "间距、簇数与深度的二维交互相图矩阵。",
                     "(a) P_1(S,n) 首峰交互相图；(b) R_end(S,n) 末首比相图；(c) 按簇数分组的 R_end-L_span 跨度关系，展示等跨度下曲线的显著分异；"
                     "(d) P_1(S,X_1) 相图；(e) \\alpha_2(S,X_1) 相图；(f) 不同 X_1 下 \\alpha_2(S) 曲线。",
                     lang="zh")

    _add_heading(doc, "3.4 深度不是可分离的标量增益与空间经验拟合", level=2, lang="zh")
    _add_para(doc, "联合 X_1 与 S 的交互分析（图 7 与图 8(d–f)）揭示：井筒沿程摩阻衰减与裂缝界面的能量吸收存在强烈的非线性交叉耦合。在 X_1=3000 m 时，固定几何指数模型拟合平均 R²=0.9864，但衰减系数 \\gamma 随间距急剧变化（例如 n=8 时，S=30 m 的 \\gamma=0.684，而 S=80 m 时降至 0.294）。图 9 证实高拟合优度仅说明特定固定构型内峰列衰减具有几何自相似性，不能外推为全工况普适物理定律。", lang="zh")

    # In-line Figure 7
    fig7_path = base_dir / "../5.3_首缝深度X1主效应分析/figures/Figure_5_3_Depth_Main_Effect.png"
    add_figure_block(doc, fig7_path, "图 7｜", "首簇深度 X1 的主效应分析。",
                     "对比首簇深度从 2000 m 增加至 4500 m 时绝对倒谱峰与相对幅值的变化，证实沿程管流摩阻对波形幅值的全局压制。",
                     lang="zh")

    # In-line Figure 9
    fig9_path = base_dir / "../5.6_空间衰减包络与模型拟合/figures/Figure_5_6_Spatial_vs_Topological_Decay.png"
    add_figure_block(doc, fig9_path, "图 9｜", "固定几何内的经验空间衰减包络拟合。",
                     "展示固定拓扑下 P_i=A\\exp[-\\gamma(i-1)] 的描述性拟合效果及参数演化，阐明 \\gamma 仅为工况内经验参数。",
                     lang="zh")

    # Section 4
    _add_heading(doc, "4 物理信息重建", level=1, lang="zh")
    _add_para(doc, "前向证伪表明，直接根据实测峰进行裂缝状态解释必然产生严重误诊。图 3 给出了本文提出的四算子非循环物理信息重建框架整体流程，旨在消除可预测的多路径混响伪峰并补偿前向传输损耗，同时坚守严格的“零保持性”。", lang="zh")

    # In-line Figure 3
    fig3_path = base_dir / "../Section3_可信度指数模型与方法论/figures/Figure_3_3_Physics_Informed_Dealiasing_Workflow.png"
    add_figure_block(doc, fig3_path, "图 3｜", "物理信息去混叠四算子重建框架体系图。",
                     "(a) 原始连续倒谱剖面及名义射孔先验与非射孔多路径候选位置；"
                     "(b) 仅由前向波场定义的候选路径凹口算子 G_peel 与有限传输增益 K_comp；"
                     "(c) 有限增益补偿核与 0.80 m 超高斯聚焦算子 F_focus 候选核；"
                     "(d) 先验/传输敏感性置信度代理掩膜 M(x) 示意（绿色：高置信，黄色：歧义过渡，红色：低信息/伪峰排除带）。",
                     lang="zh")

    _add_heading(doc, "4.1 非循环反问题数学构形", level=2, lang="zh")
    _add_para(doc, "反问题的输入为井口观测剖面 y(x)=P_{2D}(x) 以及由式 (3) 给出的射孔几何先验。未知状态量定义为非负进液活动向量 z=(z_1, \\ldots, z_n)^T \\ge 0，其中 z_k=0 严格表示该簇未进液。单位响应模板 h_k(x) 与多路径伪峰模板 g_m(x) 由前向 MOC 模型离线预计算。最优化目标表述为：", lang="zh")

    # Equation 5
    eq5_tex = r"\min_{z\ge0,\,c\ge0} \left\|W^{1/2}\left[y-Hz-Gc\right]\right\|_2^2 +\lambda_z\|z\|_1+\lambda_c\|c\|_2^2"
    add_equation_block(doc, eq5_tex, "(5)")

    _add_para(doc, "式中：H=[h_1, \\ldots, h_n]，G=[g_2, \\ldots, g_M]，W 为深度样本权重矩阵，c 为伪峰幅值向量。伪峰系数受物理约束 c_m \\le \\bar\\rho_m \\sum_k z_k 限制，从而在物理上杜绝了未激活压裂段凭空由伪峰拟合出裂缝的荒谬现象。正则化参数 \\lambda_z, \\lambda_c 必须在独立的测试集上预先确定，严禁根据目标反演剖面反向微调。", lang="zh")

    _add_heading(doc, "4.2 多路径混响剥离算子 G_peel", level=2, lang="zh")
    _add_para(doc, "水平井筒离散界面间的往复波反射会在等效声程深度生成虚假多路径伪峰，其预测候选深度集合为：", lang="zh")

    # Equation 6
    eq6_tex = r"x_{\mathrm{ghost},m}=X_1+mS,\qquad m\ge2,"
    add_equation_block(doc, eq6_tex, "(6)")

    _add_para(doc, "关键准则：若 x_{\\mathrm{ghost},m} 恰好与名义射孔集合重合，则绝不从正文中凹口剥离，而是保留在式 (5) 的活动量拟合中；仅当候选深度落入非射孔空白段时，才启动凹口抑制。令 \\kappa(u)=\\exp(-u^2)，w_g 为伪峰校准半宽，\\rho_m 为路径衰减系数，剥离算子构筑为：", lang="zh")

    # Equation 7
    eq7_tex = r"G_{\mathrm{peel}}(x)=\operatorname{clip}\left[1-\sum_{m\in\mathcal G}\rho_m\kappa\left(\frac{x-x_{\mathrm{ghost},m}}{w_g}\right), G_{\min},1\right]."
    add_equation_block(doc, eq7_tex, "(7)")

    _add_heading(doc, "4.3 传输与多界面相位补偿算子 K_comp", level=2, lang="zh")
    _add_para(doc, "为修正前向传播与穿透损耗，第 k 个射孔簇的有限物理补偿增益定义为：", lang="zh")

    # Equation 8
    eq8_tex = r"g_k(S)=\frac{1}{T_{\mathrm{eff}}^{2(k-1)}\eta(S)},\qquad K_{\mathrm{comp}}(x)=1+\sum_{k=1}^{n}[g_k(S)-1]\omega_k(x),"
    add_equation_block(doc, eq8_tex, "(8)")

    _add_para(doc, "式中：有效透射系数 T_{\\mathrm{eff}} \\in (0,1] 来自离线正演标定，相干相消因子 \\eta(S) \\in [\\eta_{\\min}, 1] 遵循连续波干涉物理模型：", lang="zh")

    # Equation 9
    eq9_tex = r"\eta(S)=\operatorname{clip}\left(\left|1+R_{\mathrm{frac}}e^{-\mathrm i4\pi S/\lambda_0}\right|,\eta_{\min},1\right)."
    add_equation_block(doc, eq9_tex, "(9)")

    _add_para(doc, "式中：R_{\\mathrm{frac}} 为复反射系数，\\lambda_0 为主频参考波长。特别强调：K_{\\mathrm{comp}} 绝不采用局部实测峰倒数强行归一化。重建活跃度为 \\tilde z_k=g_k z_k，由于 g_k 为有限增益，当且仅当 z_k=0 时必有 \\tilde z_k=0，严格捍卫了“零保持性”。", lang="zh")

    _add_heading(doc, "4.4 超高斯空间聚焦算子 F_focus", level=2, lang="zh")
    _add_para(doc, "对于反演活动量超过预设门限 \\tau_z 的活跃簇，施加特征宽度为 0.80 m 的四阶超高斯空间聚焦核，以恢复尖锐的簇界面能量边界：", lang="zh")

    # Equation 10
    eq10_tex = r"F_{\mathrm{focus}}(x)=\sum_{k:z_k>\tau_z}q_k\exp\left[-\left(\frac{x-\hat x_k}{w_{\mathrm{core}}}\right)^4\right],\qquad w_{\mathrm{core}}=0.80\,\mathrm m ."
    add_equation_block(doc, eq10_tex, "(10)")

    _add_heading(doc, "4.5 连续置信度与证据敏感度掩膜 M(x)", level=2, lang="zh")
    _add_para(doc, "连续掩膜综合量化了反演解的数学可信度与物理证据支持度：", lang="zh")

    # Equation 11
    eq11_tex = r"M(x)=\operatorname{clip}\left[\sum_k u_k\tilde z_k\,\kappa\left(\frac{x-\hat x_k}{w_t}\right)-\sum_{m\in\mathcal G}v_m\kappa\left(\frac{x-x_{\mathrm{ghost},m}}{w_g}\right),0,1\right]."
    add_equation_block(doc, eq11_tex, "(11)")

    _add_para(doc, "根据 M(x) 的取值，将其划分为三个具有明确工程指导意义的置信区间：绿色安全带 (M ≥ 0.70)，表示具有坚实的正演与信号支撑，可安全传递至下游裂缝开度反演；黄色歧义带 (0.20 ≤ M < 0.70)，表示存在界面重叠或强衰减，需结合贝叶斯后验区间审慎评估；红色危险带 (M < 0.20)，提示此处极可能为多路径伪峰或信息极匮乏区，强制排除在定量反演之外。最终重构物理剖面可紧凑表达为：", lang="zh")

    # Equation 12
    eq12_tex = r"\tilde P(x)=M(x)G_{\mathrm{peel}}(x)K_{\mathrm{comp}}(x)\left[\sum_k z_kh_k(x)\right]F_{\mathrm{focus}}(x)."
    add_equation_block(doc, eq12_tex, "(12)")

    # Section 5
    _add_heading(doc, "5 四个极端工况验证与算子规范", level=1, lang="zh")
    _add_para(doc, "为验证前向网络失真及其对反演算法的挑战，本文从 426 个网格中选取了四个具有代表性的极端几何工况，其原始倒谱、设计先验与路径深度如图 4 所示。表 2 汇总了各个工况的物理特征与可复现检验准则。", lang="zh")

    # Table 2
    t2_headers = ["工况序号与名称", "几何与参数配置", "前向物理特征与证伪证据", "非循环检验与算子规范准则"]
    t2_rows = [
        ["Case 1: 相消与伪峰陷阱", "X1=3000 m, S=30 m, n=4", "P=[3.122, 1.442, 0.536, 0.324] a.u.；在约 3119.2 m 和 3149.7 m 处观测到显著次级伪峰，与 3120/3150 m 预测深度高度契合", "采用先验模板拟合 z，仅剥离非射孔深度伪峰；必须报告 z3, z4 的支持度与残差，严禁强行补全"],
        ["Case 2: 超密集簇极限", "S=10 m, n=4", "P=[2.738, 1.421, 0.964, 0.634] a.u.；相邻簇间距 10 m 接近 30 s 分析窗的空间分辨极限，峰包络出现严重重叠", "计算并报告反演系数协方差；严禁仅凭设计位置强行分辨 4 个独立单峰，未分辨区域由黄色掩膜标出"],
        ["Case 3: 深级联耗散极限", "S=20 m, n=8", "P=[3.420, 1.616, 0.695, 0.554, 0.406, 0.274, 0.231, 0.184] a.u.，Rend 骤降至 0.0537，后序簇能量耗散殆尽", "通过独立标定施加有限增益 gk；构建未活化合成工况严格检验零保持性，并报告增益引起的不确定度膨胀"],
        ["Case 4: 高间距原始基线", "S=80 m, n=4", "P=[3.090, 1.581, 1.158, 0.878] a.u.，Rend=0.2841；波包充分分离，基本无重叠相消", "在独立标定预测较高透射率时检验 Kcomp 是否平滑趋近于 1；检验掩膜 M(x) 是否保持稳定不产生虚假畸变"],
    ]
    add_three_line_table(doc, t2_headers, t2_rows, caption_label="表 2",
                         caption_title="四个极端几何工况的前向失真特征与标准化非循环检验准则",
                         lang="zh", col_widths=[1.5, 1.4, 1.8, 1.8])

    # In-line Figure 4
    fig4_path = base_dir / "../Section3_可信度指数模型与方法论/figures/Figure_3_4_MultiCase_Physics_Correction_Matrix.png"
    add_figure_block(doc, fig4_path, "图 4｜", "四个极端工况的前向证据矩阵。",
                     "(a) S=30 m, n=4 的界面相消陷阱与 3120/3150 m 处清晰可见的多路径伪峰；"
                     "(b) S=10 m, n=4 的密集波包重叠极限；"
                     "(c) S=20 m, n=8 的深级联透射耗散极限；"
                     "(d) S=80 m, n=4 的高间距单峰基线。"
                     "色带反映敏感度代理量，酒红色标出预测的非射孔多路径候选深度。注意：该图为前向证据矩阵，绝非校正后的反演结果。",
                     lang="zh")

    # Section 6
    _add_heading(doc, "6 讨论与下游反演接口", level=1, lang="zh")
    _add_para(doc, "前向 426 网格的系统分析从根本上革新了井口水击倒谱峰的物理认知：P_i 是受整个管网状态调节的“网络条件化表观响应”，而非单簇性质的“局部独立测量”。47.5% 的首峰跌幅、n≥3 时由间距引发的非单调深谷以及等跨度曲线的分异，共同揭示了多界面网络系统的复杂动力学行为。图 10 给出的可辨识度操作包络进一步界定了可靠反演的物理边界。", lang="zh")

    # In-line Figure 10
    fig10_path = base_dir / "../5.7_可辨识度边界与可行域评估/figures/Figure_5_7_Operational_Envelope.png"
    add_figure_block(doc, fig10_path, "图 10｜", "多簇可辨识度边界与操作包络。",
                     "综合展示不同检测门限下的可识别簇数、表观信噪比以及簇间距工程可行域边界，用于指导压裂施工现场的数据质量把关。",
                     lang="zh")

    _add_para(doc, "在接口层面，下游力学反演通常试图建立 z_k 与裂缝物理开度 w_{f,k} 的映射关系：z_k = \\mathcal{H}(w_{f,k}, C_{f,k}, k_{\\mathrm{leak},k})。本文明确指出，任何宣称从单一井口压力恢复绝对裂缝开度的算法，若未考虑管网非局部反馈与多路径混响，其反演结果必然存在重大偏差。本框架输出的非负活动量 z_k 与置信掩膜 M(x)，为贝叶斯反演或确定性反演提供了经过物理净化的输入约束。", lang="zh")
    _add_para(doc, "本文研究存在若干明确边界：第一，主数据表来自确定性数值求解，未叠加复杂的现场实测随机环境噪声；第二，前向基准提取利用了已知真实邻域，不代表盲定位精度；第三，当前 426 网格未穷尽所有随机部分破裂激活模式。这些边界为后续开展现场试验与盲基准验证指明了方向。", lang="zh")

    # Section 7
    _add_heading(doc, "7 结论", level=1, lang="zh")
    conclusions = [
        "下游界面对上游响应具有显著的负反馈重构作用。在 X_1=3000 m, S=20 m 下，增加第二簇使首簇表观峰骤降 47.5%，否定了单簇标定外推的局部性假设。",
        "当簇数 n≥3 时，簇间距对后序峰幅值产生强烈的非单调物理调制。在 S=30 m 处激发了强烈的多界面相消干涉，使后序峰几近消失，而在 S=100 m 时恢复，揭示了相消干涉陷阱的存在。",
        "总压裂跨度 L_span 不是反演的充分表征变量。在相同 60 m 总跨度下，末首比随簇数增加从 0.472 暴跌至 0.072，说明多界面网络的级联损耗起决定性控制作用。",
        "提出了包含多路径剥离 G_peel、有限物理补偿 K_comp、超高斯聚焦 F_focus 和证据掩膜 M(x) 的物理信息非循环重建框架。在先验与状态解耦及有限增益保障下，严格确立了“零保持性”，防止了虚假裂缝的凭空产生。",
        "定义了四个极端几何工况的标准化检验准则。426 网格坚实支持了网络前向失真机制，确立了非循环反演规范；后续研究应进一步构建包含复相位与部分破裂真值的留出盲基准。",
    ]
    for idx, c in enumerate(conclusions, 1):
        p_c = doc.add_paragraph()
        p_c.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        p_c.paragraph_format.space_after = Pt(4)
        p_c.paragraph_format.first_line_indent = Cm(0.74)
        r_num = p_c.add_run(f"({idx}) ")
        set_font(r_num, name="Times New Roman", east_asia="宋体", size=10.5, bold=True)
        add_formatted_runs(p_c, c, default_font="宋体", east_asia="宋体", size=10.5)

    # References
    _add_heading(doc, "参考文献", level=1, lang="zh")
    refs = [
        "[1] Bakku S.K., Fehler M.C., Burns D.R. Fracture compliance estimation using borehole tube waves. *Geophysics*, 2013, 78(4): D249–D260. DOI: 10.1190/geo2012-0521.1.",
        "[2] Liang C., O'Reilly O., Dunham E.M., Moos D. Hydraulic fracture diagnostics from Krauklis-wave resonance and tube-wave reflections. *Geophysics*, 2017, 82(3): D171–D186. DOI: 10.1190/geo2016-0480.1.",
        "[3] Ghidaoui M.S., Zhao M., McInnis D.A., Axworthy D.H. A review of water hammer theory and practice. *Applied Mechanics Reviews*, 2005, 58(1): 49–76. DOI: 10.1115/1.1828050.",
        "[4] Chaudhry M.H. *Applied Hydraulic Transients*. 3rd ed. New York: Springer, 2014. DOI: 10.1007/978-1-4614-8538-4.",
        "[5] Dong X.L. et al. Research and application of hydraulic fracturing fluid entry depth detection method based on water-hammer signal. *Geoenergy Science and Engineering*, 2025, 246: 213556. DOI: 10.1016/j.geoen.2024.213556.",
        "[6] Sun S.S., He Y.M., Liu L.J., Li Y.C., Zou L.Q., Yang L. Identification of fluid-entry clusters and diagnosis of downhole events based on high-frequency water hammer pressure. *International Journal of Rock Mechanics and Mining Sciences*, 2026, 200: 106437. DOI: 10.1016/j.ijrmms.2026.106437.",
        "[7] Cheng Y.J., You J.X., Guo F.Q., Wang J.C., Li L., Zhu J.P. Development and effectiveness verification of ultra-high-frequency water-hammer wave monitoring equipment for large-scale fracturing of unconventional oil and gas. *Flow Measurement and Instrumentation*, 2026, 111: 103424. DOI: 10.1016/j.flowmeasinst.2026.103424.",
        "[8] Childers D.G., Skinner D.P., Kemerait R.C. The cepstrum: a guide to processing. *Proceedings of the IEEE*, 1977, 65(10): 1428–1443. DOI: 10.1109/PROC.1977.10747.",
        "[9] Oppenheim A.V., Schafer R.W. From frequency to quefrency: a history of the cepstrum. *IEEE Signal Processing Magazine*, 2004, 21(5): 95–106. DOI: 10.1109/MSP.2004.1328092.",
        "[10] Qiu Y. et al. Water hammer response characteristics of wellbore-fracture system: multi-dimensional analysis in time, frequency and quefrency domain. *Journal of Petroleum Science and Engineering*, 2022, 213: 110425. DOI: 10.1016/j.petrol.2022.110425.",
        "[11] Luo Y. et al. A new water hammer decay model: analyzing the interference of multiple fractures and perforations on decay rate. *SPE Journal*, 2023, 28(4): 1973–1985. DOI: 10.2118/214658-PA.",
        "[12] Hu X. et al. Evaluation of multi-fractures geometry based on water hammer signals: a new comprehensive model and field application. *Journal of Hydrology*, 2022, 612: 128240. DOI: 10.1016/j.jhydrol.2022.128240.",
        "[13] Sun S.S. et al. A novel comprehensive water hammer pressure model for fracture geometry evaluation. *SPE Journal*, 2025, 30(9): 5350–5366. DOI: 10.2118/228403-PA.",
        "[14] Liu L.J., Liu Y.Z., Wang X.G. A novel MCMC-based hydraulic fracture diagnostics approach using water hammer data. In: *57th US Rock Mechanics/Geomechanics Symposium*, 2023. DOI: 10.56952/ARMA-2023-0865.",
        "[15] Zeng B. et al. Fracture size inversion method based on water hammer signal for shale reservoir. *Frontiers in Energy Research*, 2024, 11: 1336148. DOI: 10.3389/fenrg.2023.1336148.",
        "[16] Deng S. et al. A diagnostic model for hydraulic fracture in naturally fractured reservoir utilising water-hammer signal. *Engineering Fracture Mechanics*, 2025, 325: 111347. DOI: 10.1016/j.engfracmech.2025.111347.",
        "[17] Gabry M.A., Ramadan A., Soliman M.Y. Estimating water hammer damping ratios using continuous wavelet transform for induced hydraulic fracture complexity characterization. *SPE Journal*, 2025, 30(6): 3587–3611. DOI: 10.2118/225459-PA.",
        "[18] Zhu M., Wang H. Automated water hammer analysis for fracture parameter inversion using high-frequency shut-in pressure signals during hydraulic fracturing. *Modelling*, 2026, 7(3): 87. DOI: 10.3390/modelling7030087.",
        "[19] Zielke W. Frequency-dependent friction in transient pipe flow. *Journal of Basic Engineering*, 1968, 90(1): 109–115. DOI: 10.1115/1.3605049.",
        "[20] Vardy A.E., Brown J.M.B. Transient, turbulent, smooth pipe friction. *Journal of Hydraulic Research*, 1995, 33(4): 435–456. DOI: 10.1080/00221689509498654.",
        "[21] Bergant A., Simpson A.R., Vitkovsky J. Developments in unsteady pipe flow friction modelling. *Journal of Hydraulic Research*, 2001, 39(3): 249–257. DOI: 10.1080/00221680109499828.",
        "[22] Dong X.L., Wang X.M., Zhu H.Y., Yang Y.Y., He L., Liu Z.P., Gong W. The influence of filtering methods and parameters on reflection period from pump shut-in water hammer signals: a comprehensive study. *Geoenergy Science and Engineering*, 2026, 257: 214278. DOI: 10.1016/j.geoen.2025.214278.",
    ]
    for r in refs:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.line_spacing = 1.15
        p_r.paragraph_format.space_after = Pt(3)
        p_r.paragraph_format.left_indent = Cm(0.74)
        p_r.paragraph_format.first_line_indent = Cm(-0.74)
        parts = re.split(r"(\*[^*]+\*)", r)
        for pt in parts:
            if not pt:
                continue
            is_it = pt.startswith("*") and pt.endswith("*")
            run_txt = pt[1:-1] if is_it else pt
            run = p_r.add_run(run_txt)
            set_font(run, name="Times New Roman", east_asia="宋体", size=9.0, italic=is_it)

    doc.save(str(out_path))
    print(f"Successfully created: {out_path.name}")


# ---------------------------------------------------------------------------
# English Submission Manuscript Builder (SPE Journal)
# ---------------------------------------------------------------------------
def build_english_docx(base_dir: Path, out_path: Path):
    print(f"Building English submission manuscript: {out_path.name}...")
    doc = Document()
    setup_document_page_layout(doc)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(8)
    r_title = p_title.add_run("Nonlocal Water-Hammer Cepstral Responses in Multi-Cluster Wellbores: Mechanism Falsification and a Physics-Informed, Non-Circular Reconstruction Framework")
    set_font(r_title, name="Times New Roman", size=17, bold=True)

    # Authors Placeholder
    p_author = doc.add_paragraph()
    p_author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_author.paragraph_format.space_after = Pt(3)
    r_author = p_author.add_run("Author One 1,2*, Author Two 1, and Corresponding Author 2")
    set_font(r_author, name="Times New Roman", size=11, bold=True)

    # Affiliations Placeholder
    p_affil = doc.add_paragraph()
    p_affil.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_affil.paragraph_format.space_after = Pt(12)
    r_affil = p_affil.add_run("1. State Key Laboratory of Oil and Gas Reservoir Geology and Exploitation, Chengdu 610500, China\n2. College of Petroleum Engineering, Beijing 100083, China\n* Corresponding author email: corresponding_author@domain.edu")
    set_font(r_affil, name="Times New Roman", size=9.5, italic=True)

    # Abstract Box / Paragraph
    p_abs_title = doc.add_paragraph()
    p_abs_title.paragraph_format.space_before = Pt(6)
    p_abs_title.paragraph_format.space_after = Pt(2)
    r_abs_t = p_abs_title.add_run("Abstract")
    set_font(r_abs_t, name="Times New Roman", size=12, bold=True)

    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    p_abs.paragraph_format.space_after = Pt(6)
    p_abs.paragraph_format.first_line_indent = Inches(0.5)
    abs_text = (
        "Wellhead water-hammer pressure measurements offer an attractive, non-intrusive diagnostic for evaluating fluid-taking "
        "perforation clusters in stimulated horizontal wells without requiring expensive downhole sensor arrays. However, physical "
        "interpretation becomes fundamentally nonlocal when multiple hydraulic fractures intersect the same wellbore conduit: "
        "downstream interfaces strongly modify upstream peak amplitudes, intermediate cluster spacings induce profound multi-interface "
        "destructive interference that attenuates subsequent peaks far more severely than either smaller or larger spacings, and internal "
        "wave reverberations generate pronounced ghost peaks at depths where no perforations exist. In this study, we rigorously quantify "
        "these wave-network mechanisms using one-dimensional method-of-characteristics (MOC) elastodynamic simulations and a two-dimensional "
        "time--depth marginal cepstrum over a certified database of 426 independent topological grids. For a nominal first-cluster depth "
        "X_1=3000 m and spacing S=20 m, introducing a second downstream cluster reduces the apparent first-cluster peak from 5.157 to "
        "2.705 a.u.—a 47.5% collapse that directly falsifies the conventional unconditioned single-cluster calibration proxy. In four-cluster "
        "configurations (n=4), the third and fourth peaks reach sharp destructive minima of 0.536 and 0.324 a.u. at S=30 m before recovering "
        "to 1.21 and 0.92 a.u. at S=100 m. At a fixed total span of 60 m, the terminal-to-first ratio R_end drops precipitously from 0.472 "
        "to 0.072 as cluster count increases from 2 to 7, proving that total span is an insufficient descriptive scalar. Furthermore, "
        "continuous cepstral profiles exhibit secondary peaks near 3120 and 3150 m that precisely match predicted multipath travel-time ghosts.\n"
        "To resolve these distortions without circular reasoning, we formulate a four-operator physics-informed inverse reconstruction framework: "
        "nominal perforation depths serve as the known engineering prior, whereas true fluid-taking activation is treated as an unknown non-negative state. "
        "The framework couples a multipath peeling operator G_peel acting strictly on non-perforation ghost depths, a transmission and phase "
        "compensation operator K_comp bounded by finite forward physical gains [T_eff^{2(k-1)}\\eta(S)]^{-1}, a 0.80-m super-Gaussian spatial "
        "focusing operator F_focus, and a continuous evidence uncertainty mask M(x). The combination of non-negative activity constraints and finite "
        "gains guarantees exact zero-preservation: unactivated clusters cannot be artificially created by division. Synthetic verification across four "
        "extreme geometries confirms the forward distortions and demarcates the valid operational envelope for downstream fracture opening inversion."
    )
    add_formatted_runs(p_abs, abs_text, default_font="Times New Roman", size=10.5)

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_after = Pt(12)
    r_kwt = p_kw.add_run("Keywords: ")
    set_font(r_kwt, name="Times New Roman", size=10.5, bold=True)
    r_kw = p_kw.add_run("water hammer; multi-cluster fracturing; method of characteristics; cepstrum; multipath reverberation; physics-informed inverse problem; uncertainty mask; zero-preservation")
    set_font(r_kw, name="Times New Roman", size=10.5)

    # Nomenclature Table
    p_nom_h = doc.add_paragraph()
    p_nom_h.paragraph_format.space_before = Pt(8)
    p_nom_h.paragraph_format.space_after = Pt(4)
    r_nh = p_nom_h.add_run("Nomenclature")
    set_font(r_nh, name="Times New Roman", size=12, bold=True)

    nom_rows = [[s, d, u] for s, d, u in NOMENCLATURE_EN]
    add_three_line_table(doc, ["Symbol", "Physical Description and Definition", "SI Units"], nom_rows,
                         caption_label="Table 0", caption_title="Definitions of primary physical variables and operators",
                         lang="en", col_widths=[1.5, 4.0, 1.0])

    # Section 1
    _add_heading(doc, "1. Introduction", level=1, lang="en")
    _add_para(doc, "The stimulation efficacy of multi-cluster horizontal well fracturing stages fundamentally depends on which engineered perforation clusters successfully initiate and take fluid. While downhole microseismic mapping and distributed acoustic/temperature sensing (DAS/DTS) provide valuable diagnostics, borehole tube-wave investigations demonstrate that hydraulic fracture compliance, Krauklis-wave resonance, and interface reflections can also be constrained from wellbore transient dynamics [1,2]. Given the operational overhead and survivability constraints of downhole instruments, non-intrusive surface diagnostic methods are highly desirable. Shut-in and pump-off water-hammer pressure transients propagate along the casing, continuously sampling downhole acoustic impedance variations, and can be transformed into delay or depth profiles [3,4]. Recent studies have rapidly extended this principle to fluid-entry depth detection, cluster-level event diagnosis, and ultra-high-frequency monitoring equipment validation [5–7].", lang="en")

    _add_para(doc, "Conventional interpretation pipelines typically adhere to homomorphic cepstrum principles and localized reflection assumptions [8,9]: a discrete quefrency or travel-time delay is mapped to a perforation cluster, and the amplitude of the resulting cepstral peak is interpreted as a direct scalar proxy for that cluster's fluid acceptance. However, this localized assumption is valid only under negligible wellbore--fracture coupling. In multi-cluster horizontal wells, the measured transient is the output of a connected acoustic transmission-line network. Transmission and reflection at an upstream interface alter the incident wavefield experienced by every downstream cluster, while reflected waves reverberating between interfaces superimpose back onto the wellhead signal. Consequently, equal total stimulation spans with different cluster counts exhibit radically different terminal responses, and variations in cluster spacing alter waveform amplitudes rather than merely shifting arrival times [10–13]. Figure 1 illustrates this fundamental discrepancy between the naive local-label paradigm and the true wave-network observation.", lang="en")

    # In-line Figure 1
    fig1_path = base_dir / "图表数据/Figure_1_Local_vs_Network.png"
    add_figure_block(doc, fig1_path, "Figure 1.", "Local label vs. connected wave-network observation.",
                     "Left: the naive local-proxy paradigm rejected by forward elastodynamic evidence (assuming each cepstral peak independently mirrors local fracture absorption). "
                     "Right: physical reality established in this study—the surface pressure transient is a network-conditioned response shaped by multiple discrete impedance interfaces, "
                     "downstream-to-upstream reverberations, and 2D cepstral projection operators.",
                     lang="en")

    _add_para(doc, "This study addresses two interrelated physical questions. First, which conventional assumptions regarding locality, cluster spacing, and total stage span are definitively falsified by a controlled elastodynamic method-of-characteristics (MOC) grid? Second, given an engineered perforation-gun design, how can one remove predictable internal multipath reverberations without presupposing which clusters actually fractured? The second inquiry is formulated as a constrained, physics-informed inverse problem: nominal perforation depths are known engineering priors, whereas actual fluid-taking activity is an unknown non-negative state. Forcing apparent peaks onto all nominal depths would answer the wrong question and constitute circular reasoning.", lang="en")

    _add_para(doc, "Our investigation utilizes the certified simulation database (01_几何网格/峰值表/decay_table.csv) and accompanying topological analyses in Sections 5.1--5.7. The independent physical unit is an authenticated topological MOC grid rather than an arbitrary time sample or cepstral bin. Because the benchmark represents a deterministic design-space exploration without stochastic field noise or empirical replicates, p-values are not reported. The contribution is strictly bounded as a set of falsifiable forward wave-network observations and a non-circular operator specification for future blind benchmarking [14–18].", lang="en")

    # Section 2
    _add_heading(doc, "2. Forward Model and Observation Operator", level=1, lang="en")
    _add_heading(doc, "2.1 Governing equations and lumped boundary conditions", level=2, lang="en")
    _add_para(doc, "The horizontal wellbore is modeled as a one-dimensional, constant-area, weakly compressible fluid conduit. The transient wavefield is described by the hydraulic piezometric head H(x,t) and volumetric flow rate Q(x,t). Conservation of mass and linear momentum yield:", lang="en")

    # Equation 1
    eq1_tex = r"\frac{\partial H}{\partial t}+\frac{a^2}{gA}\frac{\partial Q}{\partial x}=0, \qquad \frac{\partial Q}{\partial t}+gA\frac{\partial H}{\partial x}+\frac{f}{2DA}Q|Q|=0 ."
    add_equation_block(doc, eq1_tex, "(1)")

    _add_para(doc, "where a is the acoustic wave speed, A=\\pi D^2/4 is the cross-sectional area, D is internal casing diameter, g is gravitational acceleration, and f is the Darcy--Weisbach friction factor [19–21]. The primary benchmark adopts steady friction, where f is determined from the initial Reynolds number and held fixed; unsteady Brunone friction is excluded from our primary claim boundary.", lang="en")
    _add_para(doc, "At each perforation cluster node x_k, continuity of flow coupled with lumped fracture compliance and reservoir leak-off satisfies (Figure 2(a)):", lang="en")

    # Equation 2
    eq2_tex = r"Q_{L,k}-Q_{R,k}=Q_{f,k},\qquad Q_{f,k}=C_{f,k}\frac{\mathrm d H_{f,k}}{\mathrm dt}+k_{\mathrm{leak},k}\sqrt{H_{f,k}-H_{\mathrm{ext}}} ."
    add_equation_block(doc, eq2_tex, "(2)")

    _add_para(doc, "where C_{f,k} denotes lumped fracture compliance, k_{\\mathrm{leak},k} is the leak-off coefficient, and H_{\\mathrm{ext}} is external reservoir head. In the benchmark database, all clusters share identical nominal properties to isolate geometric topology and spacing from fracture heterogeneity. MOC integrates along characteristic curves with Courant number unity. Baseline parameters are: D=0.1397 m, a=1450 m/s, fluid density 1000 kg/m³, kinematic viscosity 10⁻⁶ m²/s, roughness 4.5×10⁻⁵ m, initial velocity 1.0 m/s, initial wellhead head 300 m, \\Delta t=10⁻³ s, C_f=10⁻⁵ m², k_\\mathrm{leak}=10⁻⁴ m^{5/2}/s, and shut-in pump-off time 1.0 s.", lang="en")

    # In-line Figure 2
    fig2_path = base_dir / "图表数据/Figure_2_Forward_and_Operator.png"
    add_figure_block(doc, fig2_path, "Figure 2.", "Forward elastodynamic model and cepstral observation operator.",
                     "(a) Lumped fracture node continuity, compliance, and leak-off boundary conditions; "
                     "(b) Comparison of cumulative time--depth marginal cepstrum profiles between single-cluster baseline and four-cluster network; "
                     "(c) Nominal perforation window, apparent peak extraction radius r, and network-conditioned peak definitions. "
                     "Note: Panel (c) utilizes simulated geometric coordinates for benchmark extraction and does not constitute blind localization.",
                     lang="en")

    _add_heading(doc, "2.2 Design space and 426 independent topological grids", level=2, lang="en")
    _add_para(doc, "The perforation design space is defined by nominal shot depths:", lang="en")

    # Equation 3
    eq3_tex = r"x_{\mathrm{perf},k}=X_1+(k-1)S,\qquad k=1,\ldots,n ."
    add_equation_block(doc, eq3_tex, "(3)")

    _add_para(doc, "covering X_1 \\in \\{2000, 2500, 3000, 3500, 4000, 4500\\} m, S \\in \\{10, 20, \\ldots, 100\\} m, and n \\in \\{1, \\ldots, 8\\}. Under steady friction, multi-cluster combinations (n=2--8) yield 6×7×10=420 grids, while single-cluster baselines contribute 6 grids, providing 426 authenticated independent topologies in decay_table.csv.", lang="en")

    _add_heading(doc, "2.3 Marginal cepstrum observation and apparent peak extraction", level=2, lang="en")
    _add_para(doc, "The wellhead pressure record is transformed via a 30-s Hamming window with 5-s hop [22], producing the 2D time--depth marginal field C(x,t). Apparent peaks at cluster i are conditionally extracted via:", lang="en")

    # Equation 4
    eq4_tex = r"P_i=\max_{|x-x_i|\le r}\left[-\sum_t C(x,t)\right],\qquad r=\min(15\,\mathrm m,0.49S) ."
    add_equation_block(doc, eq4_tex, "(4)")

    _add_para(doc, "where x_i is simulated geometric depth. P_i represents an apparent network projection rather than direct fracture energy. We also define normalized ratios \\alpha_i=P_i/P_1, terminal-to-first ratio R_{\\mathrm{end}}=P_n/P_1, total span L_{\\mathrm{span}}=(n-1)S, and the descriptive empirical envelope P_i=A\\exp[-\\gamma(i-1)]. The parameter \\gamma reflects a case-specific spatial envelope rather than an intrinsic single-interface transmission coefficient.", lang="en")

    # Section 3
    _add_heading(doc, "3. Mechanism Falsification Tests", level=1, lang="en")
    _add_heading(doc, "3.1 Downstream feedback falsifies a local first-peak label", level=2, lang="en")
    _add_para(doc, "A common intuition is that the first perforation cluster, being closest to surface, is unaffected by downstream fractures. Our numerical experiments disprove this assumption. As shown in Figure 6, at X_1=3000 m and S=20 m, a single cluster (n=1) exhibits P_1=5.157 a.u. Introducing a second downstream cluster (n=2) reduces the first peak to 2.705 a.u.—a 47.545% drop despite identical local properties and shot depth. Across the multi-cluster database, first-peak values oscillate between 2.45 and 3.55 a.u., confirming that downstream reverberation actively reshapes the initial arrival.", lang="en")

    # In-line Figure 6
    fig6_path = base_dir / "../5.2_裂缝总数n主效应分析/figures/Figure_5_2_Multiplicity_Main_Effect.png"
    add_figure_block(doc, fig6_path, "Figure 6.", "Cluster multiplicity n main effect and terminal attenuation.",
                     "Evolution of apparent peaks P_i, terminal-to-first ratio R_end, and cumulative energy proxies across cluster counts n, "
                     "illustrating how cascading interfaces systematically drain wavefield energy.",
                     lang="en")

    _add_heading(doc, "3.2 Spacing changes amplitude as well as delay (destructive interference trap)", level=2, lang="en")
    _add_para(doc, "Conventional delay models assume spacing S merely shifts arrival time (Δt=2S/a). Table 1 and Figure 5 summarize peak vectors for n=4 at X_1=3000 m under steady friction. Rather than monotonic behavior, subsequent peaks exhibit a pronounced non-monotonic 'V-shaped valley': at S=30 m, the third and fourth peaks drop to sharp minima (0.536 and 0.324 a.u.), whereas at S=10 m they remain higher (0.964 and 0.634 a.u.) and recover to 1.209 and 0.920 a.u. at S=100 m. In two-cluster controls, responses remain flat (P_1≈2.70--2.74 a.u., P_2≈1.24--1.30 a.u.). This proves that in n≥3 networks, intermediate spacing triggers severe multi-interface destructive interference that nearly obliterates later true clusters.", lang="en")

    # Table 1
    t1_headers = ["Spacing S (m)", "First Peak P1 (a.u.)", "Second Peak P2 (a.u.)", "Third Peak P3 (a.u.)", "Fourth Peak P4 (a.u.)", "Ratio Rend"]
    t1_rows = [
        ["10", "2.738", "1.421", "0.964", "0.634", "0.2316"],
        ["20", "2.844", "1.317", "0.614", "0.450", "0.1582"],
        ["30", "3.122", "1.442", "0.536", "0.324", "0.1038"],
        ["40", "3.050", "1.514", "0.596", "0.327", "0.1072"],
        ["50", "3.063", "1.541", "0.609", "0.436", "0.1423"],
        ["60", "3.088", "1.464", "0.592", "0.437", "0.1415"],
        ["70", "3.040", "1.527", "1.064", "0.693", "0.2280"],
        ["80", "3.090", "1.581", "1.158", "0.878", "0.2841"],
        ["100", "3.159", "1.705", "1.209", "0.920", "0.2912"],
    ]
    add_three_line_table(doc, t1_headers, t1_rows, caption_label="Table 1",
                         caption_title="Non-monotonic apparent peak modulation by cluster spacing S (X1=3000 m, n=4, steady friction)",
                         lang="en", col_widths=[1.0, 1.1, 1.1, 1.1, 1.1, 1.1])

    # In-line Figure 5
    fig5_path = base_dir / "../5.1_间距S主效应分析/figures/Figure_5_1_Spacing_Main_Effect.png"
    add_figure_block(doc, fig5_path, "Figure 5.", "Cluster spacing S main effect analysis.",
                     "(a) Continuous cepstral profile evolution with spacing S for n=4; "
                     "(b) Absolute apparent peaks P_i(S) highlighting the destructive interference valley at S=30--50 m; "
                     "(c) Normalized peak ratios \\alpha_i=P_i/P_1 demonstrating strong spatial phase sensitivity.",
                     lang="en")

    _add_heading(doc, "3.3 Total span is not a sufficient scalar descriptor", level=2, lang="en")
    _add_para(doc, "Stage span L_{\\mathrm{span}}=(n-1)S is widely treated as a bulk proxy for fractured interval length. However, Figure 8(a--c) demonstrates that holding total span fixed at 60 m produces radically divergent behavior: for n=2 (S=60 m), R_{\\mathrm{end}}=0.4722; for n=3 (S=30 m), R_{\\mathrm{end}}=0.1877; for n=4 (S=20 m), R_{\\mathrm{end}}=0.1582; and for n=7 (S=10 m), R_{\\mathrm{end}} plummets to 0.0716. Interface count n and spacing S must be preserved independently.", lang="en")

    # In-line Figure 8
    fig8_path = base_dir / "图表数据/Figure_8_Interaction_Phase_Maps.png"
    add_figure_block(doc, fig8_path, "Figure 8.", "2D interaction phase maps across spacing, cluster count, and depth.",
                     "(a) P_1(S,n) phase map; (b) R_end(S,n) terminal ratio map; (c) R_end vs. L_span grouped by cluster count, showing equal-span trajectory divergence; "
                     "(d) P_1(S,X_1) map; (e) \\alpha_2(S,X_1) map; (f) \\alpha_2(S) profiles across varying X_1.",
                     lang="en")

    _add_heading(doc, "3.4 Depth is not a separable scalar factor and descriptive spatial fits", level=2, lang="en")
    _add_para(doc, "Joint X_1 and S analysis (Figure 7 and Figure 8(d--f)) reveals strong coupling between casing friction and interface transmission. At X_1=3000 m, descriptive exponential fits yield mean R²=0.9864, yet \\gamma varies strongly with spacing (for n=8, \\gamma=0.684 at S=30 m vs. 0.294 at S=80 m). As shown in Figure 9, high goodness-of-fit within a fixed geometry indicates internal self-similarity rather than a universal physical decay law.", lang="en")

    # In-line Figure 7
    fig7_path = base_dir / "../5.3_首缝深度X1主效应分析/figures/Figure_5_3_Depth_Main_Effect.png"
    add_figure_block(doc, fig7_path, "Figure 7.", "First-cluster depth X1 main effect analysis.",
                     "Comparison of absolute and relative cepstral peaks as X_1 increases from 2000 to 4500 m, confirming frictional damping.",
                     lang="en")

    # In-line Figure 9
    fig9_path = base_dir / "../5.6_空间衰减包络与模型拟合/figures/Figure_5_6_Spatial_vs_Topological_Decay.png"
    add_figure_block(doc, fig9_path, "Figure 9.", "Descriptive spatial decay envelope fitting in fixed geometries.",
                     "Evaluation of empirical exponential envelope P_i=A\\exp[-\\gamma(i-1)] and parametric sensitivity.",
                     lang="en")

    # Section 4
    _add_heading(doc, "4. Physics-Informed Non-Circular Reconstruction Framework", level=1, lang="en")
    _add_para(doc, "Forward falsifications demonstrate that interpreting raw peaks induces severe misdiagnosis. Figure 3 delineates our four-operator reconstruction workflow designed to suppress predictable multipath ghosts and compensate transmission losses while preserving strict zero-preservation.", lang="en")

    # In-line Figure 3
    fig3_path = base_dir / "../Section3_可信度指数模型与方法论/figures/Figure_3_3_Physics_Informed_Dealiasing_Workflow.png"
    add_figure_block(doc, fig3_path, "Figure 3.", "Physics-informed four-operator reconstruction architecture.",
                     "(a) Raw continuous cepstrum with nominal perforation priors and multipath candidate depths; "
                     "(b) Candidate multipath notch operator G_peel and bounded transmission compensation K_comp; "
                     "(c) Compensated kernel and 0.80-m super-Gaussian focus F_focus; "
                     "(d) Continuous evidence uncertainty mask M(x) (green: high confidence, yellow: transition ambiguity, red: rejection band).",
                     lang="en")

    _add_heading(doc, "4.1 Non-circular inverse problem formulation", level=2, lang="en")
    _add_para(doc, "The inverse problem receives the observed profile y(x)=P_{2D}(x) and nominal design depths (Eq. 3). The unknown state is the non-negative fluid-entry activity vector z=(z_1, \\ldots, z_n)^T \\ge 0, where z_k=0 denotes an inactive cluster. Unit response templates h_k(x) and ghost templates g_m(x) are derived from the forward MOC solver. The optimization objective is:", lang="en")

    # Equation 5
    eq5_tex = r"\min_{z\ge0,\,c\ge0} \left\|W^{1/2}\left[y-Hz-Gc\right]\right\|_2^2 +\lambda_z\|z\|_1+\lambda_c\|c\|_2^2"
    add_equation_block(doc, eq5_tex, "(5)")

    _add_para(doc, "where H=[h_1, \\ldots, h_n], G=[g_2, \\ldots, g_M], W is a diagonal weight matrix, and c represents ghost amplitudes subject to physical upper bounds c_m \\le \\bar\\rho_m \\sum_k z_k. Regularization parameters \\lambda_z, \\lambda_c must be pre-calibrated on held-out simulations, never tuned on test profiles.", lang="en")

    _add_heading(doc, "4.2 Multipath reverberation peeling operator G_peel", level=2, lang="en")
    _add_para(doc, "Internal reflections produce candidate ghost depths at acoustic reverberation intervals:", lang="en")

    # Equation 6
    eq6_tex = r"x_{\mathrm{ghost},m}=X_1+mS,\qquad m\ge2,"
    add_equation_block(doc, eq6_tex, "(6)")

    _add_para(doc, "Crucially, if x_{\\mathrm{ghost},m} coincides with an engineered perforation depth, notch filtering is prohibited and the energy is resolved via Eq. (5). Peeling is applied strictly to non-perforation depths:", lang="en")

    # Equation 7
    eq7_tex = r"G_{\mathrm{peel}}(x)=\operatorname{clip}\left[1-\sum_{m\in\mathcal G}\rho_m\kappa\left(\frac{x-x_{\mathrm{ghost},m}}{w_g}\right), G_{\min},1\right]."
    add_equation_block(doc, eq7_tex, "(7)")

    _add_heading(doc, "4.3 Transmission and multi-interface phase compensation K_comp", level=2, lang="en")
    _add_para(doc, "To correct forward cascading transmission losses, cluster-specific physical gains are defined as:", lang="en")

    # Equation 8
    eq8_tex = r"g_k(S)=\frac{1}{T_{\mathrm{eff}}^{2(k-1)}\eta(S)},\qquad K_{\mathrm{comp}}(x)=1+\sum_{k=1}^{n}[g_k(S)-1]\omega_k(x),"
    add_equation_block(doc, eq8_tex, "(8)")

    _add_para(doc, "where T_{\\mathrm{eff}} \\in (0,1] is pre-calibrated and \\eta(S) \\in [\\eta_{\\min}, 1] models destructive interference:", lang="en")

    # Equation 9
    eq9_tex = r"\eta(S)=\operatorname{clip}\left(\left|1+R_{\mathrm{frac}}e^{-\mathrm i4\pi S/\lambda_0}\right|,\eta_{\min},1\right)."
    add_equation_block(doc, eq9_tex, "(9)")

    _add_para(doc, "K_{\\mathrm{comp}} strictly avoids division by measured peaks. Reconstructed amplitudes \\tilde z_k=g_k z_k ensure that z_k=0 identically implies \\tilde z_k=0, preserving true inactivity.", lang="en")

    _add_heading(doc, "4.4 Super-Gaussian spatial focusing operator F_focus", level=2, lang="en")
    _add_para(doc, "For active clusters exceeding threshold \\tau_z, a fourth-order super-Gaussian kernel restores sharp spatial resolution:", lang="en")

    # Equation 10
    eq10_tex = r"F_{\mathrm{focus}}(x)=\sum_{k:z_k>\tau_z}q_k\exp\left[-\left(\frac{x-\hat x_k}{w_{\mathrm{core}}}\right)^4\right],\qquad w_{\mathrm{core}}=0.80\,\mathrm m ."
    add_equation_block(doc, eq10_tex, "(10)")

    _add_heading(doc, "4.5 Continuous evidence and uncertainty mask M(x)", level=2, lang="en")
    _add_para(doc, "The uncertainty mask evaluates solution confidence and physical evidence:", lang="en")

    # Equation 11
    eq11_tex = r"M(x)=\operatorname{clip}\left[\sum_k u_k\tilde z_k\,\kappa\left(\frac{x-\hat x_k}{w_t}\right)-\sum_{m\in\mathcal G}v_m\kappa\left(\frac{x-x_{\mathrm{ghost},m}}{w_g}\right),0,1\right]."
    add_equation_block(doc, eq11_tex, "(11)")

    _add_para(doc, "Three operational bands are defined: Green (M ≥ 0.70) marks high confidence suitable for fracture width inversion; Yellow (0.20 ≤ M < 0.70) signifies interface overlap or high attenuation; Red (M < 0.20) flags probable ghosts or sparse evidence. The reconstructed profile is:", lang="en")

    # Equation 12
    eq12_tex = r"\tilde P(x)=M(x)G_{\mathrm{peel}}(x)K_{\mathrm{comp}}(x)\left[\sum_k z_kh_k(x)\right]F_{\mathrm{focus}}(x)."
    add_equation_block(doc, eq12_tex, "(12)")

    # Section 5
    _add_heading(doc, "5. Synthetic Extreme Case Studies and Validation Boundary", level=1, lang="en")
    _add_para(doc, "To stress-test forward distortions and operator requirements, four extreme geometries were selected from the 426-grid benchmark (Figure 4). Table 2 specifies the diagnostic characteristics and reproducibility audit criteria.", lang="en")

    # Table 2
    t2_headers = ["Case Identifier", "Geometric Configuration", "Forward Physical Evidence", "Non-Circular Verification Protocol"]
    t2_rows = [
        ["Case 1: Destructive Ghost Trap", "X1=3000 m, S=30 m, n=4", "P=[3.122, 1.442, 0.536, 0.324] a.u.; secondary peaks observed at 3119.2 and 3149.7 m matching 3120/3150 m multipath predictions", "Fit z using nominal templates; peel only non-perforation ghosts; report z3, z4 support and residuals without target forcing"],
        ["Case 2: Ultra-Dense Cluster Limit", "S=10 m, n=4", "P=[2.738, 1.421, 0.964, 0.634] a.u.; 10-m spacing approaches the 30-s window resolution limit; overlapping wave packets", "Compute parameter covariance; avoid claiming 4 separated peaks without evidence; flag unresolved spans with yellow mask"],
        ["Case 3: Deep Cascading Dissipation", "S=20 m, n=8", "P=[3.420, 1.616, 0.695, 0.554, 0.406, 0.274, 0.231, 0.184] a.u., Rend=0.0537; severe attenuation of later clusters", "Apply pre-calibrated finite gain gk; verify zero-preservation on synthetic unactivated clusters; report uncertainty inflation"],
        ["Case 4: High-Spacing Baseline", "S=80 m, n=4", "P=[3.090, 1.581, 1.158, 0.878] a.u., Rend=0.2841; well-isolated wave packets with minimal interference", "Verify Kcomp smoothly approaches unity under high transmission; confirm mask M(x) remains stable without spurious peaks"],
    ]
    add_three_line_table(doc, t2_headers, t2_rows, caption_label="Table 2",
                         caption_title="Synthetic extreme geometries and standardized non-circular audit protocols",
                         lang="en", col_widths=[1.5, 1.4, 1.8, 1.8])

    # In-line Figure 4
    fig4_path = base_dir / "../Section3_可信度指数模型与方法论/figures/Figure_3_4_MultiCase_Physics_Correction_Matrix.png"
    add_figure_block(doc, fig4_path, "Figure 4.", "Forward evidence matrix for the four extreme geometries.",
                     "(a) S=30 m, n=4 destructive trap and prominent 3120/3150 m ghost peaks; "
                     "(b) S=10 m, n=4 dense wave-packet overlap limit; "
                     "(c) S=20 m, n=8 cascading transmission dissipation limit; "
                     "(d) S=80 m, n=4 uncoupled baseline. Note: Forward evidence matrix, not reconstructed inversion.",
                     lang="en")

    # Section 6
    _add_heading(doc, "6. Discussion and Downstream Inversion Interface", level=1, lang="en")
    _add_para(doc, "Analysis of the 426-grid benchmark establishes that wellhead water-hammer peaks are network-conditioned apparent responses rather than local cluster observables. The 47.5% first-peak drop, intermediate-spacing interference valleys, and equal-span divergence collectively demonstrate transmission-line dynamics. Figure 10 delineates the operational envelope governing identifiability.", lang="en")

    # In-line Figure 10
    fig10_path = base_dir / "../5.7_可辨识度边界与可行域评估/figures/Figure_5_7_Operational_Envelope.png"
    add_figure_block(doc, fig10_path, "Figure 10.", "Operational identifiability envelope.",
                     "Identifiable cluster counts, apparent SNR, and spacing feasibility domains across varying diagnostic thresholds.",
                     lang="en")

    _add_para(doc, "For downstream geomechanical models estimating fracture width w_{f,k} via z_k = \\mathcal{H}(w_{f,k}, C_{f,k}, k_{\\mathrm{leak},k}), our framework provides clean non-negative inputs and rigorous uncertainty masks, preventing false fractures from entering reservoir simulators.", lang="en")
    _add_para(doc, "Key study limitations include: deterministic forward simulations without field noise, benchmark peak extraction using true coordinates rather than blind localization, and unexhausted partial-breakdown combinations. These define priorities for field trials and blind benchmarks.", lang="en")

    # Section 7
    _add_heading(doc, "7. Conclusions", level=1, lang="en")
    conclusions = [
        "Downstream fracture interfaces exert strong feedback on upstream arrivals. Adding a second cluster reduces the first peak by 47.5% at X_1=3000 m, S=20 m, disproving unconditioned single-cluster calibration.",
        "In n≥3 stages, cluster spacing strongly modulates peak amplitudes through multi-interface destructive interference. At S=30 m, later peaks nearly vanish before recovering at S=100 m.",
        "Total stage span L_span is an insufficient scalar descriptor. At a fixed 60-m span, R_end drops from 0.472 to 0.072 as cluster count increases from 2 to 7.",
        "A four-operator non-circular reconstruction framework (G_peel, K_comp, F_focus, M) decouples nominal priors from unknown activation states. Bounded physical gains enforce strict zero-preservation, preventing spurious fracture creation.",
        "Standardized verification protocols on four extreme geometries confirm forward wave-network distortions; future research should evaluate blind recovery on complex-phase holdout benchmarks.",
    ]
    for idx, c in enumerate(conclusions, 1):
        p_c = doc.add_paragraph()
        p_c.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        p_c.paragraph_format.space_after = Pt(4)
        p_c.paragraph_format.first_line_indent = Inches(0.5)
        r_num = p_c.add_run(f"({idx}) ")
        set_font(r_num, name="Times New Roman", size=11, bold=True)
        add_formatted_runs(p_c, c, default_font="Times New Roman", size=11)

    # References
    _add_heading(doc, "References", level=1, lang="en")
    refs = [
        "[1] Bakku, S.K., Fehler, M.C., and Burns, D.R. 2013. Fracture compliance estimation using borehole tube waves. *Geophysics* 78 (4): D249–D260. https://doi.org/10.1190/geo2012-0521.1.",
        "[2] Liang, C., O'Reilly, O., Dunham, E.M., and Moos, D. 2017. Hydraulic fracture diagnostics from Krauklis-wave resonance and tube-wave reflections. *Geophysics* 82 (3): D171–D186. https://doi.org/10.1190/geo2016-0480.1.",
        "[3] Ghidaoui, M.S., Zhao, M., McInnis, D.A., and Axworthy, D.H. 2005. A review of water hammer theory and practice. *Applied Mechanics Reviews* 58 (1): 49–76. https://doi.org/10.1115/1.1828050.",
        "[4] Chaudhry, M.H. 2014. *Applied Hydraulic Transients*. 3rd ed. New York: Springer. https://doi.org/10.1007/978-1-4614-8538-4.",
        "[5] Dong, X.L., Wang, X.M., Zhu, H.Y. et al. 2025. Research and application of hydraulic fracturing fluid entry depth detection method based on water-hammer signal. *Geoenergy Science and Engineering* 246: 213556. https://doi.org/10.1016/j.geoen.2024.213556.",
        "[6] Sun, S.S., He, Y.M., Liu, L.J. et al. 2026. Identification of fluid-entry clusters and diagnosis of downhole events based on high-frequency water hammer pressure. *International Journal of Rock Mechanics and Mining Sciences* 200: 106437. https://doi.org/10.1016/j.ijrmms.2026.106437.",
        "[7] Cheng, Y.J., You, J.X., Guo, F.Q. et al. 2026. Development and effectiveness verification of ultra-high-frequency water-hammer wave monitoring equipment for large-scale fracturing of unconventional oil and gas. *Flow Measurement and Instrumentation* 111: 103424. https://doi.org/10.1016/j.flowmeasinst.2026.103424.",
        "[8] Childers, D.G., Skinner, D.P., and Kemerait, R.C. 1977. The cepstrum: a guide to processing. *Proceedings of the IEEE* 65 (10): 1428–1443. https://doi.org/10.1109/PROC.1977.10747.",
        "[9] Oppenheim, A.V. and Schafer, R.W. 2004. From frequency to quefrency: a history of the cepstrum. *IEEE Signal Processing Magazine* 21 (5): 95–106. https://doi.org/10.1109/MSP.2004.1328092.",
        "[10] Qiu, Y., He, Y., Liu, L. et al. 2022. Water hammer response characteristics of wellbore-fracture system: multi-dimensional analysis in time, frequency and quefrency domain. *Journal of Petroleum Science and Engineering* 213: 110425. https://doi.org/10.1016/j.petrol.2022.110425.",
        "[11] Luo, Y., He, Y., Sun, S. et al. 2023. A new water hammer decay model: analyzing the interference of multiple fractures and perforations on decay rate. *SPE Journal* 28 (4): 1973–1985. https://doi.org/10.2118/214658-PA.",
        "[12] Hu, X., He, Y., Sun, S. et al. 2022. Evaluation of multi-fractures geometry based on water hammer signals: a new comprehensive model and field application. *Journal of Hydrology* 612: 128240. https://doi.org/10.1016/j.jhydrol.2022.128240.",
        "[13] Sun, S.S., He, Y.M., Liu, L.J. et al. 2025. A novel comprehensive water hammer pressure model for fracture geometry evaluation. *SPE Journal* 30 (9): 5350–5366. https://doi.org/10.2118/228403-PA.",
        "[14] Liu, L.J., Liu, Y.Z., and Wang, X.G. 2023. A novel MCMC-based hydraulic fracture diagnostics approach using water hammer data. In *57th US Rock Mechanics/Geomechanics Symposium*, Atlanta, Georgia, 25–28 June. https://doi.org/10.56952/ARMA-2023-0865.",
        "[15] Zeng, B., Ding, Y., Chen, W. et al. 2024. Fracture size inversion method based on water hammer signal for shale reservoir. *Frontiers in Energy Research* 11: 1336148. https://doi.org/10.3389/fenrg.2023.1336148.",
        "[16] Deng, S., Liu, L., and He, Y. 2025. A diagnostic model for hydraulic fracture in naturally fractured reservoir utilising water-hammer signal. *Engineering Fracture Mechanics* 325: 111347. https://doi.org/10.1016/j.engfracmech.2025.111347.",
        "[17] Gabry, M.A., Ramadan, A., and Soliman, M.Y. 2025. Estimating water hammer damping ratios using continuous wavelet transform for induced hydraulic fracture complexity characterization. *SPE Journal* 30 (6): 3587–3611. https://doi.org/10.2118/225459-PA.",
        "[18] Zhu, M. and Wang, H. 2026. Automated water hammer analysis for fracture parameter inversion using high-frequency shut-in pressure signals during hydraulic fracturing. *Modelling* 7 (3): 87. https://doi.org/10.3390/modelling7030087.",
        "[19] Zielke, W. 1968. Frequency-dependent friction in transient pipe flow. *Journal of Basic Engineering* 90 (1): 109–115. https://doi.org/10.1115/1.3605049.",
        "[20] Vardy, A.E. and Brown, J.M.B. 1995. Transient, turbulent, smooth pipe friction. *Journal of Hydraulic Research* 33 (4): 435–456. https://doi.org/10.1080/00221689509498654.",
        "[21] Bergant, A., Simpson, A.R., and Vitkovsky, J. 2001. Developments in unsteady pipe flow friction modelling. *Journal of Hydraulic Research* 39 (3): 249–257. https://doi.org/10.1080/00221680109499828.",
        "[22] Dong, X.L., Wang, X.M., Zhu, H.Y. et al. 2026. The influence of filtering methods and parameters on reflection period from pump shut-in water hammer signals: a comprehensive study. *Geoenergy Science and Engineering* 257: 214278. https://doi.org/10.1016/j.geoen.2025.214278.",
    ]
    for r in refs:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.line_spacing = 1.15
        p_r.paragraph_format.space_after = Pt(3)
        p_r.paragraph_format.left_indent = Inches(0.3)
        p_r.paragraph_format.first_line_indent = Inches(-0.3)
        parts = re.split(r"(\*[^*]+\*)", r)
        for pt in parts:
            if not pt:
                continue
            is_it = pt.startswith("*") and pt.endswith("*")
            run_txt = pt[1:-1] if is_it else pt
            run = p_r.add_run(run_txt)
            set_font(run, name="Times New Roman", size=9.5, italic=is_it)

    doc.save(str(out_path))
    print(f"Successfully created: {out_path.name}")


# ---------------------------------------------------------------------------
# Heading & Paragraph Helpers
# ---------------------------------------------------------------------------
def _add_heading(doc, text: str, level=1, lang="zh"):
    p = doc.add_paragraph()
    if level == 1:
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        set_font(r, name="Times New Roman" if lang == "en" else "黑体",
                 east_asia="黑体", size=14, bold=True)
    elif level == 2:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(text)
        set_font(r, name="Times New Roman" if lang == "en" else "黑体",
                 east_asia="黑体", size=12, bold=True)
    else:
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(text)
        set_font(r, name="Times New Roman" if lang == "en" else "黑体",
                 east_asia="黑体", size=11, bold=True)
    return p


def _add_para(doc, text: str, lang="zh"):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    p.paragraph_format.space_after = Pt(5)
    if lang == "zh":
        p.paragraph_format.first_line_indent = Cm(0.74)
        add_formatted_runs(p, text, default_font="Times New Roman", east_asia="宋体", size=10.5)
    else:
        p.paragraph_format.first_line_indent = Inches(0.5)
        add_formatted_runs(p, text, default_font="Times New Roman", east_asia="Times New Roman", size=11.5)
    return p


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    base_dir = Path(__file__).resolve().parents[1]
    zh_out = base_dir / "PaperA_SPE_Journal_中文投稿版_可编辑公式含图.docx"
    en_out = base_dir / "PaperA_SPE_Journal_English_Submission_EditableMath.docx"

    build_chinese_docx(base_dir, zh_out)
    build_english_docx(base_dir, en_out)
    print("\nAll submission DOCX files have been successfully compiled!")


if __name__ == "__main__":
    main()
