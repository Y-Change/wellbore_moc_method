"""
Refined Nature/SPE Publication-Grade PPTX Presentation Generator
Key Updates:
1. Slide 2: Embedded Figure_Physical_Distortions_Schematic.png (Left) + 3 Distortion Cards (Right).
2. Slide 3: Unified Single-Slide Reverse Reconstruction Principle (All 4 operators + explicit Purpose + LaTeX Formula Images).
3. Slide 4: Deep Methodological Comparison: Physics-Informed Reconstruction vs. Traditional Metric Psi(x).
4. Slide 5: Figure 3.3 Workflow Verification.
5. Slide 6: Figure 3.4 Multi-Scenario Matrix.
6. Slide 7: Quantitative Benchmark Table.
7. Slide 8: Conclusions & Academic Contributions.

Author: Antigravity (Pair Programming with User)
"""

import os
import sys
import glob
import win32com.client
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# Paths
REPO_ROOT = r"E:\water_hammer_research\wellbore_moc_method"
BASE_DIR  = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应", "文章撰写", "Section3_可信度指数模型与方法论")
FIG_DIR   = os.path.join(BASE_DIR, "figures")
FORMULA_DIR = os.path.join(FIG_DIR, "formulas")
OUTPUT_PPTX = os.path.join(BASE_DIR, "物理驱动的多簇水击倒谱去混叠与裂缝剖面重构方法_汇报.pptx")
OUTPUT_PPTX_LEGACY = os.path.join(BASE_DIR, "多簇水击倒谱可信度与辨识度评估模型体系_汇报.pptx")
ARTIFACT_DIR = r"C:\Users\Change\.gemini\antigravity\brain\15398df0-5275-4b15-b621-592a9e902217"

# Color Palette Definitions
COLOR_NAVY_DARK = RGBColor(14, 47, 68)     # #0E2F44 Main Theme / Deep Navy
COLOR_NAVY_BLUE = RGBColor(27, 79, 114)    # #1B4F72 Primary Dark Blue
COLOR_TEAL      = RGBColor(17, 120, 100)   # #117864 Highlight / True Fracture
COLOR_EMERALD   = RGBColor(22, 160, 133)   # #16A085 Secondary Accent
COLOR_ORANGE    = RGBColor(211, 84, 0)     # #D35400 Warning / Raw Signal
COLOR_CRIMSON   = RGBColor(144, 12, 63)    # #900C3F Ghost / Hazard
COLOR_SLATE     = RGBColor(44, 62, 80)     # #2C3E50 Dark Text
COLOR_MUTED     = RGBColor(127, 140, 141)  # #7F8C8D Subtitle / Caption
COLOR_BG_CARD   = RGBColor(250, 251, 252)  # #FAFBFC Clean light card background
COLOR_BORDER    = RGBColor(220, 226, 230)  # #DCE2E6 Subtle border
COLOR_WHITE     = RGBColor(255, 255, 255)
COLOR_GREEN_BG  = RGBColor(234, 250, 243)  # #EAF7F0 Light green tint
COLOR_RED_BG    = RGBColor(253, 237, 236)  # #FDEDEC Light red tint
COLOR_BLUE_BG   = RGBColor(235, 245, 251)  # #EBF5FB Light blue tint


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    def add_header(slide, title_text, category_text=""):
        top_bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.85)
        )
        top_bar.fill.solid()
        top_bar.fill.fore_color.rgb = COLOR_WHITE
        top_bar.line.color.rgb = COLOR_BORDER
        top_bar.line.width = Pt(0.75)
        
        txBox = slide.shapes.add_textbox(Inches(0.95), Inches(0.45), Inches(11.4), Inches(0.75))
        tf = txBox.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p = tf.paragraphs[0]
        if category_text:
            run_cat = p.add_run()
            run_cat.text = category_text.upper() + "  |  "
            run_cat.font.name = "Arial"
            run_cat.font.size = Pt(11)
            run_cat.font.bold = True
            run_cat.font.color.rgb = COLOR_TEAL
            
        run_title = p.add_run()
        run_title.text = title_text
        run_title.font.name = "Microsoft YaHei"
        run_title.font.size = Pt(19)
        run_title.font.bold = True
        run_title.font.color.rgb = COLOR_NAVY_DARK

    def add_card(slide, left, top, width, height, bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER):
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.0)
        return card

    # =========================================================================
    # SLIDE 1: Cover Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_NAVY_DARK
    bg1.line.fill.background()
    
    acc = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.8), Inches(0.18), Inches(3.8))
    acc.fill.solid()
    acc.fill.fore_color.rgb = COLOR_TEAL
    acc.line.fill.background()
    
    tb1 = s1.shapes.add_textbox(Inches(1.6), Inches(1.8), Inches(10.5), Inches(3.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0
    
    p_tag = tf1.paragraphs[0]
    r_tag = p_tag.add_run()
    r_tag.text = "SECTION 3 METHODOLOGY & WAVEFORM RECONSTRUCTION"
    r_tag.font.name = "Arial"
    r_tag.font.size = Pt(13)
    r_tag.font.bold = True
    r_tag.font.color.rgb = COLOR_EMERALD
    
    p_title = tf1.add_paragraph()
    p_title.space_before = Pt(14)
    r_title = p_title.add_run()
    r_title.text = "物理驱动的多簇水击倒谱\n去混叠与裂缝剖面重构方法"
    r_title.font.name = "Microsoft YaHei"
    r_title.font.size = Pt(30)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_WHITE
    
    p_sub = tf1.add_paragraph()
    p_sub.space_before = Pt(16)
    r_sub = p_sub.add_run()
    r_sub.text = "Physics-Informed Cepstral De-aliasing and True Fracture Profile Reconstruction"
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(15)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(189, 195, 199)
    
    p_meta = tf1.add_paragraph()
    p_meta.space_before = Pt(28)
    r_meta = p_meta.add_run()
    r_meta.text = "汇报主题：多簇水击倒谱去混叠与波形重构方法  |  SPE Journal 论文方法论专题"
    r_meta.font.name = "Microsoft YaHei"
    r_meta.font.size = Pt(11)
    r_meta.font.color.rgb = RGBColor(149, 165, 166)

    # =========================================================================
    # SLIDE 2: 核心挑战与物理困境 (左侧机理图 + 右侧三失真卡片)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "核心挑战：多簇水击倒谱面临的‘三大物理失真’", "Research Motivation & Physical Distortions")
    
    # Left: Embed Physical Distortion Schematic Figure
    fig_schem_path = os.path.join(FIG_DIR, "Figure_Physical_Distortions_Schematic.png")
    if os.path.exists(fig_schem_path):
        s2.shapes.add_picture(fig_schem_path, Inches(0.85), Inches(1.45), width=Inches(5.75))
        
    # Right: 3 Structured Cards
    cards_data2 = [
        {
            "num": "01",
            "title": "反相相消干涉 (Phase Cancellation Trap)",
            "badge": "深层真裂缝‘假消失’",
            "desc": "当簇间距 S 处于半波长奇数倍时，反射波与前序透射波反向叠加，深部 f3, f4 幅值严重萎缩 (<0.5 a.u.)，导致【漏诊误判】。",
            "color": COLOR_ORANGE,
            "bg": COLOR_BG_CARD
        },
        {
            "num": "02",
            "title": "多径反射谐波 (Multi-Path Ghost Echoes)",
            "badge": "无缝深部冒出‘假裂缝’",
            "desc": "声波在多道裂缝界面间来回往返反弹，在 x1 + m·S (m ≥ 2) 处冒出多次波谐波尖峰，诱发【假多缝/压窜虚警】。",
            "color": COLOR_CRIMSON,
            "bg": COLOR_RED_BG
        },
        {
            "num": "03",
            "title": "多界面级联耗竭 (Cascaded Depletion)",
            "badge": "深部能量耗竭与毛刺放大",
            "desc": "穿透多道裂缝后能量按 T_eff^{2(k-1)} 几何级数衰竭超 96%，传统暴力放大又会同步放大伴生毛刺杂波，形成【探测盲区】。",
            "color": COLOR_NAVY_BLUE,
            "bg": COLOR_BLUE_BG
        }
    ]
    
    start_y = 1.45
    card_h = 1.70
    gap_y  = 0.17
    
    for i, c in enumerate(cards_data2):
        cy = start_y + i * (card_h + gap_y)
        add_card(s2, 6.85, cy, 5.63, card_h, bg_color=c["bg"], border_color=COLOR_BORDER)
        
        txBox = s2.shapes.add_textbox(Inches(7.05), Inches(cy + 0.12), Inches(5.25), Inches(card_h - 0.24))
        tf = txBox.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p1 = tf.paragraphs[0]
        r1_num = p1.add_run()
        r1_num.text = c["num"] + "  "
        r1_num.font.name = "Arial"
        r1_num.font.size = Pt(13)
        r1_num.font.bold = True
        r1_num.font.color.rgb = c["color"]
        
        r1_title = p1.add_run()
        r1_title.text = c["title"]
        r1_title.font.name = "Microsoft YaHei"
        r1_title.font.size = Pt(11.5)
        r1_title.font.bold = True
        r1_title.font.color.rgb = COLOR_NAVY_DARK
        
        p2 = tf.add_paragraph()
        p2.space_before = Pt(2)
        r2 = p2.add_run()
        r2.text = "【核心危害】： " + c["badge"]
        r2.font.name = "Microsoft YaHei"
        r2.font.size = Pt(9.5)
        r2.font.bold = True
        r2.font.color.rgb = c["color"]
        
        p3 = tf.add_paragraph()
        p3.space_before = Pt(2)
        r3 = p3.add_run()
        r3.text = c["desc"]
        r3.font.name = "Microsoft YaHei"
        r3.font.size = Pt(9.2)
        r3.font.color.rgb = COLOR_SLATE

    # =========================================================================
    # SLIDE 3: 逆向重构原理（单页聚合：四大算子目的 + 公式图件 + 物理机制）
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "逆向重构原理：四大物理算子协同架构与数学机制", "Physics-Informed De-aliasing & Reconstruction Operators")
    
    # Master Formula Banner Card at Top
    add_card(s3, 0.85, 1.40, 11.63, 1.15, bg_color=COLOR_GREEN_BG, border_color=COLOR_TEAL)
    tb_m = s3.shapes.add_textbox(Inches(1.05), Inches(1.48), Inches(6.0), Inches(0.95))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True
    tf_m.margin_left = tf_m.margin_top = tf_m.margin_right = tf_m.margin_bottom = 0
    
    pm1 = tf_m.paragraphs[0]
    rm1 = pm1.add_run()
    rm1.text = "📐 物理级联重构总方程 (Master Waveform Equation):"
    rm1.font.name = "Microsoft YaHei"
    rm1.font.size = Pt(10.5)
    rm1.font.bold = True
    rm1.font.color.rgb = COLOR_TEAL
    
    # Embed Master Formula Image
    f_master_path = os.path.join(FORMULA_DIR, "formula_master_reconstruction.png")
    if os.path.exists(f_master_path):
        s3.shapes.add_picture(f_master_path, Inches(1.05), Inches(1.82), width=Inches(5.6))
        
    tb_m_goal = s3.shapes.add_textbox(Inches(7.0), Inches(1.48), Inches(5.3), Inches(0.95))
    tf_m_goal = tb_m_goal.text_frame
    tf_m_goal.word_wrap = True
    tf_m_goal.margin_left = tf_m_goal.margin_top = tf_m_goal.margin_right = tf_m_goal.margin_bottom = 0
    
    pmg1 = tf_m_goal.paragraphs[0]
    rmg1 = pmg1.add_run()
    rmg1.text = "🎯 总体目的与物理使命："
    rmg1.font.name = "Microsoft YaHei"
    rmg1.font.size = Pt(10.5)
    rmg1.font.bold = True
    rmg1.font.color.rgb = COLOR_NAVY_DARK
    
    pmg2 = tf_m_goal.add_paragraph()
    pmg2.space_before = Pt(3)
    rmg2 = pmg2.add_run()
    rmg2.text = "将原始畸变倒谱 P_2D(x) 映射到由正演波动规律定义的候选算子模板，并保留残差和不确定度，而不预设等高或假峰归零。"
    rmg2.font.name = "Microsoft YaHei"
    rmg2.font.size = Pt(9.5)
    rmg2.font.color.rgb = COLOR_SLATE
    
    # 4 Operators in 2x2 Grid
    ops4 = [
        {
            "tag": "算子一: 多次波谐波剥离算子 G_peel(x)",
            "purpose": "在候选路径位置施加可校准的 notch 模板",
            "img": "formula_ghost_peeling.png",
            "img_w": 4.6,
            "desc": "在理论多径时延 x1+mS (m≥2) 构造有限 notch 模板；路径候选需在留出数据上校准，不能预设完全压制。",
            "color": COLOR_CRIMSON,
            "bg": COLOR_RED_BG
        },
        {
            "tag": "算子二: 级联透射与相消增益均衡算子 K_comp(x)",
            "purpose": "恢复有限、前向校准的传输和相位增益",
            "img": "formula_gain_equalization.png",
            "img_w": 4.8,
            "desc": "依据 T_eff^{2(k-1)} 与 η_phase(S) 的独立校准提供有限增益；不把输出设为固定目标幅值。",
            "color": COLOR_NAVY_BLUE,
            "bg": COLOR_BLUE_BG
        },
        {
            "tag": "算子三: 单簇物理聚焦剥离算子 F_focus(x)",
            "purpose": "对拟合活跃分量施加单簇核心聚焦核",
            "img": "formula_peak_focusing.png",
            "img_w": 4.8,
            "desc": "利用单簇核心跨度 (w_core=0.8m) 的超高斯候选核抑制旁瓣；效果由留出残差和不确定度评估。",
            "color": COLOR_TEAL,
            "bg": COLOR_BG_CARD
        },
        {
            "tag": "配套成果: 空间真伪置信度掩模 M(x)",
            "purpose": "输出证据与不确定度驱动的三色操作带",
            "img": "formula_confidence_mask.png",
            "img_w": 4.6,
            "desc": "输出连续三色带（绿色 ≥ 0.70 高证据 / 黄色 0.20~0.70 歧义 / 红色 < 0.20 低证据或路径候选）。",
            "color": COLOR_EMERALD,
            "bg": COLOR_GREEN_BG
        }
    ]
    
    grid_coords = [
        (0.85, 2.70),
        (6.83, 2.70),
        (0.85, 4.95),
        (6.83, 4.95)
    ]
    
    for op, (gx, gy) in zip(ops4, grid_coords):
        add_card(s3, gx, gy, 5.65, 2.10, bg_color=op["bg"], border_color=COLOR_BORDER)
        
        # Header + Purpose
        tb_op_t = s3.shapes.add_textbox(Inches(gx + 0.15), Inches(gy + 0.08), Inches(5.35), Inches(0.55))
        tf_op_t = tb_op_t.text_frame
        tf_op_t.word_wrap = True
        tf_op_t.margin_left = tf_op_t.margin_top = tf_op_t.margin_right = tf_op_t.margin_bottom = 0
        
        pop1 = tf_op_t.paragraphs[0]
        rop1 = pop1.add_run()
        rop1.text = op["tag"]
        rop1.font.name = "Microsoft YaHei"
        rop1.font.size = Pt(10.5)
        rop1.font.bold = True
        rop1.font.color.rgb = op["color"]
        
        pop2 = tf_op_t.add_paragraph()
        pop2.space_before = Pt(1)
        rop2 = pop2.add_run()
        rop2.text = "🎯 【算子目的】： " + op["purpose"]
        rop2.font.name = "Microsoft YaHei"
        rop2.font.size = Pt(9.2)
        rop2.font.bold = True
        rop2.font.color.rgb = COLOR_NAVY_DARK
        
        # Formula Image
        f_path = os.path.join(FORMULA_DIR, op["img"])
        if os.path.exists(f_path):
            s3.shapes.add_picture(f_path, Inches(gx + 0.15), Inches(gy + 0.65), width=Inches(op["img_w"]))
            
        # Description
        tb_op_d = s3.shapes.add_textbox(Inches(gx + 0.15), Inches(gy + 1.35), Inches(5.35), Inches(0.70))
        tf_op_d = tb_op_d.text_frame
        tf_op_d.word_wrap = True
        tf_op_d.margin_left = tf_op_d.margin_top = tf_op_d.margin_right = tf_op_d.margin_bottom = 0
        
        pop3 = tf_op_d.paragraphs[0]
        rop3 = pop3.add_run()
        rop3.text = op["desc"]
        rop3.font.name = "Microsoft YaHei"
        rop3.font.size = Pt(8.8)
        rop3.font.color.rgb = COLOR_SLATE

    # =========================================================================
    # SLIDE 4: 方法优势对比：物理重构方法 vs. 传统综合辨识度 Ψ(x)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "方法优势对比：物理逆向重构 vs. 传统辨识度打分", "Methodological Advantages & Physical Correspondence")
    
    # Left Card: Traditional Metric Psi(x) (Width: 5.65)
    add_card(s4, 0.85, 1.45, 5.65, 4.30, bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER)
    tb_t = s4.shapes.add_textbox(Inches(1.05), Inches(1.65), Inches(5.25), Inches(3.90))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    
    pt1 = tf_t.paragraphs[0]
    rt1 = pt1.add_run()
    rt1.text = "❌ 传统辨识度指标 Ψ(x) 的局限"
    rt1.font.name = "Microsoft YaHei"
    rt1.font.size = Pt(13.5)
    rt1.font.bold = True
    rt1.font.color.rgb = COLOR_CRIMSON
    
    bullets_trad = [
        "脱离波传播动力学：本质为静态加权打分（0~1 标量），打分公式与水击波在多界面间的多径反射、透射衰减和相消干涉缺乏双向解析对应；",
        "依赖人工脑补判图：仅给出置信度分数，工程师面对相消萎缩的 f3, f4 和 3120m 假峰时，依然要在畸变倒谱图上主观猜图；",
        "无法对接定量反演：由于未纠正波传播几何衰减与干涉失真，无法直接利用幅值反演各簇进液阻抗和水力开度。"
    ]
    for b in bullets_trad:
        pb = tf_t.add_paragraph()
        pb.space_before = Pt(14)
        rb = pb.add_run()
        rb.text = "• " + b
        rb.font.name = "Microsoft YaHei"
        rb.font.size = Pt(11)
        rb.font.color.rgb = COLOR_SLATE
        
    # Right Card: Physics-Informed Waveform Reconstruction (Width: 5.65)
    add_card(s4, 6.83, 1.45, 5.65, 4.30, bg_color=COLOR_GREEN_BG, border_color=COLOR_TEAL)
    tb_p = s4.shapes.add_textbox(Inches(7.03), Inches(1.65), Inches(5.25), Inches(3.90))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True
    tf_p.margin_left = tf_p.margin_top = tf_p.margin_right = tf_p.margin_bottom = 0
    
    pp1 = tf_p.paragraphs[0]
    rp1 = pp1.add_run()
    rp1.text = "✅ 物理驱动逆向重构方法 P~_2D(x) 的核心优势"
    rp1.font.name = "Microsoft YaHei"
    rp1.font.size = Pt(13.5)
    rp1.font.bold = True
    rp1.font.color.rgb = COLOR_TEAL
    
    bullets_phy = [
        "与正演物理规律对应：算子参数由界面反射/透射和候选路径模型定义，仍需独立校准；",
        "提供算子模板和证据带：展示候选路径剥离、有限增益和聚焦核，不预设等幅恢复或 Ghost Wiped Out；",
        "为后续反演提供可审计输入：只有通过留出基准、残差和不确定度检查的区间才可进入水力参数反演。"
    ]
    for b in bullets_phy:
        pb = tf_p.add_paragraph()
        pb.space_before = Pt(14)
        rb = pb.add_run()
        rb.text = "• " + b
        rb.font.name = "Microsoft YaHei"
        rb.font.size = Pt(11)
        rb.font.color.rgb = COLOR_NAVY_DARK
        
    # Bottom Summary Banner Card
    add_card(s4, 0.85, 5.90, 11.63, 1.05, bg_color=COLOR_BLUE_BG, border_color=COLOR_NAVY_BLUE)
    tb_b = s4.shapes.add_textbox(Inches(1.05), Inches(5.98), Inches(11.2), Inches(0.85))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    tf_b.margin_left = tf_b.margin_top = tf_b.margin_right = tf_b.margin_bottom = 0
    
    pbot1 = tf_b.paragraphs[0]
    rbot_t = pbot1.add_run()
    rbot_t.text = "💡 学术与工程本质飞跃： "
    rbot_t.font.name = "Microsoft YaHei"
    rbot_t.font.size = Pt(11)
    rbot_t.font.bold = True
    rbot_t.font.color.rgb = COLOR_NAVY_BLUE
    
    rbot_d = pbot1.add_run()
    rbot_d.text = "从【后验经验评分】跨越到【物理波形逆向重构】—— 不仅判别‘信不信’，更直接把‘对的波形还原出来’！"
    rbot_d.font.name = "Microsoft YaHei"
    rbot_d.font.size = Pt(10.5)
    rbot_d.font.bold = True
    rbot_d.font.color.rgb = COLOR_NAVY_DARK

    # =========================================================================
    # SLIDE 5: 全流程重构诊断实测验证 (Figure 3.3)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "实测验证：典型相消与谐波工况下的去混叠重构流程", "Workflow Verification (Figure 3.3)")
    
    fig3_path = os.path.join(FIG_DIR, "Figure_3_3_Physics_Informed_Dealiasing_Workflow.png")
    if os.path.exists(fig3_path):
        s5.shapes.add_picture(fig3_path, Inches(0.85), Inches(1.45), width=Inches(6.0))
        
    add_card(s5, 7.15, 1.45, 5.33, 5.50, bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER)
    tb_wf = s5.shapes.add_textbox(Inches(7.35), Inches(1.65), Inches(4.95), Inches(5.10))
    tfwf = tb_wf.text_frame
    tfwf.word_wrap = True
    tfwf.margin_left = tfwf.margin_top = tfwf.margin_right = tfwf.margin_bottom = 0
    
    steps = [
        ("Track (a) 原始实测倒谱剖面", COLOR_ORANGE, "前序 f1, f2 信号正常，但深部 f3, f4 受反相相消压制几乎消失；在 3120m 与 3150m 处冒出明显的多次反射假谐波峰。"),
        ("Track (b) 正演物理算子模板", COLOR_NAVY_BLUE, "红色虚线 G_peel 和蓝色有限增益仅表示候选路径与物理增益模板，未包含已估计的 eta(S) 或活动状态。"),
        ("Track (c) 候选聚焦核", COLOR_TEAL, "展示 w_core=0.80 m 的候选核和归一化增益；不显示校正后真实剖面，也不宣称等幅恢复。"),
        ("Track (d) 敏感度代理", COLOR_EMERALD, "M_sens 是先验/传输敏感度代理，不是裂缝概率或 100% 真实性判据。")
    ]
    
    for idx, (st_t, st_c, st_d) in enumerate(steps):
        p_t = tfwf.paragraphs[0] if idx == 0 else tfwf.add_paragraph()
        if idx > 0: p_t.space_before = Pt(12)
        r_t = p_t.add_run()
        r_t.text = st_t
        r_t.font.name = "Microsoft YaHei"
        r_t.font.size = Pt(12)
        r_t.font.bold = True
        r_t.font.color.rgb = st_c
        
        p_d = tfwf.add_paragraph()
        p_d.space_before = Pt(3)
        r_d = p_d.add_run()
        r_d.text = st_d
        r_d.font.name = "Microsoft YaHei"
        r_d.font.size = Pt(10.5)
        r_d.font.color.rgb = COLOR_SLATE

    # =========================================================================
    # SLIDE 6: 四大工程极限工况全景验证 (Figure 3.4)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "普适性验证：覆盖四大工程极限工况的全景测试矩阵", "Multi-Scenario Matrix (Figure 3.4)")
    
    fig4_path = os.path.join(FIG_DIR, "Figure_3_4_MultiCase_Physics_Correction_Matrix.png")
    if os.path.exists(fig4_path):
        s6.shapes.add_picture(fig4_path, Inches(0.85), Inches(1.45), width=Inches(6.8))
        
    cases_summary = [
        ("Case 1: 相消与路径候选 (S=30m, n=4)", COLOR_ORANGE, "• 原始证据：f3, f4 响应较低；3120m/3150m 为路径候选\n• 算子要求：报告活动支持度、残差和不确定度"),
        ("Case 2: 极密混叠极限段 (S=10m, n=4)", COLOR_CRIMSON, "• 原始证据：名义窗口重叠\n• 算子要求：报告协方差并保留黄色歧义区"),
        ("Case 3: 8簇深层耗竭段 (S=20m, n=8)", COLOR_NAVY_BLUE, "• 原始证据：后序响应显著衰减\n• 算子要求：使用有限增益并报告不确定度膨胀"),
        ("Case 4: 高间距原始基线 (S=80m, n=4)", COLOR_TEAL, "• 原始证据：峰间隔较大\n• 算子要求：核对补偿不产生额外峰")
    ]
    
    c_y = 1.45
    c_h = 1.25
    c_gap = 0.17
    
    for i, (ct, cc, cd) in enumerate(cases_summary):
        cy_pos = c_y + i * (c_h + c_gap)
        add_card(s6, 7.90, cy_pos, 4.58, c_h, bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER)
        
        txBox = s6.shapes.add_textbox(Inches(8.05), Inches(cy_pos + 0.08), Inches(4.3), Inches(c_h - 0.16))
        tf = txBox.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p1 = tf.paragraphs[0]
        r1 = p1.add_run()
        r1.text = ct
        r1.font.name = "Microsoft YaHei"
        r1.font.size = Pt(11)
        r1.font.bold = True
        r1.font.color.rgb = cc
        
        p2 = tf.add_paragraph()
        p2.space_before = Pt(3)
        r2 = p2.add_run()
        r2.text = cd
        r2.font.name = "Microsoft YaHei"
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = COLOR_SLATE

    # =========================================================================
    # SLIDE 7: 修正前后量化对比与工程决策收益
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "量化对比与工程价值：报告可辨识度与不确定度", "Quantitative Benchmark & Field Impact")
    
    rows, cols = 5, 4
    t_left, t_top, t_w, t_h = Inches(0.85), Inches(1.50), Inches(11.63), Inches(4.10)
    table_shape = s7.shapes.add_table(rows, cols, t_left, t_top, t_w, t_h)
    table = table_shape.table
    
    table.columns[0].width = Inches(2.20)
    table.columns[1].width = Inches(3.10)
    table.columns[2].width = Inches(3.10)
    table.columns[3].width = Inches(3.23)
    
    table_data = [
        ["典型工况场景", "原始倒谱 P_2D(x) (修正前)", "物理修正后 P~_2D(x) (修正后)", "现场决策收益 (避免工程失误)"],
        ["Case 1: 相消与路径候选\n(S=30m, n=4)", "f3, f4 响应较低\n3120m/3150m 为路径候选", "报告活动支持度、残差和不确定度\n不预设等幅或归零", "评估深部可辨识度"],
        ["Case 2: 极密混叠\n(S=10m, n=4)", "名义窗口重叠\n单簇边界可能不可分", "报告协方差并保留歧义带\n不凭设计位置宣称四簇已分离", "标记多解性"],
        ["Case 3: 8簇深层耗竭\n(S=20m, n=8)", "多界面累积透射损耗\nf5~f8 响应较低", "使用有限增益和不确定度传播\n不宣称完全恢复", "量化信息缺失"],
        ["Case 4: 高间距原始基线\n(S=80m, n=4)", "峰间隔较大\n原始前向证据较易分离", "核对补偿接近校准基线\n不产生额外峰", "前向基线检查"]
    ]
    
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c_idx == 0 else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.name = "Microsoft YaHei"
                    if r_idx == 0:
                        r.font.size = Pt(11)
                        r.font.bold = True
                        r.font.color.rgb = COLOR_WHITE
                    else:
                        r.font.size = Pt(10)
                        r.font.color.rgb = COLOR_NAVY_DARK if c_idx == 0 else COLOR_SLATE
            
            if r_idx == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_NAVY_DARK
            elif r_idx % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_WHITE
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_BG_CARD
                
    add_card(s7, 0.85, 5.80, 11.63, 1.15, bg_color=COLOR_GREEN_BG, border_color=COLOR_TEAL)
    tb_bot = s7.shapes.add_textbox(Inches(1.05), Inches(5.90), Inches(11.2), Inches(0.95))
    tfbot = tb_bot.text_frame
    tfbot.word_wrap = True
    tfbot.margin_left = tfbot.margin_top = tfbot.margin_right = tfbot.margin_bottom = 0
    
    pbot = tfbot.paragraphs[0]
    rbot_tag = pbot.add_run()
    rbot_tag.text = "🎯 核心工程价值总结： "
    rbot_tag.font.name = "Microsoft YaHei"
    rbot_tag.font.size = Pt(11)
    rbot_tag.font.bold = True
    rbot_tag.font.color.rgb = COLOR_TEAL
    
    rbot_desc = pbot.add_run()
    rbot_desc.text = "将‘哪有裂缝’转化为带物理先验、残差和不确定度的可审计输出；只有通过留出基准的区间才进入下一步阻抗与水力开度反演。"
    rbot_desc.font.name = "Microsoft YaHei"
    rbot_desc.font.size = Pt(10.5)
    rbot_desc.font.color.rgb = COLOR_NAVY_DARK

    # =========================================================================
    # SLIDE 8: 总结与学术贡献
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "总结与学术贡献：从‘主观猜图’走向‘客观物理重构’", "Conclusions & Academic Contributions")
    
    takeaways8 = [
        {
            "num": "01",
            "title": "揭示倒谱三大失真物理形成机理",
            "bullets": [
                "摆脱了传统水击分析对单点经验肉眼判图的依赖",
                "首次将界面透射损耗、相消干涉谷与多次反射谐波时延规律抽象为严谨的物理数学基准线",
                "建立了多簇水击倒谱畸变的物理先验判据"
            ]
        },
        {
            "num": "02",
            "title": "提出四道轨物理逆向重构方法",
            "bullets": [
                "创新构建 G_peel(x) 谐波剥离算子与 K_comp(x) 级联增益补偿算子",
                "引入 F_focus(x) 单簇物理聚焦算子，消除深部伴生杂波毛刺",
                "输出去假峰、补相消、抗混叠的物理真实剖面 P~_2D(x)"
            ]
        },
        {
            "num": "03",
            "title": "提供空间置信掩模与定量反演输入",
            "bullets": [
                "输出 M(x) 绿/黄/红证据带，明确高证据、歧义和低证据区",
                "保留网络衰减、残差和不确定度，避免目标等幅化",
                "为深部水力开度反演提供经过筛选的候选输入"
            ]
        }
    ]
    
    col_w8 = 3.65
    gap8   = 0.38
    start_x8 = 0.85
    
    for i, t in enumerate(takeaways8):
        cx = start_x8 + i * (col_w8 + gap8)
        add_card(s8, cx, 1.55, col_w8, 5.35, bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER)
        
        txBox = s8.shapes.add_textbox(Inches(cx + 0.22), Inches(1.75), Inches(col_w8 - 0.44), Inches(4.95))
        tf = txBox.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p1 = tf.paragraphs[0]
        r1 = p1.add_run()
        r1.text = t["num"]
        r1.font.name = "Arial"
        r1.font.size = Pt(26)
        r1.font.bold = True
        r1.font.color.rgb = COLOR_TEAL
        
        p2 = tf.add_paragraph()
        p2.space_before = Pt(10)
        r2 = p2.add_run()
        r2.text = t["title"]
        r2.font.name = "Microsoft YaHei"
        r2.font.size = Pt(13.5)
        r2.font.bold = True
        r2.font.color.rgb = COLOR_NAVY_DARK
        
        for b_idx, bullet in enumerate(t["bullets"]):
            pb = tf.add_paragraph()
            pb.space_before = Pt(10 if b_idx == 0 else 6)
            rb = pb.add_run()
            rb.text = "• " + bullet
            rb.font.name = "Microsoft YaHei"
            rb.font.size = Pt(10)
            rb.font.color.rgb = COLOR_SLATE
        
    saved_path = OUTPUT_PPTX
    try:
        prs.save(OUTPUT_PPTX)
    except PermissionError:
        saved_path = os.path.join(BASE_DIR, "物理驱动的多簇水击倒谱去混叠与裂缝剖面重构方法_汇报_最新版.pptx")
        prs.save(saved_path)
        print(f"--> File was open in another app, saved to: {saved_path}")
        
    try:
        prs.save(OUTPUT_PPTX_LEGACY)
    except:
        pass
    print(f"--> Successfully generated presentation PPTX: {saved_path}")
    return saved_path


def export_jpg_slides(ppt_path=OUTPUT_PPTX):
    powerpoint = win32com.client.Dispatch('PowerPoint.Application')
    deck = powerpoint.Presentations.Open(os.path.abspath(ppt_path), WithWindow=False)
    out_folder = os.path.join(ARTIFACT_DIR, "ppt_slides")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    for f in glob.glob(os.path.join(out_folder, "*.*")):
        try: os.remove(f)
        except: pass
        
    deck.SaveAs(os.path.abspath(out_folder), 17) # 17 = ppSaveAsJPG
    deck.Close()
    powerpoint.Quit()
    
    files = glob.glob(os.path.join(out_folder, "*.JPG"))
    for f in files:
        base = os.path.basename(f)
        num = ''.join([c for c in base if c.isdigit()])
        dst = os.path.join(out_folder, f"Slide_{num}.jpg")
        if os.path.exists(dst) and dst != f:
            os.remove(dst)
        os.rename(f, dst)
    print("--> Exported and renamed updated JPG slides.")


if __name__ == "__main__":
    saved_path = create_presentation()
    export_jpg_slides(saved_path)
