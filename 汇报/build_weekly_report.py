from pathlib import Path
import shutil
import math

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / '汇报'
TEMPLATE = REPORT_DIR / '周报模版.pptx'
OUTPUT = REPORT_DIR / '周报_水击波多裂缝诊断_2026-07-27至08-02_v2.pptx'
ASSET_DIR = REPORT_DIR / 'generated_assets'
ASSET_DIR.mkdir(parents=True, exist_ok=True)

# Template-matched restrained palette.
INK = RGBColor(35, 49, 65)
MUTED = RGBColor(91, 105, 119)
RED = RGBColor(190, 67, 49)
RED_LIGHT = RGBColor(247, 232, 227)
BLUE = RGBColor(38, 102, 145)
BLUE_LIGHT = RGBColor(229, 239, 247)
TEAL = RGBColor(37, 128, 118)
TEAL_LIGHT = RGBColor(226, 242, 238)
GOLD = RGBColor(211, 150, 38)
GOLD_LIGHT = RGBColor(250, 242, 220)
LINE = RGBColor(208, 216, 224)
WHITE = RGBColor(255, 255, 255)


def remove_shape(shape):
    element = shape._element
    element.getparent().remove(element)


def set_shape_text(shape, text, font_size=22, color=INK, bold=False,
                   align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP,
                   font_name='Microsoft YaHei', margin=0.04):
    if not hasattr(shape, 'text_frame'):
        return
    tf = shape.text_frame
    tf.clear()
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    paragraphs = text.split('\n')
    for idx, line in enumerate(paragraphs):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_after = Pt(4)
        p.line_spacing = 1.08
        for run in p.runs:
            run.font.name = font_name
            run.font.size = Pt(font_size)
            run.font.bold = bold
            run.font.color.rgb = color


def add_text(slide, text, x, y, w, h, font_size=18, color=INK,
             bold=False, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP,
             margin=0.04, font_name='Microsoft YaHei'):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    set_shape_text(box, text, font_size, color, bold, align, valign, font_name, margin)
    return box


def add_card(slide, x, y, w, h, title, value, fill_rgb, value_color=INK):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    card.fill.solid()
    card.fill.fore_color.rgb = fill_rgb
    card.line.color.rgb = LINE
    card.line.width = Pt(1)
    add_text(slide, title, x + 0.18, y + 0.16, w - 0.36, 0.34, 12, MUTED, False)
    add_text(slide, value, x + 0.18, y + 0.53, w - 0.36, h - 0.64, 26, value_color, True)
    return card


def add_footer(slide, source_text):
    add_text(slide, source_text, 0.55, 7.00, 9.8, 0.28, 9.5, MUTED, False, margin=0)


def make_f1_chart(path):
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(8.0, 3.5), dpi=220)
    categories = ['Brunone\nsingle', 'Steady\nsingle']
    f1_10 = [0.0, 1.0]
    f1_20 = [1.0, 1.0]
    x = [0, 1]
    width = 0.31
    ax.bar([i - width/2 for i in x], f1_10, width, label='Blind F1 @ 10 m', color='#BE4331')
    ax.bar([i + width/2 for i in x], f1_20, width, label='Blind F1 @ 20 m', color='#258076')
    for i, val in enumerate(f1_10):
        ax.text(i - width/2, val + 0.045, f'{val:.1f}', ha='center', va='bottom', fontsize=11, color='#233141', fontweight='bold')
    for i, val in enumerate(f1_20):
        ax.text(i + width/2, val + 0.045, f'{val:.1f}', ha='center', va='bottom', fontsize=11, color='#233141', fontweight='bold')
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('F1 score', fontsize=11, color='#5B6977')
    ax.set_xticks(x, categories, fontsize=11)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.grid(axis='y', color='#D8E0E7', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#B8C3CD')
    ax.tick_params(axis='y', labelsize=9, colors='#5B6977')
    ax.tick_params(axis='x', colors='#233141')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.04, 0.08, 0.98, 0.92])
    fig.savefig(path, transparent=False, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def fit_contain(img, box):
    bx, by, bw, bh = box
    ratio = min(bw / img.width, bh / img.height)
    nw, nh = int(img.width * ratio), int(img.height * ratio)
    return img.resize((nw, nh), Image.Resampling.LANCZOS), (int(bx + (bw - nw)/2), int(by + (bh - nh)/2))


def make_physics_montage(path):
    collapse = ROOT / 'output/analysis/decay_regression/04_collapse_and_scaling_pidx/steady_collapse_vs_divergence_x1_4000.png'
    brunone = ROOT / 'output/analysis/brunone_spacing_effect/fig2_trend.png'
    canvas = Image.new('RGB', (1600, 980), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 34)
        font_label = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 25)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    draw.rounded_rectangle((25, 20, 775, 950), radius=22, fill='#F6F8FA', outline='#D0D8E0', width=2)
    draw.rounded_rectangle((825, 20, 1575, 950), radius=22, fill='#F6F8FA', outline='#D0D8E0', width=2)
    draw.text((60, 45), 'T02  topology signature', font=font_title, fill='#233141')
    draw.text((860, 45), 'T04  Brunone degradation', font=font_title, fill='#233141')
    if collapse.exists():
        im = Image.open(collapse).convert('RGB')
        resized, pos = fit_contain(im, (55, 125, 690, 760))
        canvas.paste(resized, pos)
    if brunone.exists():
        im = Image.open(brunone).convert('RGB')
        resized, pos = fit_contain(im, (855, 125, 690, 760))
        canvas.paste(resized, pos)
    draw.text((60, 900), 'distance axis diverges; index axis contracts', font=font_label, fill='#5B6977')
    draw.text((860, 900), 'peak shift + EST rise with k', font=font_label, fill='#5B6977')
    canvas.save(path, quality=95)


def add_bullet_block(slide, lines, x, y, w, h, font_size=16, color=INK, line_gap=8):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    for idx, item in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            label, body = item
            p.text = ''
            r1 = p.add_run(); r1.text = label; r1.font.bold = True; r1.font.color.rgb = RED; r1.font.name = 'Microsoft YaHei'; r1.font.size = Pt(font_size)
            r2 = p.add_run(); r2.text = body; r2.font.color.rgb = color; r2.font.name = 'Microsoft YaHei'; r2.font.size = Pt(font_size)
        else:
            p.text = item
            for run in p.runs:
                run.font.color.rgb = color; run.font.name = 'Microsoft YaHei'; run.font.size = Pt(font_size)
        p.space_after = Pt(line_gap)
        p.line_spacing = 1.08
    return box


def add_plan_card(slide, x, y, w, h, tag, title, body, fill_rgb, tag_rgb):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    card.fill.solid(); card.fill.fore_color.rgb = fill_rgb
    card.line.color.rgb = LINE; card.line.width = Pt(1)
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + 0.22), Inches(y + 0.20), Inches(0.68), Inches(0.36))
    pill.fill.solid(); pill.fill.fore_color.rgb = tag_rgb; pill.line.fill.background()
    set_shape_text(pill, tag, 11, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0)
    add_text(slide, title, x + 0.22, y + 0.65, w - 0.44, 0.43, 15.5, INK, True)
    add_text(slide, body, x + 0.22, y + 1.10, w - 0.44, h - 1.28, 12.2, MUTED, False)
    return card


def main():
    f1_chart = ASSET_DIR / 'weekly_f1_audit.png'
    montage = ASSET_DIR / 'weekly_physics_montage.png'
    make_f1_chart(f1_chart)
    make_physics_montage(montage)

    prs = Presentation(str(TEMPLATE))
    # Slide 1 — cover.
    s = prs.slides[0]
    set_shape_text(s.shapes[1], '水击波多裂缝诊断项目周报\n2026.07.27—08.02', 34, INK, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0.02)
    set_shape_text(s.shapes[6], '汇报人：晏程\n研究主题：证据审计与论文重组', 18, MUTED, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0.02)

    # Slide 2 — outline.
    s = prs.slides[1]
    set_shape_text(s.shapes[4], '汇 报 提 纲', 24, INK, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0)
    set_shape_text(s.shapes[5], '一、上周做了什么\n二、有什么效果与证据边界\n三、下一步怎么干', 24, INK, False, PP_ALIGN.LEFT, MSO_ANCHOR.MIDDLE, margin=0.02)

    # Slide 3 — completed work.
    s = prs.slides[2]
    set_shape_text(s.shapes[3], '一、上周工作完成情况', 26, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0.02)
    remove_shape(s.shapes[4])
    add_bullet_block(s, [
        ('评估协议：', '完成 detector / score 解耦，摘除 3 处真值泄漏。'),
        ('结果审计：', '完成 T01–T04 output 证据盘点，区分可直接使用与必须补做的结果。'),
        ('论文重组：', '将 3 份重复草稿收敛为 Paper A / Paper B 两篇 canonical 小论文。'),
        ('知识库：', '重构 4 个主题页，新增 EXP-018–025 与 2 份概念页。'),
    ], 0.65, 1.7, 12.0, 2.05, 16)
    add_card(s, 0.65, 4.25, 2.75, 1.55, '真值泄漏已摘除', '3 处', RED_LIGHT, RED)
    add_card(s, 3.65, 4.25, 2.75, 1.55, '协议 / 仓库测试', '7/7 · 28/28', BLUE_LIGHT, BLUE)
    add_card(s, 6.65, 4.25, 2.75, 1.55, '新增实验记录', 'EXP-018–025', TEAL_LIGHT, TEAL)
    add_card(s, 9.65, 4.25, 2.75, 1.55, '论文结构', '3 → 2', GOLD_LIGHT, GOLD)
    add_footer(s, '依据：EXP-20260801-001；research_ob/主题/T01–T04；EXP-018–025')

    # Slide 4 — section divider.
    s = prs.slides[3]
    set_shape_text(s.shapes[1], '二、科研进展', 24, INK, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0)
    set_shape_text(s.shapes[2], '从“有结果”转向“可审计、可投稿”的证据链', 26, INK, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0.02)

    # Slide 5 — evaluation protocol result.
    s = prs.slides[4]
    set_shape_text(s.shapes[1], '二、科研进展', 20, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0.02)
    set_shape_text(s.shapes[2], '1. 评估协议重建：把历史“成功”变成可审计结果', 18, INK, True, PP_ALIGN.LEFT, MSO_ANCHOR.MIDDLE, margin=0.02)
    set_shape_text(s.shapes[5], '三处真值泄漏被摘除后，历史匹配结果需要重新解释。这里展示固定容差下的盲协议 F1；历史 135 m 容差不与 F1 混画。', 13, MUTED, False, PP_ALIGN.LEFT, MSO_ANCHOR.TOP, margin=0.03)
    # Remove all remaining template chart placeholders and labels.
    for sh in list(s.shapes)[6:]:
        remove_shape(sh)
    s.shapes.add_picture(str(f1_chart), Inches(0.62), Inches(2.92), width=Inches(5.85), height=Inches(3.42))
    add_text(s, '注：历史 135 m 容差不是可比较的 F1 指标，未纳入柱状图。', 0.72, 6.37, 5.65, 0.34, 10.5, MUTED, False, margin=0)
    add_bullet_block(s, [
        ('协议修复：', '最小峰距、匹配容差、top_n 三处泄漏。'),
        ('测试通过：', '7/7 协议测试；仓库 28/28。'),
        ('Brunone：', '单缝 F1=0 @10 m；F1=1 @20 m。'),
        ('Steady：', 'D5 双缝 F1=1，且成功分离。'),
        ('证据边界：', '无噪声、小样本；EXP-021 正式标定。'),
    ], 7.0, 2.92, 5.55, 3.55, 13.3, line_gap=5)
    add_footer(s, '数据：output/analysis/blind_protocol_reaudit/；结论来源：EXP-20260801-001')

    # Slide 6 — evidence and paper reorganization.
    s = prs.slides[5]
    set_shape_text(s.shapes[1], '二、科研进展', 20, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0.02)
    set_shape_text(s.shapes[2], '2. 研究内容重排：四条研究线收敛为两篇小论文', 18, INK, True, PP_ALIGN.LEFT, MSO_ANCHOR.MIDDLE, margin=0.02)
    set_shape_text(s.shapes[5], 'output 审计后，把可以直接进入论文的观察与必须补实验的机制分开。', 13, MUTED, False, PP_ALIGN.LEFT, MSO_ANCHOR.TOP, margin=0.03)
    # Remove the template's empty chart area and labels; add audited result figures and paper cards.
    for sh in list(s.shapes)[6:]:
        remove_shape(sh)
    s.shapes.add_picture(str(montage), Inches(0.54), Inches(2.95), width=Inches(6.05), height=Inches(3.55))
    # Paper cards on the right.
    add_plan_card(s, 7.02, 2.85, 5.55, 1.72, 'A', 'Paper A｜T02 + T03', '拓扑轴收紧 + 首缝非局部响应\n现象已支持；统计、Cf/Kleak、相位路径待补', RED_LIGHT, RED)
    add_plan_card(s, 7.02, 4.72, 5.55, 1.72, 'B', 'Paper B｜T01 + T04', '38.36 m 条件化谱基线 + Brunone 波形退化\n30-case 已支持；盲评估与指标审计待补', BLUE_LIGHT, BLUE)
    add_text(s, '统一口径：38.36 m ≠ 时域 Rayleigh 5 m；“严格频散”暂不作为主张。', 7.02, 6.53, 5.55, 0.34, 10.8, MUTED, False, margin=0)
    add_footer(s, '图：output/analysis/decay_regression/；output/analysis/brunone_spacing_effect/')

    # Slide 7 — plan divider.
    s = prs.slides[6]
    set_shape_text(s.shapes[1], '汇 报 提 纲', 24, INK, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0)
    set_shape_text(s.shapes[2], '一、上周工作完成情况\n二、科研进展与证据边界\n三、下周工作计划', 24, INK, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0.02)
    # Make this a distinct divider by changing the heading to the active section.
    set_shape_text(s.shapes[1], '三、下周工作计划', 24, INK, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0)
    set_shape_text(s.shapes[2], '优先补齐决定论文可信度的三项实验', 26, INK, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0.02)

    # Slide 8 — plan details.
    s = prs.slides[7]
    set_shape_text(s.shapes[1], '三、下周工作计划', 22, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, margin=0.02)
    remove_shape(s.shapes[2])
    add_plan_card(s, 0.62, 1.75, 3.85, 3.45, 'P0', 'EXP-021｜T01 盲分辨率', '间距 × DR × SNR\n\n检测器不读 truth；评分器独立\n\n交付：分辨概率、F1、假峰、深度误差', RED_LIGHT, RED)
    add_plan_card(s, 4.72, 1.75, 3.85, 3.45, 'P1', 'EXP-022｜T02 统计复核', '420 case 逐组 R² / ΔR² / CI\n\n修复 Cf/Kleak 完成键\n\n统一 n=5，旧混合表不再用于 claim', BLUE_LIGHT, BLUE)
    add_plan_card(s, 8.82, 1.75, 3.85, 3.45, 'P1', 'EXP-023/024｜机制与指标', 'T03：n=1、逐缝开启、相位/路径消融\n\nT04：共同 STFT、双峰阈值、热力图重算\n\n交付：Paper A/B 证据门槛更新', TEAL_LIGHT, TEAL)
    # Three work packages share the same protocol baseline and can proceed in parallel.
    add_text(s, '统一协议后并行推进', 4.40, 5.32, 4.52, 0.30, 10.5, GOLD, True, PP_ALIGN.CENTER, margin=0)
    add_text(s, '最终交付：两篇小论文的论点—证据表、可审计图表和下一轮实验结果。', 0.72, 5.75, 11.9, 0.72, 17, INK, True, PP_ALIGN.CENTER, margin=0.02)
    add_footer(s, '计划来源：EXP-20260730-021–024；Paper A / Paper B 投稿前门槛')

    # Slide 9 — close.
    s = prs.slides[8]
    # Remove the template center logo and page placeholder, retain background.
    if len(s.shapes) > 1:
        remove_shape(s.shapes[1])
    if len(s.shapes) > 1:
        # after removal, the placeholder is last; remove any placeholder text.
        for sh in list(s.shapes):
            if getattr(sh, 'text', '').strip() == '--':
                remove_shape(sh)
    panel = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(1.05), Inches(11.65), Inches(5.35))
    panel.fill.solid(); panel.fill.fore_color.rgb = WHITE
    panel.fill.transparency = 2
    panel.line.color.rgb = WHITE
    panel.line.transparency = 100
    add_text(s, '一句话总结', 1.25, 1.4, 10.8, 0.65, 30, RED, True, PP_ALIGN.CENTER, margin=0)
    add_text(s, '上周完成了从“有结果”到“可审计、可投稿”的结构重整。', 1.35, 2.15, 10.6, 0.7, 22, INK, True, PP_ALIGN.CENTER, margin=0)
    add_bullet_block(s, [
        ('1 先纠偏：', '评估协议修复后，历史 Brunone “成功”被重新校准。'),
        ('2 再归纳：', 'T01–T04 从四条散线收敛为 Paper A / Paper B。'),
        ('3 后验证：', '下一步用盲检测、参数稳健性和路径消融决定最终论文主张。'),
    ], 1.65, 3.15, 10.0, 2.0, 18, INK, line_gap=8)
    add_text(s, '谢谢', 1.25, 5.75, 10.8, 0.5, 18, RED, True, PP_ALIGN.CENTER, margin=0)

    prs.core_properties.title = '水击波多裂缝诊断项目周报｜证据审计与论文重组'
    prs.core_properties.subject = '上周工作：评估协议重建、T01–T04证据审计、两篇论文重组、下一步实验计划'
    prs.core_properties.author = '晏程'
    prs.core_properties.comments = 'Based on research_ob EXP-018–025 and existing output results. Planned experiments are labeled as planned.'
    prs.save(str(OUTPUT))
    print(OUTPUT)


if __name__ == '__main__':
    main()
