# 文件：utils/pdf_generator.py

import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Spacer, Paragraph, ListFlowable, ListItem
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont, TTFError
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

from .markdown_parser import markdown_to_flowables

# ----------------------------------------------------------------
# 1. 动态加载“项目自带的”中文字体 NotoSansCJKsc-Regular.otf
# ----------------------------------------------------------------
# 默认使用内置 Helvetica
FONT_NAME = 'Helvetica'

# 先尝试加载放在 utils/fonts/ 中的 NotoSansCJKsc-Regular.otf
_this_dir = os.path.dirname(__file__)
_font_path = os.path.join(_this_dir, 'fonts', 'NotoSansCJKsc-Regular.otf')

if os.path.isfile(_font_path):
    try:
        pdfmetrics.registerFont(TTFont('NotoSansCJKsc', _font_path))
        FONT_NAME = 'NotoSansCJKsc'
    except TTFError:
        # 如果注册失败，就保持 Helvetica，不再抛错
        FONT_NAME = 'Helvetica'


def generate_pdf_from_markdown(title: str, info_lines: list, markdown_content: str) -> bytes:
    """
    使用 ReportLab 根据传入的标题、个人信息行和 Markdown 内容生成 PDF 字节。

    - title: 文档标题，例如 "个人八字运势报告" 或 "两人星宿配对报告"
    - info_lines: 个人信息列表，例如 ["生成日期：2025年06月01日", "姓名：张三    性别：男    出生：2000年01月01日 03时00分"]
    - markdown_content: 完整的 Markdown 文本，由 ChatGPT 输出

    返回：
    - PDF 对应的二进制字节数组，可直接给 st.download_button 使用
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    styles = getSampleStyleSheet()
    story = []

    # ----------------------------------------------------------------
    # 2. 根据 FONT_NAME 动态定义各级样式
    # ----------------------------------------------------------------
    title_style = ParagraphStyle(
        name='Title',
        parent=styles['Title'],
        fontName=FONT_NAME,
        fontSize=18,
        leading=22,
        alignment=TA_LEFT,
        spaceAfter=12
    )
    info_style = ParagraphStyle(
        name='Info',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=11,
        leading=14,
        spaceAfter=6
    )
    h1_style = ParagraphStyle(
        name='Heading1',
        parent=styles['Heading1'],
        fontName=FONT_NAME,
        fontSize=16,
        leading=20,
        spaceAfter=10,
        alignment=TA_LEFT
    )
    h2_style = ParagraphStyle(
        name='Heading2',
        parent=styles['Heading2'],
        fontName=FONT_NAME,
        fontSize=14,
        leading=18,
        spaceAfter=8,
        alignment=TA_LEFT
    )
    h3_style = ParagraphStyle(
        name='Heading3',
        parent=styles['Heading3'],
        fontName=FONT_NAME,
        fontSize=12,
        leading=16,
        spaceAfter=6,
        alignment=TA_LEFT
    )
    normal_style = ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=11,
        leading=14,
        spaceAfter=6,
        alignment=TA_LEFT
    )

    # ----------------------------------------------------------------
    # 3. 先插入“标题”和“个人信息”部分
    # ----------------------------------------------------------------
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 6))

    for line in info_lines:
        # 用 <br/> 支持换行
        story.append(Paragraph(line.replace('\n', '<br/>'), info_style))
    story.append(Spacer(1, 12))

    # ----------------------------------------------------------------
    # 4. 把 Markdown 文本转换为 Flowables，并追加到 story
    # ----------------------------------------------------------------
    flowables = markdown_to_flowables(
        markdown_content,
        normal_style=normal_style,
        h1_style=h1_style,
        h2_style=h2_style,
        h3_style=h3_style
    )
    story.extend(flowables)

    # ----------------------------------------------------------------
    # 5. 构建 PDF 并输出字节数组
    # ----------------------------------------------------------------
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
