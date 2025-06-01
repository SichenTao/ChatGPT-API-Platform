# 文件：utils/markdown_parser.py

import re
from reportlab.platypus import Paragraph, ListFlowable, ListItem, Spacer
from reportlab.lib.enums import TA_LEFT

def markdown_to_flowables(
        markdown_text: str,
        normal_style,
        h1_style,
        h2_style,
        h3_style
    ) -> list:
    """
    将 Markdown 文本转换为 ReportLab Flowables 列表。
    - "# "  一级标题 → 用 h1_style
    - "## " 二级标题 → 用 h2_style
    - "### " 三级标题 → 用 h3_style
    - "- "  无序列表  → 用 normal_style 生成 ListFlowable
    - 其余段落     → 用 normal_style，支持 **粗体** 转换
    """
    flowables = []
    lines = markdown_text.split('\n')
    i = 0

    bold_pattern = re.compile(r'\*\*(.+?)\*\*')

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            flowables.append(Spacer(1, 6))
            i += 1
            continue

        if line.startswith('# '):
            text = line[2:].strip()
            flowables.append(Paragraph(text, h1_style))
            i += 1
            continue

        if line.startswith('## '):
            text = line[3:].strip()
            flowables.append(Paragraph(text, h2_style))
            i += 1
            continue

        if line.startswith('### '):
            text = line[4:].strip()
            flowables.append(Paragraph(text, h3_style))
            i += 1
            continue

        if line.startswith('- '):
            bullet_items = []
            while i < len(lines) and lines[i].lstrip().startswith('- '):
                item_text = lines[i].lstrip()[2:].strip()
                item_html = bold_pattern.sub(r'<b>\1</b>', item_text)
                p = Paragraph(item_html, normal_style)
                bullet_items.append(ListItem(p, leftIndent=12))
                i += 1
            flowables.append(ListFlowable(bullet_items, bulletType='bullet', leftIndent=12))
            continue

        paragraph_html = bold_pattern.sub(r'<b>\1</b>', line)
        flowables.append(Paragraph(paragraph_html, normal_style))
        i += 1

    return flowables
