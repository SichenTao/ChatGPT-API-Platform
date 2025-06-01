# 文件：utils/md2pdf_xhtml.py

import os
import markdown
from io import BytesIO
from xhtml2pdf import pisa

def markdown_to_pdf_bytes(md_text: str) -> bytes:
    """
    将一段 Markdown 文本渲染为 PDF 的字节流 (bytes)，
    并尝试使用项目中提供的 NotoSansCJKsc-Regular.otf 来显示中文。
    如果找不到该文件，则使用 xhtml2pdf 默认字体（中文可能乱码或黑框）。

    返回：
      bytes 对象，可直接传给 st.download_button() 下载。
    """
    # 1. 把 Markdown 文本转成 HTML 片段
    html_body = markdown.markdown(md_text, extensions=['extra', 'smarty'])

    # 2. 找到项目里 NotoSansCJKsc-Regular.otf 的绝对路径
    _this_dir = os.path.dirname(__file__)
    noto_font_path = os.path.join(_this_dir, 'fonts', 'NotoSansCJKsc-Regular.otf')

    # 3. 拼一个完整 HTML 文档，内嵌 CSS，指定 @font-face
    if os.path.isfile(noto_font_path):
        # 在 HTML <head> 中插入 CSS，将 font-family 指向我们的字体文件
        font_css = f"""
        <style>
          @font-face {{
            font-family: "NotoSansCJKsc";
            src: url("file://{noto_font_path}");
          }}
          body {{
            font-family: "NotoSansCJKsc", serif;
            line-height: 1.6;
            font-size: 12pt;
            margin: 0;
            padding: 10pt 20pt;
          }}
          h1 {{ font-size: 18pt; margin-bottom: 6pt; margin-top: 12pt; }}
          h2 {{ font-size: 16pt; margin-bottom: 6pt; margin-top: 10pt; }}
          h3 {{ font-size: 14pt; margin-bottom: 6pt; margin-top: 8pt; }}
          p  {{ margin-bottom: 6pt; }}
          ul, ol {{ margin-bottom: 6pt; margin-left: 20px; }}
          li {{ margin-bottom: 4pt; }}
          code {{
            font-family: monospace;
            background-color: #f0f0f0;
            padding: 2px 4px;
            border-radius: 4px;
          }}
          pre {{
            background-color: #f7f7f7;
            padding: 8px;
            overflow-x: auto;
            font-size: 10pt;
          }}
        </style>
        """
    else:
        # 如果没有字体文件，就去掉 @font-face 部分
        font_css = """
        <style>
          body {
            font-family: serif;
            line-height: 1.6;
            font-size: 12pt;
            margin: 0;
            padding: 10pt 20pt;
          }
          h1 { font-size: 18pt; margin-bottom: 6pt; margin-top: 12pt; }
          h2 { font-size: 16pt; margin-bottom: 6pt; margin-top: 10pt; }
          h3 { font-size: 14pt; margin-bottom: 6pt; margin-top: 8pt; }
          p  { margin-bottom: 6pt; }
          ul, ol { margin-bottom: 6pt; margin-left: 20px; }
          li { margin-bottom: 4pt; }
          code {
            font-family: monospace;
            background-color: #f0f0f0;
            padding: 2px 4px;
            border-radius: 4px;
          }
          pre {
            background-color: #f7f7f7;
            padding: 8px;
            overflow-x: auto;
            font-size: 10pt;
          }
        </style>
        """

    full_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <title>Markdown to PDF</title>
      {font_css}
    </head>
    <body>
      {html_body}
    </body>
    </html>
    """

    # 4. 使用 xhtml2pdf 把 HTML 渲染为 PDF，输出到 BytesIO
    result = BytesIO()
    pisa_status = pisa.CreatePDF(full_html, dest=result)

    if pisa_status.err:
        # 渲染失败时抛出异常
        raise RuntimeError("xhtml2pdf 渲染 PDF 失败，请检查 HTML/CSS 是否有问题。")
    pdf_bytes = result.getvalue()
    result.close()
    return pdf_bytes
