# 文件：app.py

import streamlit as st
from openai import OpenAI
from pdfminer.high_level import extract_text
from io import BytesIO
import base64
from mimetypes import guess_type
from collections import OrderedDict
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import wave
from datetime import datetime, date, time

import streamlit.components.v1 as components  # ← 用于插入打印按钮

from utils.chatgpt_client import get_client, chat_completion

# —— 页面配置 ——
st.set_page_config(page_title="ChatGPT API 平台", layout="wide")

# —— “打印本页”按钮：放在最顶端，固定在页面右上角 —— 
print_button_html = """
<button onclick="window.print()" style="
    position: fixed;
    top: 16px;
    right: 16px;
    padding: 8px 12px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 14px;
    cursor: pointer;
    z-index: 9999;
">🖨️ 打印本页</button>
"""
# 在 Streamlit 页面里渲染这段 HTML（height 设置得很小，不会占据太多视觉区域）
components.html(print_button_html, height=40)

# —— 分类规则 ——
CATEGORY_RULES = OrderedDict([
    ("八字运势", None),
    ("多模态 / 视觉", lambda m: m.startswith("gpt-4o") or m.startswith("chatgpt-4o") or "vision" in m),
    ("推理 (O1/O3/O4)", lambda m: m.startswith(("o1", "o3", "o4"))),
    ("GPT-4 家族", lambda m: m.startswith("gpt-4") or m.startswith("chatgpt-4o")),
    ("GPT-3.5 家族", lambda m: m.startswith("gpt-3.5")),
    ("语音识别", lambda m: m.startswith("whisper") or m.startswith("gpt-4o-mini-transcribe") or m.startswith("gpt-4o-transcribe")),
    ("语音合成", lambda m: m.startswith(("tts", "audio")) or m.startswith("gpt-4o-mini-tts")),
    ("图像生成", lambda m: m.startswith(("dall", "gpt-image"))),
    ("代码模型", lambda m: m.startswith(("code", "codex"))),
    ("内容审核", lambda m: m.startswith("omni-moderation")),
    ("向量嵌入", lambda m: "embedding" in m),
    ("其他", None),
])

def is_vision_model(mid: str) -> bool:
    return mid.startswith("gpt-4o") or mid.startswith("chatgpt-4o") or "vision" in mid

MODEL_INFO = {
    "gpt-4": "上下文窗口 8K，适合复杂对话。",
    "gpt-4-32k": "上下文窗口 32K，适合大文档分析。",
    "chatgpt-4o-latest": "多模态 GPT-4o 最新版，支持更深入的中文理解与生成，推荐用于高质量分析。",
    "gpt-4o-mini-high": "轻量视觉推理，速度更快。",
    "gpt-3.5-turbo": "上下文窗口 4K，速度快，日常对话与代码生成首选。",
    "o1": "O1 推理：高效低延迟。",
    "o3": "O3 推理：大吞吐量。",
    "o4": "O4 推理：大规模并发。",
    "codex-mini-latest": "轻量化代码生成模型；仅支持 /v1/responses，SDK 不直接支持。",
    "omni-moderation-latest": "内容审核模型，精准过滤违规内容。",
    "whisper-1": "Whisper：多语种音频转文字。",
    "dall-e-3": "DALL·E 3：高质量图像生成。",
    "gpt-image-1": "旧版图像生成模型。",
    "tts-1": "文本转语音，生成自然语流。",
    "audio-2": "增强版 TTS，支持多语言与声线。",
    "text-embedding-3-small": "Embedding：小体积语义向量。",
}

# —— 侧边栏：填写 API Key ——
st.sidebar.title("配置")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="在此处粘贴你的 OpenAI API Key")
if not api_key:
    st.sidebar.error("请输入 API Key 才能继续")
    st.stop()

# 创建 ChatGPT 客户端
client = get_client(api_key)

# —— 获取可用模型列表 ——
all_models = sorted(m.id for m in client.models.list().data)

# —— 侧边栏：选择“模型类别 / 功能” ——
category = st.sidebar.selectbox("模型类别 / 功能", list(CATEGORY_RULES.keys()))
rule = CATEGORY_RULES[category]
if rule:
    models = [m for m in all_models if rule(m)]
else:
    other_rules = [r for k, r in CATEGORY_RULES.items() if k not in ("八字运势", "其他") and r]
    models = [m for m in all_models if not any(r(m) for r in other_rules)]
if not models:
    st.sidebar.warning("此类别下无可用模型，已显示全部模型")
    models = all_models

# —— 将 models 简单排序 ——
models.sort(key=lambda x: (
    0 if CATEGORY_RULES["多模态 / 视觉"](x) else
    1 if CATEGORY_RULES["推理 (O1/O3/O4)"](x) else
    2 if x.startswith(("gpt-4", "chatgpt-4o")) else
    3 if x.startswith("gpt-3.5") else
    4
))

# —— 如果不是“八字运势”分类，显示模型下拉框 ——
if category != "八字运势":
    model = st.sidebar.selectbox("模型", models)
    st.sidebar.markdown(f"**特点**：{MODEL_INFO.get(model, '暂无说明。')}")
else:
    model = None
    st.sidebar.markdown("**功能说明**：此处使用 ChatGPT 接口进行八字排盘、流年流月分析、幸运色/数字/方位推荐、桃花财运预测，以及两人星宿配对。")

# —— 初始化会话状态 ——
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_pdfs" not in st.session_state:
    st.session_state.session_pdfs = []
if "session_images" not in st.session_state:
    st.session_state.session_images = []

# 标题
st.title("💬 ChatGPT API 平台 & 八字运势")

# ====================================================
# —— “八字运势”功能分支 ——
# ====================================================
if category == "八字运势":
    st.header("🀄 八字运势 / 两人星宿配对 （基于 ChatGPT）")
    st.markdown(
        "- **个人运势查询**：输入“姓名、性别、出生公历日期与时辰”，模型会给出八字、大运、流年流月、五行分析、喜用神、幸运色数字方位、事业学业、感情桃花、健康风险、财运走势、六亲关系等，全部以 Markdown 格式输出。\n"
        "- **两人星宿配对**：输入“姓名1、性别1、出生日期与时辰1；姓名2、性别2、出生日期与时辰2”，模型会给出双方八字、配对吉凶、化解建议，全部以 Markdown 格式输出。"
    )

    # —— 当前日期，传给模型做“近期”基准 ——
    today = datetime.now().strftime("%Y年%m月%d日")

    # —— 让用户选择：单人运势 or 两人配对 ——
    mode = st.radio("请选择：", ["个人运势查询", "两人星宿配对"], index=0)

    # —— 允许在此分支选择调用的 ChatGPT 模型 ——
    astro_model = st.selectbox(
        "八字运势使用模型",
        options=["chatgpt-4o-latest", "gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="选择用于八字运势分析的 ChatGPT 模型"
    )

    # ---------- 单人运势查询 ----------
    if mode == "个人运势查询":
        st.markdown(f"**参考日期（今天）：{today}**")
        # —— 收集单人信息 ——
        col0, col1, col2 = st.columns([1, 1, 1])
        with col0:
            name = st.text_input("姓名", key="single_name", help="例如：张三")
        with col1:
            gender = st.selectbox(
                "性别",
                options=["男", "女"],
                index=0,
                key="single_gender",
                help="请选择出生性别（用于命理分析）"
            )
        with col2:
            st.write("")

        col3, col4 = st.columns(2)
        with col3:
            date_str = st.date_input(
                "出生日期",
                min_value=date(1900, 1, 1),
                max_value=date(2200, 12, 31),
                value=date(2000, 1, 1),
                key="birth_date_single",
                help="请选择公历出生日期（1900–2200 年）"
            )
        with col4:
            time_str = st.time_input(
                "出生时辰",
                value=time(0, 0),
                key="birth_time_single",
                help="请选择出生时辰（24 小时制）"
            )

        if st.button("开始排盘"):
            if not (name.strip() and gender and date_str and time_str):
                st.error("⚠️ 请完整填写：姓名、性别、出生日期与时辰")
            else:
                birth_dt = datetime.combine(date_str, time_str)
                birth_text = birth_dt.strftime("%Y年%m月%d日 %H时%M分")

                # 系统提示：要求输出 Markdown 格式
                system_prompt = (
                    "你是一位经验丰富的中文命理师，擅长八字排盘、流年流月运势分析、五行格局、喜用神、"
                    "幸运色/数字/方位推荐、事业学业、感情桃花、健康风险、财运走势、六亲关系以及化解或增益建议。"
                    "请以 **Markdown** 格式输出以下内容：\n"
                    "1. # 个人信息：确认用户姓名、性别、出生信息，用以报头。\n"
                    "2. ## 八字：按天干地支列出“年柱、月柱、日柱、时柱”，并简要说明各柱间的相生相克。\n"
                    "3. ## 大运：列出该用户从出生日开始的每十年一个大运节点，并解释各大运主要吉凶变化，至少列出前三个大运。\n"
                    "4. ## 流年流月运势：\n"
                    "   - 以“参考日期”做基准，给出**最近三年每年流年运势**要点（至少包含事业、财运、感情、健康）。\n"
                    "   - 结合**最近一个月份**和**下一个两个月份**，给出流月运势，指出关键吉凶事件。\n"
                    "5. ## 五行分析：说明八字中各五行的旺衰或缺失，并指出是否需要某个五行调和。\n"
                    "6. ## 喜用神：根据五行格局，建议最合适的喜用神，并说明理由。\n"
                    "7. ## 幸运色 / 幸运数字 / 幸运方位：根据缺失与调和需求，推荐具体颜色、数字和方位，并举例说明日常应用（如穿衣、摆件、家居布局）。\n"
                    "8. ## 事业学业：结合八字与流年流月，详细描述事业或学业机遇与挑战，并给出可落地的行动建议。\n"
                    "9. ## 感情桃花：说明当前感情/桃花运势趋势，结合流年流月给出择偶/交友建议，并提示重要吉日或时辰。\n"
                    "10. ## 健康：指出需关注的健康风险（如五行过旺/过弱对身体影响），并给出调养方案（饮食、运动、作息等）。\n"
                    "11. ## 财运：结合流年流月预测近期财运走势（正财 + 偏财），并给出理财/投资时机建议。\n"
                    "12. ## 六亲关系：简要说明父母、配偶、子女等与八字五行的相生相克关系，并给出家庭沟通或相处建议。\n"
                    "13. ## 结论与建议：最后做全局总结，语言要接地气，贴近生活。\n"
                    "请使用 Markdown 的标题、列表、粗体等格式，整篇文字不少于 1000 字。"
                )
                user_prompt = (
                    f"参考日期（今天）：**{today}**。\n\n"
                    f"**用户信息**：姓名：**{name}**；性别：**{gender}**；出生：**{birth_text}**。\n\n"
                    "请按照上述要求输出详细运势分析。"
                )

                with st.spinner("正在调用 ChatGPT 生成详细运势，请稍候……"):
                    answer = chat_completion(
                        client=client,
                        model=astro_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=2048
                    )

                # —— 渲染为 Markdown 输出 ——
                st.subheader("📜 八字运势结果（Markdown 格式）")
                st.markdown(answer)

                # —— 由于我们不再用 Python 生成 PDF，而是让用户直接“打印本页”，
                #    所以这里仅保留在页面上的渲染，不需要再拼 PDF 了。 
                #    如果你还想要生成 PDF 文件并下载，可以在这里保留之前的 PDF 生成代码。

    # ------------------- 两人星宿配对 -------------------
    else:
        st.markdown(f"**参考日期（今天）：{today}**")
        st.markdown("请分别输入两人的姓名、性别、出生日期与时辰：")
        col1, col2 = st.columns(2)
        with col1:
            name1 = st.text_input("姓名 1", key="pair_name1", help="例如：张三")
            gender1 = st.selectbox("性别 1", options=["男", "女"], key="pair_gender1")
            date1 = st.date_input(
                "出生日期 1",
                min_value=date(1900, 1, 1),
                max_value=date(2200, 12, 31),
                value=date(2000, 1, 1),
                key="pair_date1",
                help="请选择公历出生日期（1900–2200 年）"
            )
            time1 = st.time_input(
                "出生时辰 1",
                value=time(0, 0),
                key="pair_time1",
                help="请选择出生时辰（24 小时制）"
            )
        with col2:
            name2 = st.text_input("姓名 2", key="pair_name2", help="例如：李四")
            gender2 = st.selectbox("性别 2", options=["男", "女"], key="pair_gender2")
            date2 = st.date_input(
                "出生日期 2",
                min_value=date(1900, 1, 1),
                max_value=date(2200, 12, 31),
                value=date(2000, 1, 1),
                key="pair_date2",
                help="请选择公历出生日期（1900–2200 年）"
            )
            time2 = st.time_input(
                "出生时辰 2",
                value=time(0, 0),
                key="pair_time2",
                help="请选择出生时辰（24 小时制）"
            )

        if st.button("开始配对"):
            if not (name1.strip() and gender1 and date1 and time1 and name2.strip() and gender2 and date2 and time2):
                st.error("⚠️ 请完整填写：两人的姓名、性别、出生日期与时辰")
            else:
                birth1 = datetime.combine(date1, time1)
                birth2 = datetime.combine(date2, time2)
                birth_text1 = birth1.strftime("%Y年%m月%d日 %H时%M分")
                birth_text2 = birth2.strftime("%Y年%m月%d日 %H时%M分")

                system_prompt_pair = (
                    "你是一位资深的中文命理师，精通八字配对与星宿关系分析。"
                    "请以 **Markdown** 格式输出以下内容：\n"
                    "1. **个人简介**：重复列出双方姓名、性别、出生信息，以便报头。\n"
                    "2. **八字排盘**：分别列出双方“年柱、月柱、日柱、时柱”（天干地支），并简要说明各柱五行旺衰。\n"
                    "3. **星宿/生肖/天干地支配对**：详细分析两人五行相生相克、地支三合三会、天干合冲等关系，说明是否相合、相冲、相刑或相害，对双方感情或合作的影响。\n"
                    "4. **大运与流年对比**：结合“参考日期”，分别给出双方当前与下一步大运节点，并对比大运与当前流年运势，说明两人何时最易相合或相冲。\n"
                    "5. **配对吉凶评估**：根据八字和大运流年对比，给出整体配对吉凶结论，至少包含情感/婚姻层面与事业/合作层面两方面。\n"
                    "6. **日常相处建议**：结合双方八字和五行特点，给出具体生活化建议（如“宜在阴历X月Y日举办婚礼”，或“佩戴金饰、红色摆件以化解冲煞”）。\n"
                    "7. **化解或增益方法**：如果存在冲克或冲煞，说明可采用的化解方式（佩戴何种饰品、家中摆放何物、工作座位方位等）。\n"
                    "8. **结论**：最后给出全局总结，语言生动接地气，条理清晰，字数不少于 800 字。\n"
                )
                user_prompt_pair = (
                    f"参考日期（今天）：**{today}**。\n\n"
                    f"**用户1**：姓名：**{name1}**；性别：**{gender1}**；出生：**{birth_text1}**。\n"
                    f"**用户2**：姓名：**{name2}**；性别：**{gender2}**；出生：**{birth_text2}**。\n\n"
                    "请根据上述要求，输出完整八字配对与星宿关系分析。"
                )

                with st.spinner("正在调用 ChatGPT 进行星宿配对，请稍候……"):
                    answer_pair = chat_completion(
                        client=client,
                        model=astro_model,
                        messages=[
                            {"role": "system", "content": system_prompt_pair},
                            {"role": "user", "content": user_prompt_pair}
                        ],
                        temperature=0.7,
                        max_tokens=2048
                    )

                st.subheader("💞 两人星宿配对结果（Markdown 格式）")
                st.markdown(answer_pair)

                # —— 同样不再生成 PDF，用户可直接用页面右上角“🖨️ 打印本页”按钮来打印/存 PDF —— 

    # “八字运势” 分支结束后，跳过后续模型流程
    st.stop()

# ====================================================
# —— 以下为原有：多模态 / 视觉、语音识别、语音合成、图像生成、代码模型、聊天 等逻辑 —— 
#    （保持与之前一致，仅作复制粘贴） 
# ====================================================

# —— 公共上传控件：PDF & 图片 ——
if category != "语音识别":
    pdfs = st.sidebar.file_uploader(
        "上传 PDF(可选，多文件)", type=["pdf"], accept_multiple_files=True, key="pdf_uploader"
    )
    st.session_state.session_pdfs = list(pdfs) if pdfs else []
    truncate_pdf = st.sidebar.checkbox("启用 PDF 截断", help="勾选后按字数截断")
    trunc_chars = None
    if truncate_pdf:
        trunc_chars = st.sidebar.number_input("截断字数", min_value=1, value=2000, step=100)
    if category == "多模态 / 视觉" and is_vision_model(model):
        imgs = st.sidebar.file_uploader(
            "上传 图片(多选)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="img_uploader"
        )
        st.session_state.session_images = list(imgs) if imgs else []
    else:
        st.session_state.session_images = []
else:
    st.session_state.session_pdfs = []
    st.session_state.session_images = []

# —— 语音识别 专属 ——
webrtc_ctx = None
if category == "语音识别":
    webrtc_ctx = webrtc_streamer(
        key="audio_stream",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True
    )

# —— 语音合成 ——
if category == "语音合成":
    voice = st.sidebar.selectbox("选择 语音", ["alloy", "melody", "harmonia"], key="tts_voice")
    tts_prompt = st.sidebar.text_area("TTS 文本输入", height=100, key="tts_input")
    gen_tts = st.sidebar.button("生成语音", key="tts_button")
else:
    tts_prompt = None
    gen_tts = False

# —— 图像生成 ——
if category == "图像生成":
    image_prompt = st.sidebar.text_area("图像生成描述", height=100)
else:
    image_prompt = None

# —— 代码模型 ——
if category == "代码模型":
    code_request = st.chat_input("输入代码请求…")

# —— 主逻辑分支 ——
if category == "图像生成" and image_prompt:
    if st.sidebar.button("生成图片", key="gen_img_btn"):
        with st.spinner("生成中…"):
            r = client.images.generate(
                prompt=image_prompt,
                model=model if model.startswith("dall") else None,
                n=1
            )
        st.image(r.data[0].url)

elif category == "语音识别":
    # 上传文件识别
    upload_audio = st.sidebar.file_uploader(
        "上传 音频(可选)", type=["mp3", "wav", "ogg"], key="audio_uploader"
    )
    if upload_audio and st.sidebar.button("识别上传文件", key="recognize_upload"):
        with st.spinner("音频转写中…"):
            r = client.audio.transcriptions.create(
                file=upload_audio,
                model=model
            )
        st.write(r.text)

    # 录音并识别
    if webrtc_ctx and webrtc_ctx.audio_receiver and st.sidebar.button("录音并识别", key="recognize_stream"):
        frames = webrtc_ctx.audio_receiver.get_frames(timeout=3)
        if frames:
            # 合并 PCM 并打包 WAV
            pcm_data = b"".join([f.to_ndarray().tobytes() for f in frames])
            buf = BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm_data)
            buf.seek(0)
            wav_bytes = buf.getvalue()
            # 播放录音条
            st.audio(wav_bytes, format="audio/wav")
            # 转写
            with st.spinner("录音转写中…"):
                buf2 = BytesIO(wav_bytes)
                buf2.name = "recording.wav"
                r2 = client.audio.transcriptions.create(
                    file=buf2,
                    model=model
                )
            st.write(r2.text)

elif category == "语音合成" and gen_tts and tts_prompt:
    with st.spinner("生成语音…"):
        r = client.audio.speech.create(input=tts_prompt, voice=voice, model=model)
    try:
        audio_bytes = r.read()
    except:
        audio_bytes = bytes(r)
    st.audio(audio_bytes)

elif category == "代码模型" and code_request:
    with st.spinner("生成代码…"):
        if model == "codex-mini-latest":
            st.error("模型 'codex-mini-latest' 仅支持 /v1/responses，SDK 不支持。")
            code = ""
        else:
            try:
                r = client.completions.create(
                    model=model,
                    prompt=code_request,
                    max_tokens=512,
                    temperature=0.2
                )
                code = r.choices[0].text
            except:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": code_request}]
                )
                code = r.choices[0].message.content
    if code:
        st.code(code, language="python")

else:
    # 聊天 & 多模态
    if st.session_state.session_pdfs:
        st.markdown("**已上传 PDF 附件**")
        cols = st.columns(len(st.session_state.session_pdfs))
        for i, p in enumerate(st.session_state.session_pdfs):
            cols[i].markdown(f"📄 {p.name}")

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            first = content[0]
            text0 = first.get("text") if isinstance(first, dict) else first
            st.chat_message(role).markdown(text0)
            for part in content:
                if part.get("filename"):
                    with st.expander(part["filename"]):
                        st.markdown(part["text"])
            for part in content:
                if part.get("type") == "image_url":
                    st.chat_message(role).image(part["image_url"]["url"])
        else:
            st.chat_message(role).markdown(content)

    if prompt := st.chat_input("输入消息…"):
        parts = [{"type": "text", "text": prompt}]
        for pdf_file in st.session_state.session_pdfs:
            raw = pdf_file.read()
            txt = extract_text(BytesIO(raw))
            excerpt = txt[:trunc_chars] + "…" if truncate_pdf and trunc_chars else txt
            parts.append({"type": "text", "text": excerpt, "filename": pdf_file.name})
        for img in st.session_state.session_images:
            raw = img.read()
            mime, _ = guess_type(img.name)
            mime = mime or "application/octet-stream"
            url = f"data:{mime};base64," + base64.b64encode(raw).decode()
            parts.append({"type": "image_url", "image_url": {"url": url}})
        st.session_state.messages.append({"role": "user", "content": parts})

        st.chat_message("user").markdown(prompt)
        for part in parts[1:]:
            if part.get("filename"):
                with st.expander(part["filename"]):
                    st.markdown(part["text"])
            if part.get("type") == "image_url":
                st.chat_message("user").image(part["image_url"]["url"])

        st.session_state.session_pdfs = []
        st.session_state.session_images = []

        with st.spinner("思考中…"):
            r = client.chat.completions.create(
                model=model,
                messages=st.session_state.messages
            )
        ans = r.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.chat_message("assistant").markdown(ans)
