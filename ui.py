"""UI rendering: sidebar, quick-access buttons, and the chat interface.

Styled with Egyptian Flag & Digital Egypt official identity (Red, White, Gold, Black)
with pixel-perfect Right-to-Left (RTL) Arabic typography and layout.
"""

import base64
from pathlib import Path
from typing import Optional
import streamlit as st

from core.agents import stream_answer, summarize_history
from core.llm_factory import MODEL_CHOICES
from core.normalizer import ServiceResponse, normalize_line_breaks, normalize_response

COMMON_SERVICES = {
    "🚗 مخالفات رخصة مركبة": "أريد الاستعلام عن مخالفات رخصة مركبة",
    "💳 صرف بطاقة تموينية": "أريد الاستعلام عن صرف بطاقة تموينية",
    "⏱️ آخر مدة تأمينية": "ما هي آخر مدة تأمينية لي؟",
    "💼 مدد الاشتراك والأجور": "أريد معرفة مدد الاشتراك والأجور الخاصة بكل مدة في التأمين الاجتماعي",
    "🔢 الرقم التأميني": "ما هو رقمي التأميني؟",
}


def get_logo_base64() -> str:
    """Return base64 encoded string of the Digital Egypt official logo."""
    logo_path = Path(__file__).parent / "assets" / "digital_egypt_logo.jpg"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def apply_egypt_theme() -> None:
    """Inject custom CSS for Egyptian Flag & Digital Egypt luxury theme with strict RTL fix."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700;800;900&display=swap');

        /* Global Reset */
        html, body, [class*="css"], .stApp {
            font-family: 'Cairo', sans-serif !important;
            direction: rtl !important;
            text-align: right !important;
            background-color: #0f1117 !important;
            color: #f3f4f6 !important;
        }

        .main .block-container {
            max-width: 920px !important;
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
            margin: 0 auto !important;
            direction: rtl !important;
            text-align: right !important;
        }

        /* Top Egyptian Flag Ribbon */
        .egypt-flag-bar {
            width: 100%;
            height: 6px;
            background: linear-gradient(to left, 
                #C8102E 0%, #C8102E 33%, 
                #FFFFFF 33%, #FFFFFF 66%, 
                #111111 66%, #111111 100%);
            border-radius: 4px;
            margin-bottom: 18px;
            box-shadow: 0 2px 14px rgba(200, 16, 46, 0.45);
        }

        /* Egyptian Hero Header */
        .egypt-hero {
            background: linear-gradient(135deg, rgba(200, 16, 46, 0.16) 0%, rgba(26, 26, 32, 0.95) 50%, rgba(212, 175, 55, 0.15) 100%);
            border: 1px solid rgba(212, 175, 55, 0.4);
            border-radius: 18px;
            padding: 22px 26px;
            margin-bottom: 24px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
            direction: rtl !important;
            text-align: right !important;
        }

        .hero-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            direction: rtl !important;
        }

        .hero-logo-box {
            background: #ffffff;
            padding: 8px 14px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .hero-logo-img {
            height: 52px;
            width: auto;
            object-fit: contain;
        }

        .hero-content {
            flex-grow: 1;
            text-align: right !important;
        }

        .hero-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: #FFFFFF;
            margin: 0 0 6px 0;
            line-height: 1.3;
            text-align: right !important;
        }

        .hero-title .gold-text {
            background: linear-gradient(135deg, #F5D77F 0%, #D4AF37 50%, #AA771C 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            color: #D1D5DB;
            font-size: 0.98rem;
            margin: 0;
            font-weight: 400;
            line-height: 1.6;
            text-align: right !important;
        }

        .egypt-badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(200, 16, 46, 0.25);
            border: 1px solid rgba(200, 16, 46, 0.5);
            color: #FFD2D2;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-top: 8px;
        }

        /* Quick Service Chips */
        .service-heading {
            color: #E5B869;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            direction: rtl !important;
            text-align: right !important;
        }

        /* Streamlit Button Layout & No-Truncation Reset */
        div.stButton,
        div[data-testid="stButton"],
        div[data-testid="stBaseButton-secondary"] {
            width: 100% !important;
        }

        .stButton button,
        .stButton button *,
        div[data-testid="stButton"] button,
        div[data-testid="stButton"] button *,
        div[data-testid="stBaseButton-secondary"] button,
        div[data-testid="stBaseButton-secondary"] button *,
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-secondary"] *,
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-secondary"] *,
        button[kind="secondary"],
        button[kind="secondary"] * {
            white-space: normal !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            word-break: break-word !important;
            overflow: visible !important;
            text-overflow: unset !important;
            text-overflow: clip !important;
            max-width: 100% !important;
        }

        div.stButton > button,
        div[data-testid="stButton"] button,
        button[data-testid="baseButton-secondary"],
        button[data-testid="stBaseButton-secondary"],
        button[kind="secondary"] {
            background: linear-gradient(135deg, #22222a 0%, #18181f 100%) !important;
            color: #F3F4F6 !important;
            border: 1px solid rgba(212, 175, 55, 0.35) !important;
            border-radius: 12px !important;
            padding: 8px 10px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            font-family: 'Cairo', sans-serif !important;
            direction: rtl !important;
            text-align: center !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
            width: 100% !important;
            height: auto !important;
            min-height: 48px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        div.stButton > button *,
        div[data-testid="stButton"] button *,
        button[data-testid="baseButton-secondary"] *,
        button[data-testid="stBaseButton-secondary"] *,
        button[kind="secondary"] * {
            font-family: 'Cairo', sans-serif !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            line-height: 1.35 !important;
            text-align: center !important;
            margin: 0 !important;
            padding: 0 !important;
            color: #F3F4F6 !important;
        }

        div.stButton > button:hover,
        div[data-testid="stButton"] button:hover,
        button[data-testid="baseButton-secondary"]:hover,
        button[data-testid="stBaseButton-secondary"]:hover,
        button[kind="secondary"]:hover {
            background: linear-gradient(135deg, rgba(200, 16, 46, 0.45) 0%, rgba(212, 175, 55, 0.35) 100%) !important;
            border-color: #E5B869 !important;
            color: #FFFFFF !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(212, 175, 55, 0.3) !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            width: 330px !important;
            min-width: 330px !important;
            background: linear-gradient(180deg, #15151c 0%, #0d0d10 100%) !important;
            border-left: 1px solid rgba(212, 175, 55, 0.2) !important;
            direction: rtl !important;
            text-align: right !important;
        }

        section[data-testid="stSidebar"] > div:first-child,
        div[data-testid="stSidebarContent"],
        div[data-testid="stSidebarUserContent"] {
            width: 100% !important;
        }

        section[data-testid="stSidebar"] * {
            direction: rtl !important;
            text-align: right !important;
            font-family: 'Cairo', sans-serif !important;
        }

        .sidebar-logo-card {
            background: #ffffff;
            padding: 12px;
            border-radius: 14px;
            text-align: center;
            margin-bottom: 14px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }

        .sidebar-logo-card img {
            max-width: 85%;
            height: auto;
        }

        .sidebar-brand-box {
            text-align: center !important;
            margin-bottom: 20px;
            padding-bottom: 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .sidebar-brand-box h4 {
            color: #E5B869 !important;
            margin: 6px 0 2px 0 !important;
            font-size: 1.15rem !important;
            font-weight: 800 !important;
            text-align: center !important;
        }

        .sidebar-brand-box p {
            color: #9CA3AF !important;
            font-size: 0.8rem !important;
            margin: 0 !important;
            text-align: center !important;
        }

        /* =========================================================
           CHAT MESSAGE CONTAINER LAYOUT
           ========================================================= */
        div[data-testid="stChatMessage"] {
            border-radius: 16px !important;
            padding: 20px 22px !important;
            margin-bottom: 20px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
            display: flex !important;
            flex-direction: row-reverse !important;
            align-items: flex-start !important;
            gap: 14px !important;
        }

        div[data-testid="stChatMessageAvatar"] {
            flex-shrink: 0 !important;
        }

        div[data-testid="stChatMessageContent"] {
            flex: 1 1 0% !important;
            min-width: 0 !important;
            direction: rtl !important;
        }

        /* User Message */
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
            background: linear-gradient(135deg, rgba(200, 16, 46, 0.2) 0%, rgba(35, 20, 26, 0.85) 100%) !important;
            border: 1px solid rgba(200, 16, 46, 0.4) !important;
            border-right: 5px solid #C8102E !important;
        }

        /* Assistant Message */
        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
            background: linear-gradient(135deg, rgba(28, 28, 36, 0.95) 0%, rgba(18, 18, 24, 0.98) 100%) !important;
            border: 1px solid rgba(212, 175, 55, 0.25) !important;
            border-right: 5px solid #D4AF37 !important;
        }

        /* =========================================================
           ROOT CAUSE FIX:
           Streamlit renders stMarkdownContainer as a flex column
           with align-items: flex-start. This causes short paragraphs
           (like bold section headings) to shrink to content width
           and snap to the physical LEFT edge, while long text fills
           the width and appears right-aligned.
           Fix: Force block layout on the markdown container so ALL
           children (p, ul, ol) behave as normal block elements and
           inherit the RTL text-align correctly.
           ========================================================= */
        [data-testid="stMarkdownContainer"],
        div[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
            display: block !important;
            width: 100% !important;
            direction: rtl !important;
            text-align: right !important;
        }

        /* Remove anchor icons from any headings that may appear */
        .stHeaderAnchor,
        a.stHeaderAnchor,
        [data-testid="stMarkdownContainer"] a.stHeaderAnchor {
            display: none !important;
        }

        /* Markdown headings (h1-h6) — kept for fallback in case LLM uses ### */
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] h5,
        [data-testid="stMarkdownContainer"] h6 {
            display: block !important;
            width: 100% !important;
            direction: rtl !important;
            text-align: right !important;
            color: #F5D77F !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            margin: 20px 0 10px 0 !important;
            padding-bottom: 5px !important;
            border-bottom: 1px solid rgba(212, 175, 55, 0.2) !important;
        }

        /* =========================================================
           PARAGRAPHS
           ========================================================= */
        [data-testid="stMarkdownContainer"] p {
            display: block !important;
            width: 100% !important;
            direction: rtl !important;
            text-align: right !important;
            font-size: 1rem !important;
            line-height: 1.95 !important;
            margin: 0 0 12px 0 !important;
            padding: 0 !important;
            color: #E5E7EB !important;
        }

        /* Bold text used as section headings */
        [data-testid="stMarkdownContainer"] p strong,
        [data-testid="stMarkdownContainer"] strong {
            color: #F5D77F !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
        }

        /* =========================================================
           LISTS — RTL FIX
           In RTL mode:
           - List marker is at the START (right side) of the item
           - padding-right creates the gutter for the marker
           - padding-left must be 0 to avoid pushing content leftward
           ========================================================= */
        [data-testid="stMarkdownContainer"] ul,
        [data-testid="stMarkdownContainer"] ol {
            display: block !important;
            direction: rtl !important;
            text-align: right !important;
            margin: 4px 0 18px 0 !important;
            padding: 0 !important;
            padding-right: 1.4rem !important;
            padding-left: 0 !important;
            list-style-position: outside !important;
        }

        [data-testid="stMarkdownContainer"] ul {
            list-style-type: disc !important;
        }

        [data-testid="stMarkdownContainer"] ol {
            list-style-type: decimal !important;
        }

        [data-testid="stMarkdownContainer"] li {
            display: list-item !important;
            direction: rtl !important;
            text-align: right !important;
            color: #E5E7EB !important;
            font-size: 0.98rem !important;
            line-height: 1.9 !important;
            margin-bottom: 10px !important;
            padding: 0 !important;
        }

        /* Nested lists */
        [data-testid="stMarkdownContainer"] li ul,
        [data-testid="stMarkdownContainer"] li ol {
            margin: 6px 1.2rem 6px 0 !important;
        }

        /* =========================================================
           CHAT INPUT BOX
           ========================================================= */
        div[data-testid="stChatInput"] {
            direction: rtl !important;
        }

        div[data-testid="stChatInput"] textarea {
            font-family: 'Cairo', sans-serif !important;
            direction: rtl !important;
            text-align: right !important;
            font-size: 0.98rem !important;
            line-height: 1.6 !important;
        }

        .stChatInputContainer {
            border-radius: 16px !important;
            border: 1px solid rgba(212, 175, 55, 0.45) !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
            background: #181820 !important;
        }

        .stChatInputContainer:focus-within {
            border-color: #E5B869 !important;
            box-shadow: 0 0 18px rgba(229, 184, 105, 0.35) !important;
        }

        /* =========================================================
           FOOTER
           ========================================================= */
        .egypt-footer {
            text-align: center !important;
            color: #9CA3AF;
            font-size: 0.85rem;
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            direction: rtl !important;
        }

        /* =========================================================
           RESPONSIVE
           ========================================================= */
        @media (max-width: 768px) {
            div[data-testid="stChatMessage"] {
                padding: 14px 14px !important;
                gap: 10px !important;
            }
            [data-testid="stMarkdownContainer"] p {
                font-size: 0.95rem !important;
            }
            [data-testid="stMarkdownContainer"] li {
                font-size: 0.93rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )




def render_header() -> None:
    """Render prestigious Egyptian Government AI Assistant header with official logo."""
    logo_b64 = get_logo_base64()
    logo_html = ""
    if logo_b64:
        logo_html = f'<div class="hero-logo-box"><img class="hero-logo-img" src="data:image/jpeg;base64,{logo_b64}" alt="مصر الرقمية" /></div>'

    st.markdown('<div class="egypt-flag-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="egypt-hero">
            <div class="hero-row">
                <div class="hero-content">
                    <div class="hero-title">
                        <span>المساعد الرقمي لخدمات <span class="gold-text">مصر الرقمية</span></span>
                    </div>
                    <p class="hero-subtitle">المنظومة الذكية الموحدة للإرشاد والاستعلام الفوري عن خدمات بوابة مصر الرقمية</p>
                    <div class="egypt-badge-pill">
                        <span>🇪🇬</span>
                        <span>جمهورية مصر العربية</span>
                    </div>
                </div>
                {logo_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render themed sidebar with official logo and model controls."""
    logo_b64 = get_logo_base64()
    logo_sidebar_html = ""
    if logo_b64:
        logo_sidebar_html = f"""
        <div class="sidebar-logo-card">
            <img src="data:image/jpeg;base64,{logo_b64}" alt="مصر الرقمية" />
        </div>
        """

    st.sidebar.markdown(
        f"""
        {logo_sidebar_html}
        <div class="sidebar-brand-box">
            <h4>بوابة مصر الرقمية</h4>
            <p>الخدمات الحكومية الذكية</p>
            <div style="margin-top: 8px; height: 3px; background: linear-gradient(to left, #C8102E, #FFFFFF, #111111); border-radius: 2px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("#### ⚙️ محرك الذكاء الاصطناعي")

    if st.session_state.get("model_choice") not in MODEL_CHOICES:
        st.session_state["model_choice"] = MODEL_CHOICES[0]

    return st.sidebar.selectbox(
        "اختر النموذج:",
        MODEL_CHOICES,
        key="model_choice",
    )


def render_quick_services() -> None:
    """Render styled quick access chips for popular services."""
    st.markdown(
        '<div class="service-heading">⚡ الخدمات الأكثر استعلاماً:</div>',
        unsafe_allow_html=True,
    )
    items = list(COMMON_SERVICES.items())

    # Row 1: First 3 services
    row1 = items[:3]
    cols1 = st.columns(len(row1))
    for i, (label, query) in enumerate(row1):
        if cols1[i].button(label, key=f"quick_srv_{i}", use_container_width=True):
            st.session_state["preset_query"] = query

    # Row 2: Remaining 2 services
    row2 = items[3:]
    cols2 = st.columns(len(row2))
    for j, (label, query) in enumerate(row2):
        idx = len(row1) + j
        if cols2[j].button(label, key=f"quick_srv_{idx}", use_container_width=True):
            st.session_state["preset_query"] = query


def render_history(memory) -> None:
    """Display conversation history with Egyptian-themed avatars and styling."""
    for msg in memory.messages:
        role = "user" if msg.type == "human" else "assistant"
        avatar = "👤" if role == "user" else "🦅"
        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                service = normalize_response(msg.content)
                if service is not None:
                    render_service_card(service)
                else:
                    clean = normalize_line_breaks(msg.content)
                    st.markdown(clean)
            else:
                st.markdown(msg.content)



def get_user_input() -> Optional[str]:
    """Capture user input via chat input or quick service button."""
    default_input = st.session_state.pop("preset_query", None)
    user_input = st.chat_input("اسأل عن أي خدمة حكومية، شروطها، خطواتها، أو مستنداتها...")
    return user_input or default_input


def render_service_card(resp: ServiceResponse) -> None:
    """Render a normalized ServiceResponse as a clean RTL HTML card.

    This is the ONLY function that renders structured service data.
    It never sees raw LLM text — only the already-normalized dataclass.
    """

    def _row(label: str, content: str) -> str:
        return (
            f'<tr>'
            f'<td class="sc-label">{label}</td>'
            f'<td class="sc-detail">{content}</td>'
            f'</tr>'
        )

    def _list_html(items: list) -> str:
        if not items:
            return "<span class='sc-empty'>لا توجد معلومات متاحة</span>"
        lis = "".join(f"<li>{item}</li>" for item in items)
        return f"<ul class='sc-list'>{lis}</ul>"

    def _ol_html(items: list) -> str:
        if not items:
            return "<span class='sc-empty'>لا توجد معلومات متاحة</span>"
        lis = "".join(f"<li>{item}</li>" for item in items)
        return f"<ol class='sc-list'>{lis}</ol>"

    rows = []
    if resp.description:
        rows.append(_row("وصف الخدمة", f"<p class='sc-text'>{resp.description}</p>"))
    if resp.conditions:
        rows.append(_row("الشروط والمتطلبات", _list_html(resp.conditions)))
    if resp.required_documents:
        rows.append(_row("المستندات المطلوبة", _list_html(resp.required_documents)))
    if resp.steps:
        rows.append(_row("الخطوات اللازمة", _ol_html(resp.steps)))
    if resp.notes:
        rows.append(_row("ملاحظة", f"<p class='sc-text'>{resp.notes}</p>"))
    if resp.support:
        rows.append(_row("الدعم", f"<p class='sc-text'>{resp.support}</p>"))
    if resp.similar_services:
        rows.append(_row("خدمات مشابهة", _list_html(resp.similar_services)))

    title_html = (
        f"<div class='sc-title'>{resp.service_name}</div>" if resp.service_name else ""
    )
    table_html = (
        f"<table class='sc-table'>{''.join(rows)}</table>" if rows else ""
    )

    st.markdown(
        f"""
        <style>
        .sc-wrap {{ direction: rtl; text-align: right; font-family: 'Cairo', sans-serif; }}
        .sc-title {{
            font-size: 1.2rem; font-weight: 800; color: #F5D77F;
            margin-bottom: 16px; padding-bottom: 10px;
            border-bottom: 2px solid rgba(212,175,55,0.4);
        }}
        .sc-table {{
            width: 100%; border-collapse: collapse;
            direction: rtl; text-align: right;
        }}
        .sc-table tr {{
            border-bottom: 1px solid rgba(255,255,255,0.07);
            vertical-align: top;
        }}
        .sc-table tr:last-child {{ border-bottom: none; }}
        .sc-label {{
            font-size: 0.97rem; font-weight: 700; color: #E5B869;
            padding: 12px 0 12px 20px; white-space: nowrap;
            width: 1%; /* shrink to content */
        }}
        .sc-detail {{
            font-size: 0.96rem; color: #E5E7EB; padding: 12px 0;
            line-height: 1.85;
        }}
        .sc-text {{ margin: 0; padding: 0; }}
        .sc-list {{
            margin: 4px 0 0 0;
            padding-right: 1.3rem;
            padding-left: 0;
            list-style-position: outside;
        }}
        .sc-list li {{
            margin-bottom: 8px;
            direction: rtl;
            text-align: right;
        }}
        .sc-empty {{ color: #9CA3AF; font-style: italic; }}
        </style>
        <div class='sc-wrap'>
            {title_html}
            {table_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer(chain_with_history, session_id: str, user_input: str) -> None:
    """Stream response then normalize and render it.

    Flow:
    1. Stream raw LLM text (shown incrementally with a cursor).
    2. After streaming completes, pass the full text through
       ``normalize_response()``.
    3. If a ``ServiceResponse`` is extracted → render via ``render_service_card()``.
    4. Otherwise (general question / error message) → render via ``st.markdown()``
       after cleaning any residual line-break variants.
    """
    st.chat_message("user", avatar="👤").markdown(user_input)

    with st.chat_message("assistant", avatar="🦅"):
        full_answer = ""
        placeholder = st.empty()

        # --- stream phase ---
        for chunk in stream_answer(chain_with_history, session_id, user_input):
            full_answer += chunk
            
            # Hotfix: Qwen on Groq may stream utf-8 bytes incorrectly decoded as latin1.
            # Good Arabic will raise UnicodeEncodeError and be ignored. Mojibake will be fixed.
            display_text = full_answer
            try:
                display_text = full_answer.encode('latin1').decode('utf-8', errors='replace')
            except Exception:
                pass
            
            placeholder.markdown(display_text + "▌")

        # --- normalize phase ---
        placeholder.empty()  # clear the streaming placeholder

        # Apply the hotfix permanently to the final answer
        try:
            full_answer = full_answer.encode('latin1').decode('utf-8')
        except Exception:
            pass

        service = normalize_response(full_answer)
        if service is not None:
            # Structured service data → rich card
            render_service_card(service)
        else:
            # General response (greeting, error, etc.) → clean markdown
            clean = normalize_line_breaks(full_answer)
            st.markdown(clean)


def render_summary_action(llm, memory) -> None:
    """Sidebar action to summarize conversation."""
    st.sidebar.divider()
    if st.sidebar.button("🧾 تلخيص المحادثة الحالية"):
        if not memory.messages:
            st.sidebar.warning("لا توجد رسائل سابقة لتلخيصها بعد.")
            return

        with st.sidebar.status("جاري إعداد ملخص رسمي للمحادثة...", expanded=True):
            summary = summarize_history(llm, memory.messages)
        st.sidebar.markdown("### 📋 ملخص المحادثة")
        st.sidebar.info(summary)


def render_footer() -> None:
    """Render footer with Egyptian identity notice."""
    st.markdown(
        """
        <div class="egypt-footer">
            🇪🇬 منظومة المساعد الرقمي الموحد — مدعومة بتقنيات الذكاء الاصطناعي لبوابة مصر الرقمية
        </div>
        """,
        unsafe_allow_html=True,
    )
