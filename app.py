"""
YouTube RAG Chatbot — Streamlit Cloud-ready
Credentials loaded from .env locally or st.secrets on Streamlit Cloud.
"""

import os
import re
import streamlit as st

# ─── MUST be first Streamlit call ────────────────────────────────────────────
st.set_page_config(
    page_title="YT RAG · Chat with any video",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── SECURE CREDENTIALS via .env + st.secrets ────────────────────────────────
from dotenv import load_dotenv
load_dotenv(override=True)

def _get_secret(key: str, env_fallback: str | None = None) -> str | None:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get(env_fallback or key)


GROQ_API_KEY = _get_secret("GROQ_API_KEY")
DATABASE_URL = _get_secret("DATABASE_URL")
HF_TOKEN     = _get_secret("HUGGINGFACEHUB_API_TOKEN")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# ─── LAZY IMPORTS ────────────────────────────────────────────────────────────
from sqlalchemy import create_engine, text
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, VideoUnavailable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ─── STYLES ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}

.stApp { background: #0f0f13; color: #e8e8f0; }

[data-testid="stSidebar"] {
    background: #16161f !important;
    border-right: 1px solid #22223a !important;
}
[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

header[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stDecoration"]    { display: none; }

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.25rem;
}
.brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #ff4e50, #f9d423);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}
.brand-name {
    font-size: 1.25rem; font-weight: 700;
    background: linear-gradient(135deg, #ff4e50, #f9d423);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.brand-tagline { color: #555; font-size: 0.78rem; margin-bottom: 1.25rem; }

.hr { border: none; border-top: 1px solid #22223a; margin: 1rem 0; }

.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #555; margin-bottom: 0.5rem;
}

.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600;
}
.pill-green  { background:#0d2e1a; color:#4ade80; border:1px solid #166534; }
.pill-yellow { background:#2a1f00; color:#f9d423; border:1px solid #854d0e; }
.pill-red    { background:#2e0d0d; color:#f87171; border:1px solid #7f1d1d; }

.video-card {
    background: #1a1a26; border: 1px solid #22223a;
    border-radius: 12px; padding: 12px 14px; margin-top: 0.75rem;
}
.video-card-title { font-weight: 600; font-size: 0.85rem; margin-bottom: 6px; color:#c0c0d8; }
.video-id-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; color: #888; background: #0f0f1a;
    padding: 3px 8px; border-radius: 6px; display: inline-block;
}
.video-stat { font-size: 0.78rem; color:#555; margin-top: 8px; }

.stButton button[kind="secondary"],
.stButton > button {
    background: #1a1a26 !important;
    border: 1px solid #2a2a3e !important;
    color: #a0a0c0 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 6px 12px !important;
    text-align: left !important;
    width: 100%;
    transition: border-color 0.15s, color 0.15s;
}
.stButton > button:hover {
    border-color: #ff4e50 !important;
    color: #ff4e50 !important;
    background: #1f1520 !important;
}

div[data-testid="column"] .stButton > button,
.load-btn .stButton > button {
    background: linear-gradient(135deg,#ff4e50,#f9d423) !important;
    color: #0f0f13 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    padding: 10px !important;
}

.chat-wrap { display: flex; flex-direction: column; gap: 10px; }

.msg-row-user { display: flex; justify-content: flex-end; }
.msg-row-bot  { display: flex; justify-content: flex-start; }

.bubble {
    max-width: 78%; padding: 11px 15px;
    font-size: 0.92rem; line-height: 1.65;
    border-radius: 16px; word-break: break-word;
}
.bubble-user {
    background: #1e1e30;
    border: 1px solid #2a2a42;
    border-bottom-right-radius: 4px;
    color: #ddddf0;
}
.bubble-bot {
    background: #191926;
    border: 1px solid #ff4e5026;
    border-left: 3px solid #ff4e50;
    border-bottom-left-radius: 4px;
    color: #ddddf0;
}

.msg-label {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 4px;
}
.msg-label-user { color:#555; text-align:right; }
.msg-label-bot  { color:#ff4e50; }

.empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 4rem 1rem; gap: 0.5rem;
    text-align: center;
}
.empty-icon   { font-size: 2.8rem; }
.empty-title  { font-size: 1.05rem; font-weight: 600; color: #444; }
.empty-hint   { font-size: 0.83rem; color: #333; }

.page-header {
    display: flex; align-items: baseline; gap: 14px;
    margin-bottom: 0.25rem;
}
.page-title {
    font-size: 1.55rem; font-weight: 700;
    background: linear-gradient(135deg,#ff4e50,#f9d423);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing:-0.02em;
}

.stTextInput input {
    background: #16161f !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-size: 0.92rem !important;
    padding: 12px 16px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus {
    border-color: #ff4e50 !important;
    box-shadow: 0 0 0 2px #ff4e5018 !important;
}

.callout {
    padding: 12px 16px; border-radius: 10px;
    font-size: 0.87rem; margin: 0.5rem 0;
}
.callout-warn {
    background: #2a1f00; border-left: 3px solid #f9d423; color: #c9a920;
}
.callout-err {
    background: #2e0d0d; border-left: 3px solid #f87171; color: #ef8080;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─── GUARDS ──────────────────────────────────────────────────────────────────
def _missing_secrets() -> list[str]:
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    return missing


# ─── UTILITIES ───────────────────────────────────────────────────────────────
_YT_PATTERN = re.compile(
    r"(?:v=|youtu\.be/|/shorts/|/embed/|/v/)([A-Za-z0-9_\-]{11})"
)

def extract_video_id(raw: str) -> str | None:
    raw = raw.strip()
    m = _YT_PATTERN.search(raw)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_\-]{11}", raw):
        return raw
    return None


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def drop_pgvector_tables():
    """Drop old PGVector tables so they are recreated with the correct schema."""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection CASCADE;"))
        conn.commit()
    engine.dispose()


# ─── CACHED RESOURCES ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=512,
        api_key=GROQ_API_KEY,
    )


@st.cache_resource(show_spinner=False)
def build_chain(video_id: str):
    """
    Build the full RAG chain for a video.
    Cached by video_id — subsequent loads of the same video are instant.
    """
    # 1. Transcript
    ytt = YouTubeTranscriptApi()
    try:
        fetched = ytt.fetch(video_id, languages=["en"])
    except NoTranscriptFound:
        fetched = ytt.fetch(video_id)
    transcript = " ".join(part.text for part in fetched)

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # 3. Drop old tables to avoid schema mismatch, then build vector store
    drop_pgvector_tables()

    embeddings = load_embeddings()
    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"yt_{video_id}",
        connection_string=DATABASE_URL,
        pre_delete_collection=False,
    )

    # 4. Retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )

    # 5. Prompt
    prompt = PromptTemplate(
        template="""You are a helpful assistant that answers questions about a YouTube video.
Answer ONLY from the transcript context provided below.
If the context does not contain enough information, say exactly:
"I don't have enough information from this video to answer that."
Be concise, clear, and do not make anything up.

Transcript context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    # 6. Chain
    llm = load_llm()
    parallel = RunnableParallel({
        "context":  retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    chain = parallel | prompt | llm | StrOutputParser()

    return chain, len(chunks)


# ─── SESSION STATE ────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":     [],
        "chain":        None,
        "video_id":     None,
        "video_loaded": False,
        "chunk_count":  0,
        "pending_q":    "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🎬</div>
        <div class="brand-name">YT RAG</div>
    </div>
    <div class="brand-tagline">Chat with any YouTube video using AI</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ── Secrets check ─────────────────────────────────────────────────────────
    missing = _missing_secrets()
    if missing:
        st.markdown(f"""
        <div class="callout callout-err">
            <b>⚠ Missing secrets:</b><br>
            {', '.join(f'<code>{k}</code>' for k in missing)}<br><br>
            Add them to your <code>.env</code> file in the project root,
            or in <b>Streamlit Cloud → App settings → Secrets</b>.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Video input ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Load a video</div>', unsafe_allow_html=True)
    video_input = st.text_input(
        "YouTube URL or video ID",
        placeholder="youtube.com/watch?v=... or 11-char ID",
        label_visibility="collapsed",
    )

    st.markdown('<div class="load-btn">', unsafe_allow_html=True)
    load_clicked = st.button("⚡ Load & Index Video", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if load_clicked:
        if not video_input.strip():
            st.warning("Paste a YouTube URL or video ID first.")
        else:
            vid_id = extract_video_id(video_input)
            if not vid_id:
                st.markdown(
                    '<div class="callout callout-err">Could not parse a valid video ID from that input.</div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Fetching transcript & building vector index…"):
                    try:
                        chain, n_chunks = build_chain(vid_id)
                        st.session_state.chain        = chain
                        st.session_state.video_id     = vid_id
                        st.session_state.video_loaded = True
                        st.session_state.chunk_count  = n_chunks
                        st.session_state.messages     = []
                        st.rerun()
                    except VideoUnavailable:
                        st.error("This video is unavailable or private.")
                    except NoTranscriptFound:
                        st.error("No English transcript found for this video.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ── Active video card ──────────────────────────────────────────────────────
    if st.session_state.video_loaded:
        st.markdown(f"""
        <div class="video-card">
            <div class="video-card-title">📺 Active video</div>
            <div class="video-id-chip">{st.session_state.video_id}</div>
            <div class="video-stat">{st.session_state.chunk_count} chunks indexed · MMR retrieval</div>
        </div>
        """, unsafe_allow_html=True)

        yt_url = f"https://youtube.com/watch?v={st.session_state.video_id}"
        st.markdown(f'<a href="{yt_url}" target="_blank" style="font-size:0.8rem; color:#666;">↗ Open on YouTube</a>', unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ── Quick questions ────────────────────────────────────────────────────────
    if st.session_state.video_loaded:
        st.markdown('<div class="section-label">Quick questions</div>', unsafe_allow_html=True)
        suggestions = [
            "Summarise this video",
            "What are the main topics?",
            "What is the key takeaway?",
            "List the most important points",
            "Are there any actionable tips?",
        ]
        for s in suggestions:
            if st.button(s, key=f"sug__{s}"):
                st.session_state["pending_q"] = s
                st.rerun()

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ── Clear chat ─────────────────────────────────────────────────────────────
    if st.session_state.messages:
        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ─── MAIN AREA ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([5, 1], gap="small")
with col_left:
    st.markdown("""
    <div class="page-header">
        <div class="page-title">YouTube RAG Chatbot</div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    if st.session_state.video_loaded:
        st.markdown('<div style="padding-top:6px"><span class="pill pill-green">● Live</span></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding-top:6px"><span class="pill pill-yellow">○ No video</span></div>',
                    unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    if not st.session_state.video_loaded:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎬</div>
            <div class="empty-title">No video loaded yet</div>
            <div class="empty-hint">Paste a YouTube URL in the sidebar to index a video, then ask anything about it.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <div class="empty-title">Ask anything about the video</div>
            <div class="empty-hint">Type below or pick a quick question from the sidebar.</div>
        </div>
        """, unsafe_allow_html=True)
else:
    chat_html = ['<div class="chat-wrap">']
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html.append(f"""
            <div class="msg-row-user">
                <div>
                    <div class="msg-label msg-label-user">You</div>
                    <div class="bubble bubble-user">{msg["content"]}</div>
                </div>
            </div>""")
        else:
            chat_html.append(f"""
            <div class="msg-row-bot">
                <div>
                    <div class="msg-label msg-label-bot">🤖 Assistant</div>
                    <div class="bubble bubble-bot">{msg["content"]}</div>
                </div>
            </div>""")
    chat_html.append('</div>')
    st.markdown("".join(chat_html), unsafe_allow_html=True)

st.markdown('<div class="hr" style="margin-top:1.5rem"></div>', unsafe_allow_html=True)

# ── Input row ──────────────────────────────────────────────────────────────────
inp_col, btn_col = st.columns([6, 1], gap="small")
with inp_col:
    pending = st.session_state.pop("pending_q", "")
    user_q = st.text_input(
        "Question",
        value=pending,
        placeholder="Ask anything about the video…",
        label_visibility="collapsed",
        key="user_input",
    )
with btn_col:
    send = st.button("Send →", use_container_width=True)

# ── Handle send ────────────────────────────────────────────────────────────────
question = user_q.strip() if send and user_q.strip() else ""
if question:
    if not st.session_state.video_loaded:
        st.markdown(
            '<div class="callout callout-warn">⚠ Load a YouTube video first using the sidebar.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Retrieving context & generating answer…"):
            try:
                answer = st.session_state.chain.invoke(question)
            except Exception as e:
                answer = f"❌ Something went wrong: {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()