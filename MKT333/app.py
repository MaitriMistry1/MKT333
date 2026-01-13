###########################################
# MKT333 AI Assistant (Streamlit + HF Inference)
# - Hosted LLM via Hugging Face Inference (no Ollama)
# - Local embeddings + FAISS retrieval with citations
###########################################

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import fitz  # PyMuPDF
import numpy as np

try:
    import faiss  # faiss-cpu
except Exception as e:
    faiss = None

from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# -----------------------------
# Config
# -----------------------------
PDF_DIR = os.path.join(os.path.dirname(__file__), "pdfs")

# Embeddings: primary + fallback (in case HF download/model id fails)
EMBED_MODEL_PRIMARY = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"

# Hosted LLM model (must be accessible to your HF account/token)
# If a model is "gated", you must accept its license on HF first.
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Token (Secrets / Env)
# -----------------------------
def get_hf_token() -> Optional[str]:
    # Streamlit secrets first
    try:
        tok = st.secrets.get("HF_TOKEN", None)
        if tok:
            return str(tok).strip()
    except Exception:
        pass

    # Local env vars next
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return tok.strip() if tok else None


HF_TOKEN = get_hf_token()


# -----------------------------
# Streamlit Page UI
# -----------------------------
st.set_page_config(
    page_title="MKT 333 — Beer AI & Video Games",
    page_icon="🍺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Theme toggle state
if "ui_dark_mode" not in st.session_state:
    st.session_state.ui_dark_mode = True

left, right = st.columns([0.97, 0.20], vertical_alignment="center")
with left:
    st.markdown(
        """
        <div class="top-banner">
          <div class="hero-title">Beer • AI • Video Games</div>
          <div class="hero-sub">Ask the course PDFs. Get clean, cited answers.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.session_state.ui_dark_mode = st.toggle("Dark mode", value=st.session_state.ui_dark_mode)

# Theme variables (UI CSS)
if st.session_state.ui_dark_mode:
    bg = "#0b0d12"
    panel = "rgba(15, 18, 28, 0.86)"
    panel_solid = "#0f121c"
    text = "#e7eaf0"
    mut = "#a7b0c0"
    border = "rgba(231,234,240,0.12)"
    accent = "#990000"       # USC cardinal
    accent2 = "#ffcc00"      # USC gold
    user_bg = "rgba(30, 34, 46, 0.92)"
    ai_bg = "rgba(153, 0, 0, 0.22)"
    input_bg = "rgba(12, 14, 22, 0.85)"
else:
    bg = "#fafafa"
    panel = "rgba(255,255,255,0.92)"
    panel_solid = "#ffffff"
    text = "#0b1220"
    mut = "#4b5563"
    border = "rgba(11,18,32,0.10)"
    accent = "#990000"
    accent2 = "#b38600"
    user_bg = "rgba(248,250,252,0.98)"
    ai_bg = "rgba(153, 0, 0, 0.10)"
    input_bg = "rgba(255,255,255,0.98)"

st.markdown(
    f"""
<style>
.stApp {{
  background: {bg};
  color: {text};
}}
.block-container {{
  padding-top: 1.10rem;
  max-width: 980px;
}}
.top-banner {{
  background: {panel};
  border: 1px solid {border};
  border-radius: 18px;
  padding: 18px 18px;
  text-align: center;
}}
.hero-title {{
  margin-top: 8px;
  font-size: 1.70rem;
  font-weight: 900;
  letter-spacing: 0.2px;
}}
.hero-sub {{
  margin-top: 6px;
  font-size: 1.02rem;
  color: {mut};
}}
.stChatMessage {{
  padding: 1.05rem 1.10rem;
  border-radius: 18px;
  margin: 0.80rem 0;
  max-width: 88%;
  border: 1px solid {border};
  background: {panel};
}}
[data-testid="stChatMessage"][aria-label="user"] {{
  background: {user_bg};
  margin-left: auto;
}}
[data-testid="stChatMessage"][aria-label="AI"] {{
  background: {ai_bg};
  margin-right: auto;
}}
[data-testid="stChatMessage"] * {{
  color: {text} !important;
}}
.reasoning, .reasoning * {{
  color: {mut} !important;
  font-style: italic;
}}
[data-testid="stChatMessage"] a {{
  color: {accent2} !important;
}}
.stChatInput textarea {{
  background: {input_bg} !important;
  color: {text} !important;
  border-radius: 16px !important;
  border: 1px solid {border} !important;
  font-size: 1.08rem !important;
  line-height: 1.45 !important;
  min-height: 72px !important;
  padding: 14px 16px !important;
}}
.stChatInput textarea::placeholder {{
  color: {mut} !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Backend")
    st.write("**LLM:** Hugging Face Inference")
    st.write("**Embeddings:** SentenceTransformers + FAISS")
    st.divider()

    if HF_TOKEN:
        st.success("HF_TOKEN loaded (from secrets/env).")
    else:
        st.warning("HF_TOKEN not set. Add it to Streamlit Secrets or env var.")

    st.markdown("### Settings")
    if "model_config" not in st.session_state:
        st.session_state.model_config = {"temperature": 0.2, "max_tokens": 512}
    st.session_state.model_config["temperature"] = st.slider("Temperature", 0.0, 1.0, st.session_state.model_config["temperature"], 0.05)
    st.session_state.model_config["max_tokens"] = st.slider("Max tokens", 64, 1024, st.session_state.model_config["max_tokens"], 32)

# -----------------------------
# PDF -> chunks w/ page metadata
# -----------------------------
def clean_text(t: str) -> str:
    t = re.sub(r"\n\s*\n+", "\n", t)
    t = re.sub(r"Page\s+\d+\s*", "", t, flags=re.IGNORECASE)
    return t.strip()

def extract_pages_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            txt = page.get_text("text") or ""
            if not txt.strip():
                blocks = page.get_text("blocks") or []
                txt = "\n".join([b[4] for b in blocks if len(b) > 4 and isinstance(b[4], str)])
            txt = clean_text(txt)
            if txt.strip():
                pages.append((i, txt))
    return pages

def split_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks, buf = [], ""

    for ln in lines:
        if len(buf) + len(ln) + 1 <= max_chars:
            buf += ln + "\n"
        else:
            chunks.append(buf.strip())
            buf = (buf[-overlap:] if overlap and len(buf) > overlap else "") + ln + "\n"

    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c.strip()]

# -----------------------------
# Models (cached)
# -----------------------------
@st.cache_resource
def get_embedder() -> SentenceTransformer:
    # Try primary; if it fails, fallback.
    try:
        return SentenceTransformer(EMBED_MODEL_PRIMARY)
    except Exception:
        st.warning(f"Could not load {EMBED_MODEL_PRIMARY}. Falling back to {EMBED_MODEL_FALLBACK}.")
        return SentenceTransformer(EMBED_MODEL_FALLBACK)

@st.cache_resource
def get_llm_client(token: Optional[str]) -> InferenceClient:
    # Hugging Face InferenceClient supports chat_completion; use hf-inference provider.
    # Auth: pass token via api_key (HF user access token). :contentReference[oaicite:2]{index=2}
    if token:
        return InferenceClient(provider="hf-inference", api_key=token)
    return InferenceClient(provider="hf-inference")

embedder = get_embedder()
llm_client = get_llm_client(HF_TOKEN)

# -----------------------------
# Vector store build (cached in session)
# -----------------------------
def build_vector_store(pdf_dir: str):
    if faiss is None:
        raise RuntimeError("faiss-cpu is not available. Install faiss-cpu or switch to a numpy-only retriever.")

    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    all_chunks: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    for fn in sorted(files):
        path = os.path.join(pdf_dir, fn)
        pages = extract_pages_from_pdf(path)
        for (page_num, page_text) in pages:
            chunks = split_text(page_text)
            for ci, ch in enumerate(chunks):
                all_chunks.append(ch)
                all_meta.append({"file": fn, "page": page_num, "chunk": ci})

    if not all_chunks:
        return None, [], []

    embs = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    # Use cosine similarity: normalize and use inner product index
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index, all_chunks, all_meta

def retrieve(query: str, k: int = TOP_K):
    if st.session_state.get("vector_index") is None:
        return [], []

    index = st.session_state["vector_index"]
    chunks = st.session_state["chunks"]
    meta = st.session_state["meta"]

    q_emb = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(q_emb)

    scores, idxs = index.search(q_emb, k)
    idxs = idxs[0].tolist()

    chosen_chunks = []
    chosen_meta = []
    for i in idxs:
        if i == -1:
            continue
        chosen_chunks.append(chunks[i])
        chosen_meta.append(meta[i])
    return chosen_chunks, chosen_meta

def format_sources(meta: List[Dict[str, Any]]) -> str:
    if not meta:
        return ""
    # unique file+page pairs
    seen = []
    for m in meta:
        key = (m["file"], m["page"])
        if key not in seen:
            seen.append(key)
    return "\n".join([f"- {f} (p. {p})" for f, p in seen[:6]])

# -----------------------------
# Init index
# -----------------------------
if "vector_index" not in st.session_state:
    os.makedirs(PDF_DIR, exist_ok=True)
    try:
        idx, chunks, meta = build_vector_store(PDF_DIR)
        st.session_state["vector_index"] = idx
        st.session_state["chunks"] = chunks
        st.session_state["meta"] = meta
    except Exception as e:
        st.session_state["vector_index"] = None
        st.session_state["chunks"] = []
        st.session_state["meta"] = []
        st.error(f"Index build failed: {e}")

# Status line
pdf_count = len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]) if os.path.exists(PDF_DIR) else 0
chunk_count = len(st.session_state.get("chunks", []))
st.caption(f"● PDFs: {pdf_count} | Chunks: {chunk_count} | LLM: HF Inference")

# -----------------------------
# Chat state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything from the MKT 333 PDFs. 🍺🎮🤖"}
    ]

def call_llm(question: str, contexts: List[str], meta: List[Dict[str, Any]]) -> str:
    if not HF_TOKEN:
        return "HF_TOKEN is missing. Add it to Streamlit Secrets (Cloud) or set $env:HF_TOKEN locally."

    ctx_block = "\n\n---\n\n".join(contexts[:3]) if contexts else ""
    src_block = format_sources(meta)

    system_prompt = (
        "You are a course assistant. Answer ONLY using the provided context.\n"
        "If the answer is not in the context, say: \"I don’t have enough information in the PDFs to answer that.\".\n"
        "Always include citations in the form (File p.#).\n"
    )

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{ctx_block}\n\n"
        f"SOURCES:\n{src_block}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Use chat_completion API (OpenAI-like output). :contentReference[oaicite:3]{index=3}
    out = llm_client.chat_completion(
        messages=messages,
        model=LLM_MODEL,
        max_tokens=int(st.session_state.model_config["max_tokens"]),
        temperature=float(st.session_state.model_config["temperature"]),
    )
    return out.choices[0].message.content

# Render history
for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "user" else "AI"):
        st.markdown(m["content"])

# Input
prompt = st.chat_input("Type your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            contexts, meta = retrieve(prompt, TOP_K)
            answer = call_llm(prompt, contexts, meta)

            # Append sources block at the bottom (readable)
            if meta:
                answer += "\n\n**Sources**\n" + format_sources(meta)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
