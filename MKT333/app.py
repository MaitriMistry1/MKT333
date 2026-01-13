###########################################
# MKT 333 AI Assistant (Streamlit + RAG + HF Inference)
###########################################

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from huggingface_hub import HfApi, InferenceClient
from huggingface_hub.utils import HfHubHTTPError

# -----------------------------
# Config
# -----------------------------
PDF_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")

# Embeddings
EMBED_MODEL_PRIMARY = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP = 150
TOP_K = 5

# HF LLM (change in sidebar or env var HF_CHAT_MODEL)
DEFAULT_CHAT_MODEL = os.getenv("HF_CHAT_MODEL") or "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_PROVIDER = os.getenv("HF_PROVIDER") or "hf-inference"  # or "auto"

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="MKT 333 — Beer AI & Video Games",
    page_icon="🍺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Token helpers
# -----------------------------
def get_hf_token() -> Optional[str]:
    # Streamlit Cloud secrets first
    try:
        tok = st.secrets.get("HF_TOKEN", None)
        if tok:
            return str(tok).strip()
    except Exception:
        pass

    # Local env vars next
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return tok.strip() if tok else None

def validate_token(token: str) -> bool:
    try:
        _ = HfApi().whoami(token=token)
        return True
    except Exception:
        return False

HF_TOKEN = get_hf_token()
if not HF_TOKEN:
    st.error("HF_TOKEN not found. Set it in PowerShell ($env:HF_TOKEN='...') or Streamlit Secrets.")
    st.stop()

if not validate_token(HF_TOKEN):
    st.error(
        "Your HF_TOKEN is invalid. Generate a new token on Hugging Face and set it again.\n\n"
        "PowerShell:  $env:HF_TOKEN='hf_...'\n"
        "Persistent:  setx HF_TOKEN 'hf_...'\n\n"
        "Then restart the terminal and re-run Streamlit."
    )
    st.stop()

# -----------------------------
# UI theme toggle (keep simple)
# -----------------------------
if "ui_dark_mode" not in st.session_state:
    st.session_state.ui_dark_mode = True

left, right = st.columns([0.80, 0.20], vertical_alignment="center")
with left:
    st.markdown("## Beer • AI • Video Games")
    st.caption("Ask the course PDFs. Get clean, cited answers.")
with right:
    st.session_state.ui_dark_mode = st.toggle("Dark mode", value=st.session_state.ui_dark_mode)

if st.session_state.ui_dark_mode:
    bg = "#0b0d12"
    text = "#e7eaf0"
    mut = "#a7b0c0"
    border = "rgba(231,234,240,0.12)"
    panel = "rgba(15, 18, 28, 0.86)"
else:
    bg = "#fafafa"
    text = "#0b1220"
    mut = "#4b5563"
    border = "rgba(11,18,32,0.10)"
    panel = "rgba(255,255,255,0.92)"

st.markdown(
    f"""
<style>
.stApp {{ background: {bg}; color: {text}; }}
.block-container {{ max-width: 980px; padding-top: 1rem; }}
[data-testid="stChatMessage"] {{ border: 1px solid {border}; background: {panel}; border-radius: 16px; }}
[data-testid="stChatMessage"] * {{ color: {text} !important; }}
.stChatInput textarea {{
  border: 1px solid {border} !important;
  border-radius: 16px !important;
  min-height: 72px !important;
  font-size: 1.05rem !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# PDF + chunking
# -----------------------------
def clean_text(t: str) -> str:
    t = re.sub(r"\n\s*\n+", "\n", t)
    t = re.sub(r"Page\s+\d+", "", t, flags=re.IGNORECASE)
    return t.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    out = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            txt = page.get_text("text") or ""
            if not txt.strip():
                blocks = page.get_text("blocks") or []
                txt = "\n".join([b[4] for b in blocks if len(b) > 4 and isinstance(b[4], str)])
            out.append(txt)
    return clean_text("\n".join(out))

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

def pdf_signature(folder: str) -> str:
    if not os.path.exists(folder):
        return "missing"
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")])
    sig = []
    for f in files:
        p = os.path.join(folder, f)
        sig.append((f, os.path.getmtime(p), os.path.getsize(p)))
    return hashlib.sha256(json.dumps(sig).encode("utf-8")).hexdigest()

# -----------------------------
# Embeddings + FAISS (cosine)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    # If primary fails for any reason, fallback to a smaller public model
    try:
        return SentenceTransformer(EMBED_MODEL_PRIMARY)
    except Exception:
        return SentenceTransformer(EMBED_MODEL_FALLBACK)

def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

@st.cache_resource(show_spinner=True)
def build_index(sig: str) -> Tuple[Optional[faiss.Index], List[str], List[Dict[str, Any]]]:
    os.makedirs(PDF_DIR, exist_ok=True)
    pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])

    embedder = load_embedder()

    all_chunks: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    for fn in pdfs:
        path = os.path.join(PDF_DIR, fn)
        txt = extract_text_from_pdf(path)
        if not txt.strip():
            continue

        chunks = split_text(txt)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            all_meta.append({"file": fn, "chunk": i})

    if not all_chunks:
        return None, [], []

    embs = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    embs = normalize_rows(embs)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(embs)

    return index, all_chunks, all_meta

def retrieve(query: str, index: faiss.Index, chunks: List[str], meta: List[Dict[str, Any]], k: int = TOP_K):
    embedder = load_embedder()
    q = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    q = normalize_rows(q)
    scores, ids = index.search(q, k)
    ids = ids[0].tolist()
    out_chunks, out_meta = [], []
    for idx in ids:
        if idx == -1:
            continue
        out_chunks.append(chunks[idx])
        out_meta.append(meta[idx])
    return out_chunks, out_meta

# -----------------------------
# HF LLM call
# -----------------------------
def call_hf_chat(model_id: str, provider: str, question: str, contexts: List[str], meta: List[Dict[str, Any]]) -> str:
    client = InferenceClient(model=model_id, provider=provider, token=HF_TOKEN)  # model/provider/token supported :contentReference[oaicite:1]{index=1}

    # Build a citations-ready context block
    ctx_blocks = []
    for i, (c, m) in enumerate(zip(contexts, meta), start=1):
        tag = f"SOURCE {i} [{m['file']} | chunk {m['chunk']}]"
        ctx_blocks.append(f"{tag}\n{c}")

    ctx_text = "\n\n---\n\n".join(ctx_blocks) if ctx_blocks else "NO_CONTEXT"

    system = (
        "You answer ONLY using the provided SOURCES.\n"
        "If the answer isn't in the sources, say: \"I don’t have enough information in the PDFs to answer that.\" \n"
        "When you use a source, cite it like: [filename | chunk N].\n"
        "Keep the final answer clean and concise."
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"SOURCES:\n{ctx_text}\n\n"
        "Write the answer with citations."
    )

    try:
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        # huggingface_hub returns OpenAI-like structure
        return resp.choices[0].message.content.strip()
    except HfHubHTTPError as e:
        # This typically means: model not supported by provider, gated model, rate limit, etc.
        raise e

# -----------------------------
# Sidebar status + controls
# -----------------------------
with st.sidebar:
    st.markdown("### Status")
    os.makedirs(PDF_DIR, exist_ok=True)
    pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    st.write(f"PDFs found: **{len(pdfs)}**")
    st.caption(f"PDF folder: {PDF_DIR}")

    st.markdown("### HF Model")
    model_id = st.text_input("HF chat model id", value=DEFAULT_CHAT_MODEL)
    provider = st.text_input("Provider", value=DEFAULT_PROVIDER)
    st.caption("Tip: if a model is gated or unsupported, switch to a different instruct model.")

    if st.button("Rebuild index (re-read PDFs)"):
        # bump signature by clearing cache resource tied to signature
        st.cache_resource.clear()
        st.rerun()

# -----------------------------
# Build/load index
# -----------------------------
sig = pdf_signature(PDF_DIR)
index, chunks, meta = build_index(sig)

# show live count
with st.sidebar:
    st.write(f"Chunks indexed: **{len(chunks)}**")
    st.write(f"Indexed: **{index is not None}**")

if index is None or not chunks:
    st.warning("No chunks indexed. Put PDFs inside the ./pdfs folder and click 'Rebuild index'.")
    st.stop()

# -----------------------------
# Chat state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything from the MKT 333 PDFs. 🍺🎮🤖"}
    ]

for m in st.session_state.messages:
    with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
        st.markdown(m["content"])

# -----------------------------
# Chat input
# -----------------------------
prompt = st.chat_input("Ask something from the course PDFs…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve
    ctx, ctx_meta = retrieve(prompt, index, chunks, meta, TOP_K)

    # Answer
    with st.chat_message("assistant"):
        try:
            answer = call_hf_chat(model_id=model_id, provider=provider, question=prompt, contexts=ctx, meta=ctx_meta)
            st.markdown(answer)

            # show sources list
            if ctx_meta:
                with st.expander("Sources used"):
                    for m in ctx_meta:
                        st.write(f"- {m['file']} (chunk {m['chunk']})")

        except HfHubHTTPError:
            st.error(
                "HF Inference failed for this model/provider.\n\n"
                "Most common causes:\n"
                "• model is gated / requires acceptance\n"
                "• provider doesn’t support this model for chat\n"
                "• rate limit (429)\n\n"
                "Try a different model id (e.g., another *Instruct* model) or set provider='auto'."
            )
            answer = "I couldn’t reach the Hugging Face chat endpoint for that model/provider."

    st.session_state.messages.append({"role": "assistant", "content": answer})

