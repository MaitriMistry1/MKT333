###########################################
# MKT 333 — Beer • AI • Video Games
# Streamlit RAG + Hugging Face Inference (Option 3)
###########################################

from __future__ import annotations

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from huggingface_hub import HfApi
from huggingface_hub import InferenceClient


# -----------------------------
# Config
# -----------------------------
PDF_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"  # embeddings
DEFAULT_LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # pick any public instruct model
DEFAULT_PROVIDER = "hf-inference"  # you can also try "auto" if available in your version

CHUNK_MAX_CHARS = 1400
CHUNK_OVERLAP_CHARS = 180
TOP_K = 5


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="MKT 333 — Beer • AI & Video Games",
    page_icon="🍺",
    layout="centered",
    initial_sidebar_state="expanded",
)


# -----------------------------
# HF Token handling
# -----------------------------
def get_hf_token() -> Optional[str]:
    # Streamlit Cloud secrets
    try:
        tok = st.secrets.get("HF_TOKEN", None)
        if tok:
            return str(tok).strip()
    except Exception:
        pass

    # Local env vars (PowerShell: $env:HF_TOKEN="...")
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return tok.strip() if tok else None


def validate_hf_token(token: str) -> bool:
    try:
        HfApi().whoami(token=token)
        return True
    except Exception:
        return False


HF_TOKEN = get_hf_token()
if HF_TOKEN and not validate_hf_token(HF_TOKEN):
    # If token is wrong, Hugging Face requests can fail even for public models.
    st.warning("HF_TOKEN looks invalid. Ignoring it and continuing without auth.")
    HF_TOKEN = None
    # Also remove env vars so downstream libraries don't keep using a bad token:
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)


# -----------------------------
# Text cleanup / chunking
# -----------------------------
def clean_text(t: str) -> str:
    t = re.sub(r"\n\s*\n+", "\n", t)
    t = re.sub(r"Page\s+\d+\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            t = page.get_text("text") or ""
            if not t.strip():
                # fallback
                blocks = page.get_text("blocks") or []
                t = "\n".join(
                    [b[4] for b in blocks if len(b) > 4 and isinstance(b[4], str)]
                )
            text += t + "\n"
    return clean_text(text)


def split_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks: List[str] = []
    buf = ""

    for ln in lines:
        if len(buf) + len(ln) + 1 <= max_chars:
            buf += ln + "\n"
        else:
            if buf.strip():
                chunks.append(buf.strip())
            # overlap
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = tail + ln + "\n"

    if buf.strip():
        chunks.append(buf.strip())

    return chunks


def pdf_signature() -> str:
    os.makedirs(PDF_DIR, exist_ok=True)
    files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    sig = []
    for fn in sorted(files):
        p = os.path.join(PDF_DIR, fn)
        try:
            sig.append((fn, os.path.getmtime(p), os.path.getsize(p)))
        except Exception:
            sig.append((fn, 0, 0))
    return json.dumps(sig)


# -----------------------------
# Embeddings model (cached)
# -----------------------------
@st.cache_resource
def get_embedder() -> SentenceTransformer:
    # IMPORTANT: don’t hard-fail if HF_TOKEN is missing; public model should load normally.
    return SentenceTransformer(EMBED_MODEL_ID)


# -----------------------------
# Index build/load (cached)
# -----------------------------
def ensure_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)


def build_index(embedder: SentenceTransformer) -> Tuple[Optional[faiss.IndexFlatIP], List[str], List[Dict[str, Any]]]:
    ensure_dirs()
    files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    all_chunks: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    for fn in files:
        path = os.path.join(PDF_DIR, fn)
        txt = extract_text_from_pdf(path)
        if not txt.strip():
            continue

        chunks = [c for c in split_text(txt) if c.strip()]
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            all_meta.append({"file": fn, "chunk": i})

    if not all_chunks:
        # save empty meta
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"sig": pdf_signature(), "chunks": [], "meta": []}, f)
        return None, [], []

    embs = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    # Use cosine similarity (normalize + inner product)
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)

    faiss.write_index(idx, FAISS_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"sig": pdf_signature(), "chunks": all_chunks, "meta": all_meta}, f)

    return idx, all_chunks, all_meta


@st.cache_resource
def load_or_build_index() -> Tuple[Optional[faiss.IndexFlatIP], List[str], List[Dict[str, Any]]]:
    ensure_dirs()
    embedder = get_embedder()

    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("sig") == pdf_signature():
                idx = faiss.read_index(FAISS_PATH)
                return idx, blob.get("chunks", []), blob.get("meta", [])
        except Exception:
            pass

    return build_index(embedder)


def retrieve(query: str, idx: Optional[faiss.IndexFlatIP], chunks: List[str], meta: List[Dict[str, Any]], k: int = TOP_K):
    if idx is None or not chunks:
        return [], []

    embedder = get_embedder()
    q = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(q)

    scores, ids = idx.search(q, k)
    ids = ids[0].tolist()

    picked_chunks = []
    picked_meta = []
    for i in ids:
        if i == -1:
            continue
        picked_chunks.append(chunks[i])
        picked_meta.append(meta[i])
    return picked_chunks, picked_meta


# -----------------------------
# HF LLM call (chat -> fallback to text_generation)
# -----------------------------
def build_prompt(question: str, contexts: List[str], meta: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], str]:
    # for chat_completion
    sys = (
        "You are a course assistant. Answer ONLY using the provided excerpts.\n"
        "If the answer is not in the excerpts, say: 'I don’t have enough information in the PDFs to answer that.'\n"
        "Cite sources at the end like: (FileName.pdf, chunk 3)."
    )

    ctx_lines = []
    for c, m in zip(contexts, meta):
        ctx_lines.append(f"[{m['file']}, chunk {m['chunk']}]\n{c}")

    ctx_block = "\n\n---\n\n".join(ctx_lines) if ctx_lines else "(no excerpts)"

    user = f"Question: {question}\n\nExcerpts:\n{ctx_block}\n\nAnswer:"

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

    # for text_generation fallback (single string)
    flat_prompt = f"{sys}\n\n{user}"
    return messages, flat_prompt


def call_hf_llm(
    model_id: str,
    provider: str,
    question: str,
    contexts: List[str],
    meta: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    client = InferenceClient(model=model_id, token=HF_TOKEN, provider=provider)

    messages, flat_prompt = build_prompt(question, contexts, meta)

    # 1) Try chat_completion (ONLY works if model/provider supports chat-completion)
    try:
        resp = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=float(temperature),
        )
        # huggingface_hub returns an object with choices[0].message.content
        return (resp.choices[0].message.content or "").strip()

    except ValueError:
        # Model doesn’t support chat-completion task -> fall back to text_generation
        # text_generation params documented here. :contentReference[oaicite:1]{index=1}
        out = client.text_generation(
            prompt=flat_prompt,
            max_new_tokens=max_tokens,
            temperature=float(temperature),
            do_sample=(float(temperature) > 0),
            return_full_text=False,
        )
        return (out or "").strip()


# -----------------------------
# UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything from the MKT 333 PDFs. 🍺🎮🤖"}]

if "ui_dark_mode" not in st.session_state:
    st.session_state.ui_dark_mode = True

if "model_config" not in st.session_state:
    st.session_state.model_config = {"temperature": 0.2, "max_tokens": 700}

# Header
left, right = st.columns([0.85, 0.15], vertical_alignment="center")
with left:
    st.markdown(
        """
        <div style="padding:14px 16px; border:1px solid rgba(231,234,240,0.12); border-radius:16px; text-align:center;">
          <div style="font-size:1.65rem; font-weight:900;">Beer • AI • Video Games</div>
          <div style="opacity:0.75; margin-top:4px;">Ask the course PDFs. Get clean, cited answers.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.session_state.ui_dark_mode = st.toggle("Dark", value=st.session_state.ui_dark_mode)

# Sidebar
with st.sidebar:
    st.markdown("### Backend")
    idx, chunks, meta = load_or_build_index()
    pdf_count = len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]) if os.path.exists(PDF_DIR) else 0
    st.write(f"**PDFs:** {pdf_count}")
    st.write(f"**Chunks:** {len(chunks)}")
    st.write(f"**Indexed:** {idx is not None}")

    st.divider()
    st.markdown("### Hugging Face LLM")
    model_id = st.text_input("Model ID", value=DEFAULT_LLM_MODEL_ID)
    provider = st.text_input("Provider", value=DEFAULT_PROVIDER)

    st.divider()
    st.markdown("### Settings")
    st.session_state.model_config["temperature"] = st.slider("Temperature", 0.0, 1.0, float(st.session_state.model_config["temperature"]), 0.05)
    st.session_state.model_config["max_tokens"] = st.slider("Max tokens", 128, 1200, int(st.session_state.model_config["max_tokens"]), 32)

    if st.button("🔁 Reindex PDFs"):
        # Clear cache + rebuild on next load
        load_or_build_index.clear()
        st.rerun()

# Chat history
for m in st.session_state.messages:
    with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
        st.markdown(m["content"])

# Input
prompt = st.chat_input("Type your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve + Answer
    idx, chunks, meta = load_or_build_index()
    ctx, ctx_meta = retrieve(prompt, idx, chunks, meta, TOP_K)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = call_hf_llm(
                model_id=model_id.strip(),
                provider=provider.strip(),
                question=prompt,
                contexts=ctx,
                meta=ctx_meta,
                temperature=float(st.session_state.model_config["temperature"]),
                max_tokens=int(st.session_state.model_config["max_tokens"]),
            )

            # (Optional) show sources at bottom
            if ctx_meta:
                cites = ", ".join([f"({m['file']}, chunk {m['chunk']})" for m in ctx_meta[:3]])
                answer = f"{answer}\n\n**Sources:** {cites}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

