###########################################
# MKT 333 — Course PDF Chatbot (Streamlit)
# Hugging Face Inference (Option 3)
# - Fixes StopIteration by pinning provider="hf-inference"
# - Uses text_generation as primary (more compatible on free tier)
# - Disk caches FAISS index so it doesn't rebuild every prompt
###########################################

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from typing import Optional, List, Dict, Any, Tuple

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from huggingface_hub import HfApi, InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# -----------------------------
# Paths / Config
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
PDF_DIR = os.path.join(BASE_DIR, "knowledge_base")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")

# For Streamlit Cloud stability, MiniLM is faster/smaller than BGE.
# You can override with env/secrets: EMBED_MODEL_ID="BAAI/bge-small-en-v1.5"
DEFAULT_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Default HF model (override with env/secrets HF_LLM_MODEL)
DEFAULT_HF_LLM_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Helpers: secrets / env
# -----------------------------
def get_secret(name: str) -> Optional[str]:
    try:
        v = st.secrets.get(name, None)
        if v:
            return str(v).strip()
    except Exception:
        pass
    v = os.getenv(name)
    return v.strip() if v else None


def get_hf_token() -> Optional[str]:
    return (
        get_secret("HF_TOKEN")
        or get_secret("HUGGINGFACEHUB_API_TOKEN")
        or get_secret("HUGGINGFACE_API_TOKEN")
    )


def validate_hf_token(token: str) -> bool:
    try:
        HfApi().whoami(token=token)
        return True
    except Exception:
        return False


HF_TOKEN = get_hf_token()
if HF_TOKEN and not validate_hf_token(HF_TOKEN):
    # If env has a stale token, HF libs may send it and fail even for public models.
    HF_TOKEN = None
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
    os.environ.pop("HUGGINGFACE_API_TOKEN", None)

HF_LLM_MODEL = get_secret("HF_LLM_MODEL") or DEFAULT_HF_LLM_MODEL
EMBED_MODEL_ID = get_secret("EMBED_MODEL_ID") or DEFAULT_EMBED_MODEL_ID


# -----------------------------
# PDF text utils
# -----------------------------
def clean_text(t: str) -> str:
    t = re.sub(r"\n\s*\n+", "\n", t)
    t = re.sub(r"Page\s+\d+", "", t, flags=re.IGNORECASE)
    return t.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            t = page.get_text("text") or ""
            if not t.strip():
                blocks = page.get_text("blocks") or []
                t = "\n".join([b[4] for b in blocks if len(b) > 4 and isinstance(b[4], str)])
            text += t + "\n"
    return clean_text(text)


def split_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks: List[str] = []
    buf = ""

    for ln in lines:
        if len(buf) + len(ln) + 1 <= max_chars:
            buf += ln + "\n"
        else:
            chunks.append(buf.strip())
            keep = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = keep + ln + "\n"

    if buf.strip():
        chunks.append(buf.strip())

    return [c for c in chunks if c.strip()]


def pdf_signature() -> str:
    """
    Use filename + size (avoid mtimes to reduce unnecessary rebuilds on Cloud).
    """
    if not os.path.exists(PDF_DIR):
        return "[]"
    files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    sig = []
    for f in files:
        p = os.path.join(PDF_DIR, f)
        try:
            sig.append((f, os.path.getsize(p)))
        except Exception:
            sig.append((f, 0))
    return json.dumps(sig)


def sig_hash(sig: str) -> str:
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


# -----------------------------
# Embedder (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_ID)


# -----------------------------
# FAISS build + disk cache
# -----------------------------
def cache_paths(sig: str) -> Tuple[str, str, str]:
    h = sig_hash(sig)
    os.makedirs(CACHE_DIR, exist_ok=True)
    index_path = os.path.join(CACHE_DIR, f"faiss_{h}.index")
    chunks_path = os.path.join(CACHE_DIR, f"chunks_{h}.json")
    meta_path = os.path.join(CACHE_DIR, f"meta_{h}.json")
    return index_path, chunks_path, meta_path


def try_load_cached_index(sig: str) -> Tuple[Optional[faiss.IndexFlatIP], List[str], List[Dict[str, Any]]]:
    index_path, chunks_path, meta_path = cache_paths(sig)
    if not (os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(meta_path)):
        return None, [], []
    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, chunks, meta
    except Exception:
        return None, [], []


def save_cached_index(sig: str, index: faiss.IndexFlatIP, chunks: List[str], meta: List[Dict[str, Any]]) -> None:
    index_path, chunks_path, meta_path = cache_paths(sig)
    faiss.write_index(index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def build_faiss_index(sig: str) -> Tuple[Optional[faiss.IndexFlatIP], List[str], List[Dict[str, Any]]]:
    os.makedirs(PDF_DIR, exist_ok=True)

    # 1) disk cache
    cached_index, cached_chunks, cached_meta = try_load_cached_index(sig)
    if cached_index is not None and cached_chunks and cached_meta:
        return cached_index, cached_chunks, cached_meta

    # 2) build fresh
    files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    embedder = get_embedder()

    all_chunks: List[str] = []
    meta: List[Dict[str, Any]] = []

    for fn in files:
        path = os.path.join(PDF_DIR, fn)
        txt = extract_text_from_pdf(path)
        if not txt.strip():
            continue
        chunks = split_text(txt)
        all_chunks.extend(chunks)
        meta.extend([{"file": fn, "chunk": i} for i in range(len(chunks))])

    if not all_chunks:
        return None, [], []

    emb = embedder.encode(
        all_chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    save_cached_index(sig, index, all_chunks, meta)
    return index, all_chunks, meta


def retrieve(
    query: str,
    index: Optional[faiss.IndexFlatIP],
    chunks: List[str],
    meta: List[Dict[str, Any]],
    k: int = TOP_K,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if index is None or not chunks:
        return [], []

    embedder = get_embedder()
    q = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q, k)
    idxs = idxs[0].tolist()

    ctx, m = [], []
    for i in idxs:
        if i == -1:
            continue
        ctx.append(chunks[i])
        m.append(meta[i])
    return ctx, m


# -----------------------------
# HF LLM call (provider pinned)
# -----------------------------
def call_hf_llm_text_generation(
    question: str,
    contexts: List[str],
    citations: List[Dict[str, Any]],
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> str:
    labeled = []
    for i, (c, m) in enumerate(zip(contexts, citations), start=1):
        labeled.append(f"[S{i}] ({m['file']} • chunk {m['chunk']})\n{c}")
    context_block = "\n\n".join(labeled) if labeled else "(no context)"

    prompt = (
        "You are a course assistant.\n"
        "Answer ONLY using the provided sources.\n"
        "If the answer is not in the sources, say exactly:\n"
        "\"I don’t have enough information in the PDFs to answer that.\"\n"
        "When you use a source, cite it like [S1], [S2].\n\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n{context_block}\n\n"
        "Answer:\n"
    )

    # ✅ Pin provider to avoid StopIteration and auto-routing issues
    client = InferenceClient(provider="hf-inference", token=HF_TOKEN, timeout=60)

    # Primary: text_generation (most compatible on free tier)
    out = client.text_generation(
        model=model_id,
        prompt=prompt,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        do_sample=True,
        return_full_text=False,
    )
    return str(out).strip()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="MKT 333 — Beer AI & Video Games",
    page_icon="🍺",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <div style="text-align:center; padding: 8px 0 6px 0;">
      <div style="font-size:1.75rem; font-weight:900;">Beer • AI • Video Games</div>
      <div style="opacity:0.75;">Ask the course PDFs. Get clean, cited answers.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Model")
    st.caption("Free HF serverless works best with small instruct models.")
    presets = [
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "HuggingFaceH4/zephyr-7b-beta",
    ]
    default_idx = 0 if HF_LLM_MODEL not in presets else presets.index(HF_LLM_MODEL)
    model_id = st.selectbox("HF model id", presets, index=default_idx)
    model_id = st.text_input("Or type a model id", value=model_id)

    st.subheader("Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 768, 256, 64)

    st.subheader("Index")
    if st.button("Rebuild index now"):
        # Clear session bundle and rerun (disk cache is signature-based; rebuild if PDFs changed)
        st.session_state.pop("index_bundle", None)
        st.rerun()

    st.divider()
    st.caption("Token status (not shown): " + ("✅ present" if HF_TOKEN else "ℹ️ none"))

# Session init
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything from the MKT 333 PDFs. 🍺🎮🤖"}]

# Load/build index ONCE per session (fast afterwards due to disk cache)
if "index_bundle" not in st.session_state:
    sig = pdf_signature()
    with st.spinner("Loading PDFs + building index (first run can take a bit)..."):
        t0 = time.time()
        index, chunks, meta = build_faiss_index(sig)
        st.session_state.index_bundle = (sig, index, chunks, meta, time.time() - t0)

sig, index, chunks, meta, build_seconds = st.session_state.index_bundle

pdf_count = len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]) if os.path.exists(PDF_DIR) else 0
chunk_count = len(chunks) if chunks else 0
st.caption(f"● Backend: {'OK' if index is not None else 'NOT READY'} — {pdf_count} PDFs, {chunk_count} chunks (index load/build: {build_seconds:.1f}s)")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    contexts, cites = retrieve(prompt, index, chunks, meta, TOP_K)

    with st.chat_message("assistant"):
        if not contexts:
            st.markdown("I don’t have enough information in the PDFs to answer that. Add PDFs to `knowledge_base/` and reload.")
        else:
            try:
                answer = call_hf_llm_text_generation(
                    question=prompt,
                    contexts=contexts,
                    citations=cites,
                    model_id=model_id.strip(),
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                st.markdown(answer)

                with st.expander("Sources used"):
                    for i, m in enumerate(cites, start=1):
                        st.write(f"[S{i}] {m['file']} — chunk {m['chunk']}")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except HfHubHTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                st.error(
                    f"HF Inference error: status={status}. "
                    f"If 403/404, that model likely isn’t available on free hf-inference. Try another preset."
                )
                st.markdown(
                    "Meanwhile, here are the most relevant excerpts:\n\n"
                    + "\n\n---\n\n".join(contexts[:3])
                )
            except Exception:
                st.error("Unexpected LLM error. Check Streamlit logs.")
                st.markdown(
                    "Meanwhile, here are the most relevant excerpts:\n\n"
                    + "\n\n---\n\n".join(contexts[:3])
                )
