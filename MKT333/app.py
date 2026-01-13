###########################################
# MKT 333 — Beer AI & Video Games (Streamlit)
# RAG (FAISS + SentenceTransformers) + HF Inference LLM
###########################################

from __future__ import annotations

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss

from sentence_transformers import SentenceTransformer

from huggingface_hub import InferenceClient, HfApi


# =============================
# Page config (MUST be first Streamlit call)
# =============================
st.set_page_config(
    page_title="MKT 333 — Beer AI & Video Games",
    page_icon="🍺",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =============================
# CONFIG
# =============================
PDF_FOLDER = "./knowledge_base"  # put your PDFs here for Streamlit Cloud too
PDF_CACHE_JSON = os.path.join(PDF_FOLDER, "pdf_data.json")

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"

# IMPORTANT: this must be a GENERATIVE model for HF Inference.
# If you use a non-chat model, this app will automatically fall back to text_generation().
DEFAULT_LLM_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

TOP_K = 5
CHUNK_MAX_CHARS = 1400
CHUNK_OVERLAP = 200


# =============================
# HF TOKEN (env/secrets) + validation
# =============================
def get_hf_token() -> Optional[str]:
    # Streamlit secrets (Cloud)
    try:
        tok = st.secrets.get("HF_TOKEN", None)
        if tok and str(tok).strip():
            return str(tok).strip()
    except Exception:
        pass

    # Local env vars
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return tok.strip() if tok and tok.strip() else None


def validate_token(token: str) -> bool:
    try:
        HfApi().whoami(token=token)
        return True
    except Exception:
        return False


HF_TOKEN = get_hf_token()
HF_TOKEN_VALID = bool(HF_TOKEN and validate_token(HF_TOKEN)) if HF_TOKEN else False

if HF_TOKEN and not HF_TOKEN_VALID:
    # Prevent “token poisoning” (invalid token causing public models to fail)
    st.warning("HF_TOKEN is set but invalid. Continuing without token.")
    try:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
    except Exception:
        pass
    HF_TOKEN = None


# =============================
# PDF + TEXT UTILS
# =============================
def clean_text(t: str) -> str:
    t = re.sub(r"\n\s*\n+", "\n", t)
    t = re.sub(r"Page\s+\d+", "", t, flags=re.IGNORECASE)
    return t.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            raw = page.get_text("text") or ""
            if not raw.strip():
                # fallback
                blocks = page.get_text("blocks") or []
                raw = "\n".join([b[4] for b in blocks if len(b) > 4 and isinstance(b[4], str)])
            text += raw + "\n"
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
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = tail + ln + "\n"

    if buf.strip():
        chunks.append(buf.strip())

    return [c for c in chunks if c.strip()]


def load_all_pdfs(folder_path: str) -> List[Dict[str, Any]]:
    """
    Loads PDFs and caches extracted text to pdf_data.json.
    Auto-refreshes cache if PDFs change.
    """
    os.makedirs(folder_path, exist_ok=True)
    current_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    # Cache read
    if os.path.exists(PDF_CACHE_JSON):
        try:
            with open(PDF_CACHE_JSON, "r", encoding="utf-8") as f:
                saved = json.load(f)

            saved_map = {x["filename"]: x for x in saved}
            current_set = set(current_files)
            saved_set = set(saved_map.keys())

            needs_refresh = current_set != saved_set
            if not needs_refresh:
                for fn in current_files:
                    p = os.path.join(folder_path, fn)
                    if saved_map[fn].get("last_modified", 0) < os.path.getmtime(p):
                        needs_refresh = True
                        break

            if not needs_refresh:
                return [{"filename": x["filename"], "text": x["text"]} for x in saved]
        except Exception:
            pass

    # Rebuild cache
    docs = []
    for fn in current_files:
        p = os.path.join(folder_path, fn)
        txt = extract_text_from_pdf(p)
        docs.append(
            {
                "filename": fn,
                "text": txt,
                "last_modified": os.path.getmtime(p),
            }
        )

    with open(PDF_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(docs, f)

    return [{"filename": d["filename"], "text": d["text"]} for d in docs]


# =============================
# EMBEDDINGS + FAISS (cosine)
# =============================
@st.cache_resource(show_spinner=True)
def get_embedder(hf_token: Optional[str]) -> SentenceTransformer:
    # sentence-transformers supports different kwarg names across versions;
    # try robustly.
    try:
        return SentenceTransformer(EMBED_MODEL_ID, token=hf_token) if hf_token else SentenceTransformer(EMBED_MODEL_ID)
    except TypeError:
        return SentenceTransformer(EMBED_MODEL_ID, use_auth_token=hf_token) if hf_token else SentenceTransformer(EMBED_MODEL_ID)


def build_vector_store(embedder: SentenceTransformer, docs: List[Dict[str, Any]]) -> Tuple[faiss.IndexFlatIP, List[str], List[Dict[str, Any]]]:
    all_chunks: List[str] = []
    meta: List[Dict[str, Any]] = []

    for doc in docs:
        chunks = split_text(doc["text"])
        all_chunks.extend(chunks)
        meta.extend([{"filename": doc["filename"], "chunk": i} for i in range(len(chunks))])

    if not all_chunks:
        # empty index
        idx = faiss.IndexFlatIP(384)  # placeholder dim; won't be used
        return idx, [], []

    emb = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)

    # cosine similarity via inner product on normalized vectors
    faiss.normalize_L2(emb)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, all_chunks, meta


def retrieve(embedder: SentenceTransformer, index: faiss.IndexFlatIP, chunks: List[str], meta: List[Dict[str, Any]], query: str, k: int = TOP_K):
    if not chunks:
        return [], []

    q = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q = np.asarray(q, dtype=np.float32)
    faiss.normalize_L2(q)

    scores, idxs = index.search(q, k)
    idxs = idxs[0].tolist()

    ctx = []
    ctx_meta = []
    for i in idxs:
        if i == -1:
            continue
        ctx.append(chunks[i])
        ctx_meta.append(meta[i])
    return ctx, ctx_meta


# =============================
# HF INFERENCE (chat -> fallback to text-generation)
# =============================
def make_llm_client(hf_token: Optional[str]) -> InferenceClient:
    if hf_token:
        return InferenceClient(token=hf_token)
    return InferenceClient()


def build_rag_prompt(question: str, contexts: List[str], ctx_meta: List[Dict[str, Any]]) -> str:
    # Add lightweight citations in the prompt so the model sticks to your PDFs.
    cited_blocks = []
    for c, m in zip(contexts, ctx_meta):
        cited_blocks.append(f"[{m['filename']} | chunk {m['chunk']}]\n{c}")

    ctx_text = "\n\n---\n\n".join(cited_blocks[:TOP_K])

    return (
        "You are a helpful course assistant. Answer ONLY using the provided PDF excerpts.\n"
        "If the answer is not contained in the excerpts, say: \"I don’t have enough information in the PDFs to answer that.\"\n\n"
        f"PDF EXCERPTS:\n{ctx_text}\n\n"
        f"QUESTION: {question}\n\n"
        "Return a clear answer and include citations like (Filename, chunk #) at the end of sentences."
    )


def call_hf_llm(
    client: InferenceClient,
    model_id: str,
    question: str,
    contexts: List[str],
    ctx_meta: List[Dict[str, Any]],
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 700,
) -> str:
    prompt = build_rag_prompt(question, contexts, ctx_meta)

    # First try chat_completion (works when model supports it / has chat template)
    try:
        resp = client.chat_completion(
            model=model_id,
            messages=[
                {"role": "system", "content": "You answer using only the given excerpts."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return resp.choices[0].message.content.strip()
    except ValueError:
        # Model does not support chat task → fallback to text_generation
        pass
    except Exception:
        # Any other HF inference issue → fallback to text_generation
        pass

    # Fallback: plain text generation endpoint
    out = client.text_generation(
        model=model_id,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        return_full_text=False,
    )
    return (out or "").strip() or "I don’t have enough information in the PDFs to answer that."


# =============================
# UI THEME (USC palette) + header
# =============================
if "ui_dark_mode" not in st.session_state:
    st.session_state.ui_dark_mode = True

USC_CARDINAL = "#990000"
USC_GOLD = "#FFCC00"
USC_BLACK = "#000000"
USC_WHITE = "#FFFFFF"
USC_GRAY_30 = "#CCCCCC"
USC_GRAY_70 = "#767676"

if st.session_state.ui_dark_mode:
    bg = USC_BLACK
    panel = "rgba(0, 0, 0, 0.70)"
    text = USC_WHITE
    mut = USC_GRAY_30
    border = "rgba(204, 204, 204, 0.18)"
    accent = USC_CARDINAL
    accent2 = USC_GOLD
    user_bg = "rgba(0, 0, 0, 0.78)"
    ai_bg = "rgba(153, 0, 0, 0.18)"
    input_bg = "rgba(0, 0, 0, 0.82)"
else:
    bg = USC_WHITE
    panel = "rgba(255, 255, 255, 0.96)"
    text = USC_BLACK
    mut = USC_GRAY_70
    border = "rgba(118, 118, 118, 0.22)"
    accent = USC_CARDINAL
    accent2 = USC_GOLD
    user_bg = "rgba(255, 255, 255, 0.98)"
    ai_bg = "rgba(153, 0, 0, 0.08)"
    input_bg = "rgba(255, 255, 255, 0.98)"

left, right = st.columns([0.80, 0.20], vertical_alignment="center")
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
    st.session_state.ui_dark_mode = st.toggle(
        "Dark mode" if st.session_state.ui_dark_mode else "Light mode",
        value=st.session_state.ui_dark_mode,
    )

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
.top-banner{{
  background: {panel};
  border: 1px solid {border};
  border-left: 8px solid {accent};
  border-bottom: 2px solid rgba(255,204,0,0.35);
  border-radius: 18px;
  padding: 16px 18px;
  text-align: center;
}}
.hero-title{{
  margin-top: 2px;
  font-size: 1.55rem;
  font-weight: 900;
}}
.hero-sub {{
  margin-top: 6px;
  font-size: 1.02rem;
  color: {mut};
}}
/* chat bubbles */
.stChatMessage {{
  padding: 1.05rem 1.10rem;
  border-radius: 18px;
  margin: 0.80rem 0;
  max-width: 88%;
  border: 1px solid {border};
  background: {panel};
}}
[data-testid="stChatMessage"][aria-label="AI"] {{
  background: {ai_bg};
  border: 1px solid rgba(153,0,0,0.35);
}}
[data-testid="stChatMessage"][aria-label="user"] {{
  background: {user_bg};
}}
[data-testid="stChatMessage"] * {{
  color: {text} !important;
}}
/* input */
.stChatInput textarea {{
  background: {input_bg} !important;
  color: {text} !important;
  border-radius: 16px !important;
  border: 1px solid {border} !important;
  font-size: 1.02rem !important;
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


# =============================
# SESSION STATE
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything from the MKT 333 PDFs. 🍺🎮🤖"}
    ]

if "model_config" not in st.session_state:
    st.session_state.model_config = {
        "llm_model_id": DEFAULT_LLM_MODEL_ID,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 700,
    }


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.subheader("Backend / Models")
    st.caption("HF token is optional for public models, but required for private/paid models.")

    st.write(f"**HF_TOKEN detected:** {'✅' if HF_TOKEN else '❌'}")
    st.write(f"**HF_TOKEN valid:** {'✅' if HF_TOKEN_VALID else '—'}")

    st.session_state.model_config["llm_model_id"] = st.text_input(
        "HF LLM Model ID",
        value=st.session_state.model_config["llm_model_id"],
        help="Must be a generative model. Example: HuggingFaceH4/zephyr-7b-beta",
    )

    st.session_state.model_config["temperature"] = st.slider(
        "Temperature", 0.0, 1.0, float(st.session_state.model_config["temperature"]), 0.05
    )
    st.session_state.model_config["top_p"] = st.slider(
        "Top-p", 0.1, 1.0, float(st.session_state.model_config["top_p"]), 0.05
    )
    st.session_state.model_config["max_tokens"] = st.slider(
        "Max tokens", 128, 1200, int(st.session_state.model_config["max_tokens"]), 50
    )

    st.divider()
    st.subheader("USC Links")
    st.markdown("- https://www.usc.edu")
    st.markdown("- https://www.marshall.usc.edu")


# =============================
# BUILD / LOAD VECTOR STORE
# =============================
@st.cache_resource(show_spinner=True)
def init_rag(hf_token: Optional[str]):
    embedder = get_embedder(hf_token)
    docs = load_all_pdfs(PDF_FOLDER)
    index, chunks, meta = build_vector_store(embedder, docs)
    return embedder, docs, index, chunks, meta


embedder, docs, faiss_index, all_chunks, all_meta = init_rag(HF_TOKEN)

st.caption(f"● PDFs: **{len(docs)}** | ● Chunks: **{len(all_chunks)}**")


# =============================
# CHAT RENDER
# =============================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# =============================
# ASK
# =============================
def answer_question(user_q: str) -> str:
    ctx, ctx_meta = retrieve(embedder, faiss_index, all_chunks, all_meta, user_q, TOP_K)

    if not ctx:
        return "I don’t have enough information in the PDFs to answer that. Try adding the syllabus PDF into ./pdfs."

    llm_client = make_llm_client(HF_TOKEN)
    return call_hf_llm(
        client=llm_client,
        model_id=st.session_state.model_config["llm_model_id"],
        question=user_q,
        contexts=ctx,
        ctx_meta=ctx_meta,
        temperature=float(st.session_state.model_config["temperature"]),
        top_p=float(st.session_state.model_config["top_p"]),
        max_tokens=int(st.session_state.model_config["max_tokens"]),
    )


# =============================
# INPUT
# =============================
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans = answer_question(prompt)
        st.markdown(ans)

    st.session_state.messages.append({"role": "assistant", "content": ans})
