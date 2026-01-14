###########################################
# MKT 333 — Course PDF Chatbot (Streamlit)
# Option 3: Hugging Face Inference (no Ollama)
###########################################

from __future__ import annotations

import os
import re
import json
from typing import Optional, List, Dict, Any, Tuple

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from huggingface_hub import HfApi, InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# -----------------------------
# Config
# -----------------------------
PDF_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")

# Embedding model (public; should work WITHOUT token)
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"

# HF chat model (you can override with env/secrets: HF_LLM_MODEL)
DEFAULT_HF_LLM_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP = 150
TOP_K = 5

# -----------------------------
# Helpers: env/secrets
# -----------------------------
def get_secret(name: str) -> Optional[str]:
    # Streamlit secrets first
    try:
        v = st.secrets.get(name, None)
        if v:
            return str(v).strip()
    except Exception:
        pass
    # Env vars next
    v = os.getenv(name)
    return v.strip() if v else None

def get_hf_token() -> Optional[str]:
    # Common env var names
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

# Read + validate token (IMPORTANT: if invalid, remove it so public downloads still work)
HF_TOKEN = get_hf_token()
if HF_TOKEN and not validate_hf_token(HF_TOKEN):
    # If your env has a stale token, huggingface libs may send it and fail.
    # Remove it so public model downloads work.
    HF_TOKEN = None
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
    os.environ.pop("HUGGINGFACE_API_TOKEN", None)

HF_LLM_MODEL = get_secret("HF_LLM_MODEL") or DEFAULT_HF_LLM_MODEL


# -----------------------------
# PDF + text utils
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
                t = "\n".join(
                    [b[4] for b in blocks if len(b) > 4 and isinstance(b[4], str)]
                )
            text += t + "\n"
    return clean_text(text)

def split_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # stable newline-based chunking
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

    # final safety
    return [c for c in chunks if c.strip()]

def pdf_signature() -> str:
    if not os.path.exists(PDF_DIR):
        return "[]"
    files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    sig = []
    for f in files:
        p = os.path.join(PDF_DIR, f)
        try:
            sig.append((f, os.path.getmtime(p), os.path.getsize(p)))
        except Exception:
            sig.append((f, 0, 0))
    return json.dumps(sig)


# -----------------------------
# Cached embedder + index build
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    # Do NOT pass token here; public model should load without it.
    return SentenceTransformer(EMBED_MODEL_ID)

def build_faiss_index(sig: str) -> Tuple[Optional[faiss.IndexFlatIP], List[str], List[Dict[str, Any]]]:
    """
    Build a cosine-similarity FAISS index (IndexFlatIP on normalized vectors).
    Cache is controlled by `sig` so it rebuilds when PDFs change.
    """
    os.makedirs(PDF_DIR, exist_ok=True)
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

    # embeddings (normalized)
    emb = embedder.encode(
        all_chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, all_chunks, meta

@st.cache_resource(show_spinner=False)
def get_index_bundle(sig: str):
    # cached across reruns; rebuild only when sig changes
    return build_faiss_index(sig)

def retrieve(query: str, index: Optional[faiss.IndexFlatIP], chunks: List[str], meta: List[Dict[str, Any]], k: int = TOP_K):
    if index is None or not chunks:
        return [], []

    embedder = get_embedder()
    q = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q, k)
    idxs = idxs[0].tolist()

    ctx = []
    m = []
    for i in idxs:
        if i == -1:
            continue
        ctx.append(chunks[i])
        m.append(meta[i])
    return ctx, m


# -----------------------------
# HF LLM call (Chat completion + fallback)
# -----------------------------
def call_hf_llm(
    question: str,
    contexts: List[str],
    citations: List[Dict[str, Any]],
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> str:
    # Label sources
    labeled = []
    for i, (c, m) in enumerate(zip(contexts, citations), start=1):
        labeled.append(f"[S{i}] ({m['file']} • chunk {m['chunk']})\n{c}")
    context_block = "\n\n".join(labeled) if labeled else "(no context)"

    system = (
        "You are a course assistant.\n"
        "Answer ONLY using the provided sources.\n"
        "If the answer is not in the sources, say exactly:\n"
        "\"I don’t have enough information in the PDFs to answer that.\"\n"
        "When you use a source, cite it like [S1], [S2].\n"
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Sources:\n{context_block}\n\n"
        "Write a clear, helpful answer. Use bullets if helpful.\n"
    )

    # Important: create client with token only, pass model_id per call
    client = InferenceClient(token=HF_TOKEN)  # HF_TOKEN can be None

    # 1) Try chat endpoint
    try:
        resp = client.chat_completion(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return resp.choices[0].message["content"].strip()

    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        st.error(
            f"HF Inference error (chat): status={status}. "
            f"If this is 401/403/404, your model likely isn’t available for chat on free HF Inference."
        )

        # 2) Fallback to text_generation for models that don’t support chat_completion
        try:
            prompt = f"{system}\n\n{user}\nAnswer:\n"
            out = client.text_generation(
                model=model_id,
                prompt=prompt,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
            )
            return str(out).strip()
        except HfHubHTTPError as e2:
            status2 = getattr(getattr(e2, "response", None), "status_code", None)
            st.error(
                f"HF Inference error (text): status={status2}. "
                f"Try setting HF_LLM_MODEL to 'HuggingFaceH4/zephyr-7b-beta' in Streamlit secrets."
            )
            # Final fallback: return top excerpts so the app still responds
            return (
                "I couldn’t call the hosted LLM right now, but here are the most relevant excerpts:\n\n"
                + "\n\n---\n\n".join(contexts[:3])
            )

    except Exception:
        st.error("Unexpected LLM error. Check logs.")
        raise

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="MKT 333 — Beer AI & Video Games",
    page_icon="🍺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Theme state
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

# Theme colors
if st.session_state.ui_dark_mode:
    bg = "#0b0d12"
    panel = "rgba(15, 18, 28, 0.86)"
    text = "#e7eaf0"
    mut = "#a7b0c0"
    border = "rgba(231,234,240,0.12)"
    accent2 = "#ffcc00"
    user_bg = "rgba(30, 34, 46, 0.92)"
    ai_bg = "rgba(153, 0, 0, 0.22)"
    input_bg = "rgba(12, 14, 22, 0.85)"
else:
    bg = "#fafafa"
    panel = "rgba(255,255,255,0.92)"
    text = "#0b1220"
    mut = "#4b5563"
    border = "rgba(11,18,32,0.10)"
    accent2 = "#b38600"
    user_bg = "rgba(248,250,252,0.98)"
    ai_bg = "rgba(153, 0, 0, 0.10)"
    input_bg = "rgba(255,255,255,0.98)"

st.markdown(
    f"""
<style>
.stApp {{ background: {bg}; color: {text}; }}
.block-container {{ padding-top: 1.10rem; max-width: 980px; }}

.top-banner {{
  background: {panel};
  border: 1px solid {border};
  border-radius: 18px;
  padding: 18px 18px;
  text-align: center;
}}
.hero-title {{ margin-top: 8px; font-size: 1.70rem; font-weight: 900; }}
.hero-sub {{ margin-top: 6px; font-size: 1.02rem; color: {mut}; }}

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
[data-testid="stChatMessage"][aria-label="assistant"] {{
  background: {ai_bg};
  margin-right: auto;
}}
[data-testid="stChatMessage"] * {{
  color: {text} !important;
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
.stChatInput textarea::placeholder {{ color: {mut} !important; }}

.sidebar-card {{
  background: rgba(15, 18, 28, 0.86);
  border: 1px solid rgba(231,234,240,0.12);
  border-radius: 18px;
  padding: 16px;
}}
.sidebar-title {{ font-weight: 900; font-size: 1.05rem; margin: 0; }}
.sidebar-sub {{ margin-top: 8px; color: rgba(167,176,192,1); font-size: 0.95rem; }}
.sidebar-badge {{
  display: inline-block;
  margin-left: 10px;
  padding: 3px 10px;
  border-radius: 999px;
  background: rgba(255,204,0,0.12);
  border: 1px solid rgba(255,204,0,0.22);
  color: rgba(231,234,240,1);
  font-size: 0.78rem;
  font-weight: 800;
}}
.sidebar-links a {{
  display: block;
  text-decoration: none;
  margin-top: 12px;
  padding: 16px 14px;
  border-radius: 14px;
  border: 1px solid rgba(231,234,240,0.10);
  background: rgba(12, 14, 22, 0.75);
  color: rgba(231,234,240,1) !important;
  font-weight: 700;
}}
.sidebar-links a:hover {{
  border-color: rgba(255,204,0,0.35);
  box-shadow: 0 0 0 2px rgba(255,204,0,0.08);
}}
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
          <div class="sidebar-title">USC Links <span class="sidebar-badge">Quick</span></div>
          <div class="sidebar-sub">Open official pages in a new tab.</div>
          <div class="sidebar-links">
            <a href="https://www.usc.edu" target="_blank">USC — University of Southern California</a>
            <a href="https://gould.usc.edu/faculty/profile/d-daniel-sokol/" target="_blank">Professor D. Sokol</a>
            <a href="https://www.marshall.usc.edu" target="_blank">USC Marshall School of Business</a>
            <a href="https://www.marshall.usc.edu/departments/marketing" target="_blank">Marshall Marketing Department</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("HF Settings")
    st.caption("Set HF_TOKEN in Streamlit secrets or PowerShell env. Optionally set HF_LLM_MODEL.")
    st.text_input("HF model id", value=HF_LLM_MODEL, key="hf_model_id", help="Use a public HF Inference model like HuggingFaceH4/zephyr-7b-beta")

    st.subheader("Generation")
    st.session_state.setdefault("temperature", 0.2)
    st.session_state.setdefault("max_tokens", 512)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, float(st.session_state.temperature), 0.05)
    st.session_state.max_tokens = st.slider("Max tokens", 64, 1024, int(st.session_state.max_tokens), 64)

# Session init
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! Ask me anything from the MKT 333 PDFs. 🍺🎮🤖"
    }]

# Build / load index (cache keyed by signature)
sig = pdf_signature()
with st.spinner("Loading PDFs + building index (first run can take a bit)..."):
    index, chunks, meta = get_index_bundle(sig)

pdf_count = len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]) if os.path.exists(PDF_DIR) else 0
chunk_count = len(chunks)

st.caption(f"● Backend: {'OK' if index is not None else 'NOT READY'} ({pdf_count} PDFs, {chunk_count} chunks)")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve
    contexts, cites = retrieve(prompt, index, chunks, meta, TOP_K)

    with st.chat_message("assistant"):
        if not contexts:
            st.markdown("I don’t have enough information in the PDFs to answer that. Try adding the syllabus PDF into `/pdfs` and refresh.")
        else:
            try:
                answer = call_hf_llm(
                    question=prompt,
                    contexts=contexts,
                    citations=cites,
                    model_id=st.session_state.get("hf_model_id", HF_LLM_MODEL),
                    temperature=float(st.session_state.temperature),
                    max_tokens=int(st.session_state.max_tokens),
                )
                st.markdown(answer)

                # Show sources list (simple + useful)
                with st.expander("Sources used"):
                    for i, m in enumerate(cites, start=1):
                        st.write(f"[S{i}] {m['file']} — chunk {m['chunk']}")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception:
                st.markdown("LLM call failed. Open Streamlit logs to see the exact status code/message.")


