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
HERE = os.path.dirname(__file__)
PDF_DIR = os.path.join(HERE, "knowledge_base")  # put your PDFs here in the repo

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"  # embeddings (public)
DEFAULT_HF_LLM_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # smaller tends to work more often

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

# If your env accidentally has an old/invalid token, it can break requests.
# We drop it so public models still download.
if HF_TOKEN and not validate_hf_token(HF_TOKEN):
    HF_TOKEN = None
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
    os.environ.pop("HUGGINGFACE_API_TOKEN", None)

HF_LLM_MODEL = get_secret("HF_LLM_MODEL") or DEFAULT_HF_LLM_MODEL


# -----------------------------
# PDF + chunking
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
# Cached embedder + cached index bundle
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_ID)


@st.cache_resource(show_spinner=False)
def get_index_bundle(sig: str) -> Tuple[Optional[faiss.IndexFlatIP], List[str], List[Dict[str, Any]]]:
    """
    Build cosine-similarity FAISS index once per PDF signature.
    Cached across reruns, so your UI won't keep "reloading PDFs" every chat.
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


def retrieve(query: str, index: Optional[faiss.IndexFlatIP], chunks: List[str], meta: List[Dict[str, Any]], k: int = TOP_K):
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
# HF LLM calls
# -----------------------------
def _extract_chat_content(resp: Any) -> str:
    """
    Support both object-style and dict-style responses.
    """
    try:
        # object style: resp.choices[0].message.content
        choice0 = resp.choices[0]
        msg = getattr(choice0, "message", None)
        content = getattr(msg, "content", None)
        if content:
            return str(content).strip()
    except Exception:
        pass

    try:
        # dict style: resp["choices"][0]["message"]["content"]
        return str(resp["choices"][0]["message"]["content"]).strip()
    except Exception:
        return str(resp).strip()


def probe_hf_model(model_id: str) -> Tuple[bool, str]:
    """
    Quick smoke test to see if the model works on HF Inference.
    """
    client = InferenceClient(token=HF_TOKEN)  # token can be None
    try:
        resp = client.chat_completion(
            model=model_id,
            messages=[{"role": "user", "content": "Say OK in one word."}],
            max_tokens=10,
            temperature=0.0,
        )
        out = _extract_chat_content(resp)
        return True, f"✅ Chat OK: {out[:80]}"
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        return False, f"❌ HF error status={status} (common: 401/403/404/429/503)"
    except Exception as e:
        return False, f"❌ Unexpected error: {type(e).__name__}: {e}"


def call_hf_llm(
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

    client = InferenceClient(token=HF_TOKEN)  # HF_TOKEN can be None

    # 1) chat endpoint
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
        return _extract_chat_content(resp)

    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        st.error(f"HF Inference error (chat): status={status}. Trying text-generation fallback...")

        # 2) fallback: text_generation
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
            st.error(f"HF Inference error (text): status={status2}.")
            return (
                "I couldn’t call the hosted LLM right now, but here are the most relevant excerpts:\n\n"
                + "\n\n---\n\n".join(contexts[:3])
            )

    except Exception as e:
        # show details in app (still safe; Streamlit redacts secrets)
        st.exception(e)
        return (
            "Unexpected LLM error. Check Streamlit logs.\n\n"
            "Meanwhile, here are the most relevant excerpts:\n\n"
            + "\n\n---\n\n".join(contexts[:3])
        )


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="MKT 333 — Beer AI & Video Games",
    page_icon="🍺",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Beer • AI • Video Games")
st.caption("Ask the course PDFs. Get clean, cited answers.")

# Sidebar controls
with st.sidebar:
    st.subheader("HF Settings")
    st.caption("Set HF_TOKEN in Streamlit secrets (Cloud) or env var (local).")
    st.text_input("HF model id", value=HF_LLM_MODEL, key="hf_model_id")

    if st.button("🔎 Test model", use_container_width=True):
        ok, msg = probe_hf_model(st.session_state.hf_model_id.strip())
        (st.success if ok else st.warning)(msg)

    st.subheader("Generation")
    st.session_state.setdefault("temperature", 0.2)
    st.session_state.setdefault("max_tokens", 512)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, float(st.session_state.temperature), 0.05)
    st.session_state.max_tokens = st.slider("Max tokens", 64, 1024, int(st.session_state.max_tokens), 64)

    st.subheader("Free model ideas")
    st.code(
        "\n".join([
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ]),
        language="text",
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! Ask me anything from the MKT 333 PDFs. 🍺🎮🤖"
    }]

# Build/load index (cached)
sig = pdf_signature()
index, chunks, meta = get_index_bundle(sig)

pdf_count = len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]) if os.path.exists(PDF_DIR) else 0
st.caption(f"● Backend: {'OK' if index is not None else 'NOT READY'} ({pdf_count} PDFs, {len(chunks)} chunks)")

if index is None:
    st.warning(f"No PDFs found in: {PDF_DIR}\n\nAdd PDFs to that folder in your repo, push, and refresh the app.")

# Render history
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
            answer = "I don’t have enough information in the PDFs to answer that."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            answer = call_hf_llm(
                question=prompt,
                contexts=contexts,
                citations=cites,
                model_id=st.session_state.hf_model_id.strip(),
                temperature=float(st.session_state.temperature),
                max_tokens=int(st.session_state.max_tokens),
            )
            st.markdown(answer)

            with st.expander("Sources used"):
                for i, m in enumerate(cites, start=1):
                    st.write(f"[S{i}] {m['file']} — chunk {m['chunk']}")

            st.session_state.messages.append({"role": "assistant", "content": answer})
