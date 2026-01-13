import streamlit as st
import ollama
import re
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import json

from typing import Optional  # ✅ ADD THIS

from sentence_transformers import SentenceTransformer
from huggingface_hub import login, HfApi 
# -----------------------------
# Syllabus file helper (UI)
# -----------------------------
SYLLABUS_CANDIDATE_PATHS = [
    r"D:\Maitri\USC\Grader\knowledge_base\MKT 333 - Innovation Economics and Business - Beer AI and Video Games - Syllabus - 12-26-2025.pdf"
    ]

def load_syllabus_bytes():
    for p in SYLLABUS_CANDIDATE_PATHS:
        try:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return f.read(), os.path.basename(p)
        except Exception:
            continue
    return None, None

# -----------------------------
# Page config (UI)
# -----------------------------
st.set_page_config(
    page_title="MKT 333 — Beer AI & Video Games",
    page_icon="🍺",
    layout="centered",
    initial_sidebar_state="expanded",  # show side panel
)

# -----------------------------
# (Optional) Hugging Face auth (kept minimal)
# NOTE: Put your token in env var HF_TOKEN or Streamlit secrets, NOT in code.
# -----------------------------
def get_hf_token() -> Optional[str]:
    # Streamlit Cloud secrets first
    try:
        tok = st.secrets.get("HF_TOKEN", None)
        if tok:
            return str(tok).strip()
    except Exception:
        pass

    # Local env vars next (PowerShell: $env:HF_TOKEN="...")
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return tok.strip() if tok else None

HF_TOKEN = get_hf_token()

def hf_token_ok(token: str) -> bool:
    try:
        HfApi().whoami(token=token)
        return True
    except Exception:
        return False 

###########################################
# PDF Extraction and RAG Functions with Caching
###########################################

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            raw_text = page.get_text()
            cleaned_text = re.sub(r"\n\s*\n+", "\n", raw_text)
            cleaned_text = re.sub(r"Page \d+", "", cleaned_text)
            text += cleaned_text + "\n"
    return text.strip()

def load_all_pdfs(folder_path):
    """Load all PDFs using cached JSON if available and up-to-date."""
    json_path = os.path.join(folder_path, "pdf_data.json")
    current_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    # Try to load cached data if exists
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                saved_data = json.load(f)

            needs_refresh = False
            saved_files = {entry["filename"]: entry for entry in saved_data}

            current_set = set(current_files)
            saved_set = set(saved_files.keys())

            if current_set != saved_set:
                needs_refresh = True
            else:
                for filename in current_files:
                    file_path = os.path.join(folder_path, filename)
                    current_mtime = os.path.getmtime(file_path)
                    if saved_files[filename]["last_modified"] < current_mtime:
                        needs_refresh = True
                        break

            if not needs_refresh:
                return [{"filename": entry["filename"], "text": entry["text"]} for entry in saved_data]

        except (json.JSONDecodeError, KeyError):
            pass  # Invalid cache, regenerate

    docs = []
    for filename in current_files:
        path = os.path.join(folder_path, filename)
        text = extract_text_from_pdf(path)
        docs.append(
            {
                "filename": filename,
                "text": text,
                "last_modified": os.path.getmtime(path),
            }
        )

    with open(json_path, "w") as f:
        json.dump(docs, f)

    return [{"filename": doc["filename"], "text": doc["text"]} for doc in docs]

def split_text(text, max_length=5000):
    """Split text into chunks of specified maximum length."""
    sentences = text.split("\n")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Initialize the SentenceTransformer model for embeddings
@st.cache_resource
def get_embedder():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

model = get_embedder()
def build_vector_store(docs):
    """Build a FAISS vector store from document chunks."""
    all_chunks = []
    metadata = []
    for doc in docs:
        chunks = split_text(doc["text"])
        all_chunks.extend(chunks)
        metadata.extend([{"filename": doc["filename"]}] * len(chunks))
    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, all_chunks, metadata

def retrieve_context(query, index, chunks, top_k=5):
    """Retrieve relevant context from the vector store."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    return "\n\n".join(retrieved)

# Theme toggle state
###########################################
# Chatbot Interface and Styling (UI ONLY)
###########################################

# --- UI State (theme + navigation) ---
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
    st.session_state.ui_dark_mode = st.toggle(
        "Dark mode",
        value=st.session_state.ui_dark_mode,
    )

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

# CSS (UI only)
st.markdown(
    f"""
<style>
/* App base */
.stApp {{
  background: {bg};
  color: {text};
}}
/* Sidebar tidy */
section[data-testid="stSidebar"] {{
  border-right: 1px solid var(--sb-border, rgba(231,234,240,0.12));
}}

/* Sidebar card (like your old screenshot) */
.sidebar-card {{
  background: rgba(15, 18, 28, 0.86);
  border: 1px solid rgba(231,234,240,0.12);
  border-radius: 18px;
  padding: 16px;
}}

.sidebar-title {{
  font-weight: 900;
  font-size: 1.05rem;
  margin: 0;
}}

.sidebar-sub {{
  margin-top: 8px;
  color: rgba(167,176,192,1);
  font-size: 0.95rem;
}}

/* Quick badge */
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

/* Link buttons */
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

/* Layout width + spacing */
.block-container {{
  padding-top: 1.10rem;
  max-width: 980px;
}}

/* Banner */
.top-banner {{
  background: {panel};
  border: 1px solid {border};
  border-radius: 18px;
  padding: 18px 18px;
  text-align: center;                 /* ✅ center everything */
}}
.course-line {{
  font-size: 0.98rem;
  color: {mut};
  font-weight: 700;
  letter-spacing: 0.2px;
}}
.hero-title {{
  margin-top: 8px;
  font-size: 1.70rem;                 /* ✅ bigger */
  font-weight: 900;                   /* ✅ strong heading */
  letter-spacing: 0.2px;
}}
.hero-sub {{
  margin-top: 6px;
  font-size: 1.02rem;                 /* ✅ readable */
  color: {mut};
}}
.tagline {{
  margin-top: 12px;
  font-size: 0.98rem;
  color: {text};
  opacity: 0.92;
}}

/* Download button -> pill look */
.stDownloadButton > button {{
  border-radius: 999px !important;
  border: 1px solid {border} !important;
  background: {panel} !important;
  color: {text} !important;
  font-weight: 800 !important;
  padding: 10px 14px !important;
}}
.stDownloadButton > button:hover {{
  border-color: rgba(153,0,0,0.45) !important;
  box-shadow: 0 0 0 2px rgba(153,0,0,0.10) !important;
}}

/* Chat bubbles */
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

/* ✅ Fix: dark mode showing black chat text
   Force ALL chat content to use theme text color */
[data-testid="stChatMessage"] * {{
  color: {text} !important;
}}
/* Keep reasoning muted */
.reasoning, .reasoning * {{
  color: {mut} !important;
  font-style: italic;
}}
/* Links readable */
[data-testid="stChatMessage"] a {{
  color: {accent2} !important;
}}

/* ✅ Bigger chat input (height + font) */
.stChatInput {{
  border-top: 1px solid {border};
  background: transparent;
}}
.stChatInput textarea {{
  background: {input_bg} !important;
  color: {text} !important;
  border-radius: 16px !important;
  border: 1px solid {border} !important;
  font-size: 1.08rem !important;      /* ✅ bigger font */
  line-height: 1.45 !important;
  min-height: 72px !important;        /* ✅ taller box */
  padding: 14px 16px !important;
}}
.stChatInput textarea::placeholder {{
  color: {mut} !important;
}}

/* Sidebar tidy */
section[data-testid="stSidebar"] {{
  border-right: 1px solid {border};
}}
</style>
""",
    unsafe_allow_html=True,
)

###########################################
# Session State Initialization
###########################################

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today? 🚀"}]
    st.session_state.model_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 912,
        "repeat_penalty": 1.1,
    }
    st.session_state.show_thinking = True
    st.session_state.show_reasoning = True

# Load saved settings
if os.path.exists("settings.json"):
    with open("settings.json", "r") as f:
        saved_settings = json.load(f)
        st.session_state.show_thinking = saved_settings.get("show_thinking", st.session_state.show_thinking)
        st.session_state.show_reasoning = saved_settings.get("show_reasoning", st.session_state.show_reasoning)
        st.session_state.model_config["temperature"] = saved_settings.get(
            "temperature", st.session_state.model_config["temperature"]
        )
        st.session_state.model_config["max_tokens"] = saved_settings.get(
            "max_tokens", st.session_state.model_config["max_tokens"]
        )

# Initialize vector store with cached PDF loading
if "vector_index" not in st.session_state:
    pdf_folder = "./pdfs"
    if os.path.exists(pdf_folder):
        docs = load_all_pdfs(pdf_folder)
        vector_index, chunks, metadatas = build_vector_store(docs)
        st.session_state.vector_index = vector_index
        st.session_state.chunks = chunks
        st.session_state.metadatas = metadatas
    else:
        st.session_state.vector_index = None
        st.session_state.chunks = None
        st.session_state.metadatas = None

###########################################
# Sidebar Controls + USC Links (UI-only add)
###########################################

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

    # Keep your existing Controls below (Regenerate, toggles, sliders, etc.)


    st.markdown("<hr/>", unsafe_allow_html=True)

    st.subheader("Model Settings")
    st.toggle("Show Thinking Animation", key="show_thinking")
    st.toggle("Show AI Reasoning", key="show_reasoning")
    st.session_state.model_config["temperature"] = st.slider(
        "Temperature", 0.0, 1.0, st.session_state.model_config["temperature"], 0.1
    )
    st.session_state.model_config["max_tokens"] = st.slider(
        "Max Tokens", 128, 1024, st.session_state.model_config["max_tokens"], 128
    )

    def save_settings():
        settings = {
            "show_thinking": st.session_state.show_thinking,
            "show_reasoning": st.session_state.show_reasoning,
            "temperature": st.session_state.model_config["temperature"],
            "max_tokens": st.session_state.model_config["max_tokens"],
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)
        st.sidebar.success("Settings saved!")

    if st.button("💾 Save Settings"):
        save_settings()

    # Function to reload all PDFs and update JSON (super experimental) not sure if this helps or not, in theory JSON should help
    def recalculate_pdf_data():
        pdf_folder = "./knowledge_base"  # Root directory folder
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

        if not pdf_files:
            st.sidebar.error("No PDFs found in 'pdfs' folder.")
            return

        pdf_data = {"files": pdf_files}

        with open("pdf_data.json", "w") as json_file:
            json.dump(pdf_data, json_file, indent=4)

        st.sidebar.success("PDF data recalculated!")

    st.sidebar.header("Options")
    if st.sidebar.button("♻️ Recalculate PDF Data"):
        with st.spinner("Processing PDFs..."):
            recalculate_pdf_data()

# Avatars
user_avatar = "👤"
ai_avatar = "🤖"

###########################################
# Chat Functions
###########################################

def parse_response(response):
    """Extract reasoning and content from response using <think> tags."""
    match = re.search(r"<think>(.*?)</think>(.*)", response, re.DOTALL)
    if match:
        return {"reasoning": match.group(1).strip(), "content": match.group(2).strip()}
    return {"reasoning": "", "content": response}

def display_response(parsed, placeholder):
    """Display response with optional reasoning."""
    final_display = []
    if st.session_state.show_reasoning and parsed["reasoning"]:
        final_display.append(f"<div class='reasoning'>🤔 {parsed['reasoning']}</div>")
    final_display.append(parsed["content"])
    placeholder.markdown("\n".join(final_display), unsafe_allow_html=True)

def generate_response():
    """Generate and display AI response with RAG context."""
    user_prompt = st.session_state.messages[-1]["content"]
    retrieved_context = ""
    if st.session_state.vector_index is not None and st.session_state.chunks is not None:
        retrieved_context = retrieve_context(user_prompt, st.session_state.vector_index, st.session_state.chunks)

    system_prompt = f"""
         Use the following retrieved context to answer the query accurately:
         {retrieved_context}

         Try to always cite information from the documents. If unsure, say 'I don’t have enough information to answer this.'
         """

    augmented_messages = []
    if system_prompt:
        augmented_messages.append({"role": "system", "content": system_prompt})
    augmented_messages.extend(st.session_state.messages)

    with st.chat_message("assistant", avatar=ai_avatar):
        response_placeholder = st.empty()
        if st.session_state.show_thinking:
            response_placeholder.markdown(
                """
                <div style="display: flex; align-items: center; gap: 0.5rem">
                    <div class="typing-animation">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        full_response = ""
        for chunk in ollama.chat(
            model="deepseek-r1:1.5b",
            messages=augmented_messages,
            stream=True,
            options={
                "temperature": st.session_state.model_config["temperature"],
                "top_p": st.session_state.model_config["top_p"],
                "num_predict": st.session_state.model_config["max_tokens"],
                "repeat_penalty": st.session_state.model_config["repeat_penalty"],
            },
        ):
            full_response += chunk["message"]["content"]
            cursor = "▌" if not st.session_state.show_thinking else ""
            response_placeholder.markdown(full_response + cursor)

        parsed = parse_response(full_response)
        message = {"role": "assistant", "content": parsed["content"], "reasoning": parsed["reasoning"]}
        st.session_state.messages.append(message)
        display_response(parsed, response_placeholder)

def is_response_incomplete(response):
    """Check if response appears incomplete."""
    response = response.strip()
    return response and response[-1] not in [".", "!", "?", '"', "'"]

def continue_response():
    """Continue the last assistant response."""
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_assistant = st.session_state.messages.pop()
        st.session_state.messages.append({"role": "user", "content": "Please continue your previous answer."})
        generate_response()
        new_assistant = st.session_state.messages.pop()
        combined_content = last_assistant["content"].strip() + "\n" + new_assistant["content"].strip()
        combined_reasoning = (
            last_assistant.get("reasoning", "").strip()
            + "\n"
            + new_assistant.get("reasoning", "").strip()
        ).strip()
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_content, "reasoning": combined_reasoning}
        )

###########################################
# Chat History Display
###########################################

for message in st.session_state.messages:
    
    role = "user" if message["role"] == "user" else "AI"
    with st.chat_message(role, avatar=user_avatar if role == "user" else ai_avatar):
        reasoning = message.get("reasoning", "")
        content = message.get("content", "")
        if st.session_state.show_reasoning and reasoning:
            st.markdown(f"<div class='reasoning'>🤔 {reasoning}</div>{content}", unsafe_allow_html=True)
        else:
            st.markdown(content)

if hasattr(st.session_state, "regenerate") and st.session_state.regenerate:
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        generate_response()
    st.session_state.regenerate = False

###########################################
# User Input Handling
###########################################

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    generate_response()

# --- Top row above chat: title left, regenerate right (UI) ---
row_l, row_r = st.columns([0.78, 0.22], vertical_alignment="center")
with row_l:
    st.markdown("###")

with row_r:
    last_is_ai = len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "assistant"
    if st.button("Regenerate", disabled=not last_is_ai, use_container_width=True):
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()
            st.session_state.regenerate = True
            st.rerun()