import streamlit as st
import ollama
import os
import fitz
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import pandas as pd
from io import StringIO
import time


model = SentenceTransformer('bge-small-en-v1.5')



def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text


def load_all_pdfs(folder_path):
    json_path = os.path.join(folder_path, "pdf_data.json")
    current_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                saved_data = json.load(f)
            saved_files = {entry["filename"]: entry for entry in saved_data}
            current_set = set(current_files)
            saved_set = set(saved_files.keys())
            needs_refresh = current_set != saved_set
            if not needs_refresh:
                for filename in current_files:
                    file_path = os.path.join(folder_path, filename)
                    if saved_files[filename]["last_modified"] < os.path.getmtime(file_path):
                        needs_refresh = True
                        break
            if not needs_refresh:
                return [{"filename": entry["filename"], "text": entry["text"]} for entry in saved_data]
        except (json.JSONDecodeError, KeyError):
            pass
    docs = []
    for filename in current_files:
        path = os.path.join(folder_path, filename)
        text = extract_text_from_pdf(path)
        docs.append({"filename": filename, "text": text, "last_modified": os.path.getmtime(path)})
    with open(json_path, "w") as f:
        json.dump(docs, f)
    return [{"filename": doc["filename"], "text": doc["text"]} for doc in docs]


def split_text(text, max_length=5000):
    sentences = text.split('\n')
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


def build_vector_store(docs):
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


def retrieve_context(query, index, chunks, top_k=2):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    return "\n\n".join(retrieved)


# Evaluation functions
TEST_SET = [
    {
        "query": "When was the School of Engineering and Physical Sciences founded?",
        "expected": "The School of Engineering and Physical Sciences (SEPS) started its journey in 1993 as the School of Engineering and Applied Sciences (SEAS), later renamed SEPS in 2014."
    },
    {
        "query": "How many students are currently enrolled in SEPS?",
        "expected": "SEPS is currently home to over 7,500 undergraduate and graduate students."
    },
    {
        "query": "Which departments are part of SEPS?",
        "expected": "SEPS includes the Department of Architecture (DoA), the Department of Civil and Environmental Engineering (CEE), the Department of Electrical and Computer Engineering (ECE), and the Department of Mathematics and Physics (DMP)."
    },
    {
        "query": "What undergraduate programs does the Department of Electrical and Computer Engineering offer?",
        "expected": "The Department of Electrical and Computer Engineering (ECE) offers Bachelor of Science in Computer Science and Engineering (BS CSE), Bachelor of Science in Electrical and Electronic Engineering (BS EEE), and Bachelor of Science in Electronics and Telecommunication Engineering (BS ETE)."
    },
    {
        "query": "What is the duration and credit hours for the Bachelor of Architecture program?",
        "expected": "The Bachelor of Architecture (B. Arch) program requires 176 credits and takes 5 years."
    },
    {
        "query": "How is NSU ranked for engineering according to Times Higher Education 2023?",
        "expected": "NSU is ranked #1 in Bangladesh for engineering by the Times Higher Education World University Rankings 2023, with a global rank of 301-400."
    },
    {
        "query": "What is the vision of SEPS?",
        "expected": "The vision of SEPS is to be a center of excellence in innovation and technological entrepreneurship by building a knowledge and skill-based learning environment in engineering, architecture, and physical sciences with technical competency, social responsibility, communication skills, and ethical standards."
    },
    {
        "query": "What is one of the missions of SEPS?",
        "expected": "One mission of SEPS is to maintain international standards in program curricula, instruction style, laboratory and research facilities, faculty recruitment, and student intake."
    },
    {
        "query": "How has undergraduate student intake changed over the past five years?",
        "expected": "Over the past five years, the undergraduate student intake at SEPS has grown steadily from 1,100 to over 2,000."
    },
    {
        "query": "What laboratory facilities does the Department of Civil and Environmental Engineering currently have?",
        "expected": "The Department of Civil and Environmental Engineering (CEE) currently has 6 testing labs and 1 drawing lab."
    },
    {
        "query": "What new labs are planned for the Department of Architecture by 2023?",
        "expected": "The Department of Architecture plans to add 1 Design Lab, 1 3D Printing Lab, 1 Photography Lab, and 1 Simulation Lab by 2023."
    },
    {
        "query": "Are the engineering programs at SEPS accredited?",
        "expected": "Yes, all engineering programs under SEPS are accredited by the Board of Accreditation for Engineering and Technical Education (BAETE)."
    }
]

def evaluate_accuracy(test_set, vector_index, chunks, model_config):
    if vector_index is None or chunks is None:
        return [{"query": "N/A", "expected": "N/A", "response": "No PDFs loaded for evaluation",
                 "similarity": 0.0, "is_correct": False}]

    results = []
    for item in test_set:
        query = item["query"]
        expected = item["expected"]
        retrieved_context = retrieve_context(query, vector_index, chunks)
        system_prompt = f"""
        Use the following retrieved context to answer the query accurately:
        {retrieved_context}
        Try to always cite information from the documents. If unsure, say 'I don’t have enough information to answer this.'
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
        response = ollama.chat(
            model="deepseek-r1:1.5b",
            messages=messages,
            options=model_config
        )["message"]["content"]
        emb_exp = model.encode(expected)
        emb_res = model.encode(response)
        similarity = np.dot(emb_exp, emb_res) / (np.linalg.norm(emb_exp) * np.linalg.norm(emb_res))
        is_correct = similarity > 0.85
        results.append({
            "query": query,
            "expected": expected,
            "response": response,
            "similarity": similarity,
            "is_correct": is_correct
        })
    return results


def generate_report(results):
    report = StringIO()
    report.write("=== NSU Campus AI Assistant Accuracy Report ===\n")
    report.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.write(f"Total Queries Evaluated: {len(results)}\n")
    correct_count = sum(1 for r in results if r["is_correct"])
    accuracy = correct_count / len(results) * 100 if results else 0
    report.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
    report.write("Detailed Results:\n")
    report.write("-" * 50 + "\n")
    for i, result in enumerate(results, 1):
        report.write(f"Query {i}: {result['query']}\n")
        report.write(f"Expected: {result['expected']}\n")
        report.write(f"Response: {result['response']}\n")
        report.write(f"Similarity: {result['similarity']:.3f}\n")
        report.write(f"Correct: {'Yes' if result['is_correct'] else 'No'}\n")
        report.write("-" * 50 + "\n")
    return report.getvalue()


# Streamlit app setup
st.set_page_config(initial_sidebar_state="collapsed")


# Session state initialization with explicit vector store setup
def initialize_vector_store():
    pdf_folder = "./pdfs"
    if os.path.exists(pdf_folder) and any(f.lower().endswith(".pdf") for f in os.listdir(pdf_folder)):
        docs = load_all_pdfs(pdf_folder)
        if docs:
            vector_index, chunks, metadatas = build_vector_store(docs)
            st.session_state.vector_index = vector_index
            st.session_state.chunks = chunks
            st.session_state.metadatas = metadatas
        else:
            st.session_state.vector_index = None
            st.session_state.chunks = None
            st.session_state.metadatas = None
    else:
        st.session_state.vector_index = None
        st.session_state.chunks = None
        st.session_state.metadatas = None
        st.warning("No PDFs found in './pdfs' folder. Please add PDFs to enable functionality.")


if "vector_index" not in st.session_state:
    initialize_vector_store()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today? 🚀"}]
    st.session_state.model_config = {"temperature": 0.4, "top_p": 0.9, "max_tokens": 912, "repeat_penalty": 1.1}

# Sidebar with evaluation
with st.sidebar:
    st.header("Controls")
    if st.button("♻️ Reload PDFs"):
        with st.spinner("Reloading PDFs..."):
            initialize_vector_store()
            st.success("PDFs reloaded successfully!")

    st.subheader("Evaluation")
    if st.button("📊 Generate Accuracy Report"):
        with st.spinner("Evaluating chatbot accuracy..."):
            eval_results = evaluate_accuracy(
                TEST_SET,
                st.session_state.get("vector_index"),
                st.session_state.get("chunks"),
                st.session_state.model_config
            )
            report_text = generate_report(eval_results)
            st.text_area("Accuracy Report", report_text, height=300)
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"chatbot_accuracy_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )


st.title("NSU Campus AI Assistant")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
