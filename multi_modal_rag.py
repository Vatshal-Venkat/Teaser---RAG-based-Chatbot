# ===============================
# multi_modal_rag.py (Refactored with Copy Button, fixed model + mode toggle)
# ===============================
import os
import tempfile
import datetime
import pickle
import re
import numpy as np
from PyPDF2 import PdfReader
from PIL import Image
import faiss
import streamlit as st
import json
import base64
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
import time
from transformers import CLIPProcessor, CLIPModel
import torch
from sentence_transformers import SentenceTransformer
from html import escape

# -------------------------------
# GPU Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# API Setup
# -------------------------------
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.write("API Key loaded successfully")
except Exception as e:
    st.error(f"Secret loading failed: {e}")

# ✅ Use supported model
chat_model = genai.GenerativeModel("models/gemini-2.5-pro")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# -------------------------------
# Paths
# -------------------------------
TEXT_FAISS_PATH = "faiss_store/text_index.pkl"
IMAGE_FAISS_PATH = "faiss_store/image_index.pkl"
os.makedirs("faiss_store", exist_ok=True)

# -------------------------------
# PDF Helpers
# -------------------------------
def extract_text_from_pdfs(pdf_files):
    docs = []
    for pdf_file in pdf_files:
        fname = getattr(pdf_file, "name", None) or "uploaded.pdf"
        try:
            pdf_reader = PdfReader(pdf_file)
        except Exception:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
                tmpf.write(pdf_file.read())
                tmpf.flush()
                pdf_reader = PdfReader(tmpf.name)
                fname = tmpf.name
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": fname, "page": i + 1}))
    return docs

# -------------------------------
# Embedding Helpers
# -------------------------------
def embed_text(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_file):
    if hasattr(image_file, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_file.read())
            tmp.flush()
            path = tmp.name
    else:
        path = image_file
    image = Image.open(path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs).squeeze().cpu().numpy()
    if hasattr(image_file, "read"):
        os.remove(path)
    return path, emb

class ImageRetriever:
    def __init__(self):
        self.paths = []
        self.embeddings = None

    def add_images(self, new_paths, new_embeddings):
        if not new_paths or not new_embeddings:
            return
        self.paths.extend(new_paths)
        arr = np.array(new_embeddings).astype("float32")
        if self.embeddings is None:
            dim = arr.shape[1] if arr.ndim == 2 else arr.shape[0]
            self.embeddings = faiss.IndexFlatL2(dim)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.embeddings.add(arr)
        with open(IMAGE_FAISS_PATH + ".tmp", "wb") as f:
            pickle.dump({"paths": self.paths, "embeddings": self.embeddings}, f)
        os.replace(IMAGE_FAISS_PATH + ".tmp", IMAGE_FAISS_PATH)

    def get_relevant_images(self, query_text, top_k=3):
        if not self.paths or self.embeddings is None:
            return []
        inputs = clip_processor(text=query_text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            query_emb = clip_model.get_text_features(**inputs).cpu().numpy()
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        k = min(top_k, self.embeddings.ntotal)
        if k <= 0:
            return []
        D, I = self.embeddings.search(query_emb.astype("float32"), k)
        indices = I[0].tolist()
        return [self.paths[i] for i in indices]

image_retriever = ImageRetriever()
if os.path.exists(IMAGE_FAISS_PATH):
    try:
        with open(IMAGE_FAISS_PATH, "rb") as f:
            data = pickle.load(f)
            image_retriever.paths = data.get("paths", [])
            image_retriever.embeddings = data.get("embeddings", None)
    except Exception:
        image_retriever = ImageRetriever()

# -------------------------------
# FAISS + BM25
# -------------------------------
text_index = None
text_docs = []
bm25 = None
if os.path.exists(TEXT_FAISS_PATH):
    try:
        with open(TEXT_FAISS_PATH, "rb") as f:
            data = pickle.load(f)
            text_index = data.get("index")
            text_docs = data.get("docs", [])
            if text_docs:
                bm25 = BM25Okapi([d.page_content.split() for d in text_docs])
    except Exception:
        text_index, text_docs, bm25 = None, [], None

def save_text_faiss(index, docs, path=TEXT_FAISS_PATH):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"index": index, "docs": docs}, f)
    os.replace(tmp, path)

def delete_document_by_source(source_name):
    global text_docs, text_index, bm25
    new_docs = [d for d in text_docs if os.path.basename(d.metadata.get("source","")) != source_name]
    if new_docs != text_docs:
        text_docs = new_docs
        if text_docs:
            embeddings = embed_text([d.page_content for d in text_docs])
            dim = embeddings.shape[1]
            text_index = faiss.IndexFlatL2(dim)
            text_index.add(np.array(embeddings).astype("float32"))
            bm25 = BM25Okapi([d.page_content.split() for d in text_docs])
        else:
            text_index, bm25 = None, None
        save_text_faiss(text_index, text_docs)

# -------------------------------
# RAG Chat (with modes: KB, General, Hybrid)
# -------------------------------
def rag_chat_stream(query, use_images=True, top_k_text=6, top_k_images=2,
                   faiss_weight=0.6, bm25_weight=0.4, threshold=0.3):

    # --- Retrieve FAISS + BM25 documents ---
    faiss_scores, faiss_results = [], []
    if text_index and text_docs:
        try:
            query_emb = embed_text([query])[0].astype("float32")
            D, I = text_index.search(query_emb.reshape(1, -1), top_k_text)
            for dist, idx in zip(D[0], I[0]):
                if 0 <= idx < len(text_docs):
                    faiss_results.append(text_docs[idx])
                    faiss_scores.append(1 / (1 + float(dist)))
        except Exception:
            faiss_results, faiss_scores = [], []

    bm25_results, bm25_scores = [], []
    if bm25:
        try:
            scores = bm25.get_scores(query.split())
            top_idx = np.argsort(scores)[::-1][:top_k_text]
            for idx in top_idx:
                if 0 <= idx < len(text_docs):
                    bm25_results.append(text_docs[idx])
                    bm25_scores.append(float(scores[idx]))
        except Exception:
            bm25_results, bm25_scores = [], []

    # --- Combine and rank ---
    combined = {}
    for doc, score in zip(faiss_results, faiss_scores):
        key = (doc.page_content, (doc.metadata or {}).get("source"), (doc.metadata or {}).get("page"))
        combined[key] = combined.get(key, 0.0) + faiss_weight * score
    for doc, score in zip(bm25_results, bm25_scores):
        key = (doc.page_content, (doc.metadata or {}).get("source"), (doc.metadata or {}).get("page"))
        combined[key] = combined.get(key, 0.0) + bm25_weight * score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    # --- Apply relevance threshold ---
    filtered = [
        Document(page_content=k[0],
                 metadata={"source": k[1] or "uploaded.pdf", "page": k[2] or "?"})
        for k, score in ranked[:top_k_text] if score >= threshold
    ]

    retrieved_texts = filtered if filtered else []

    # --- Retrieve images ---
    retrieved_images = []
    if use_images and image_retriever:
        try:
            retrieved_images = image_retriever.get_relevant_images(query, top_k_images)
        except Exception:
            retrieved_images = []

    # --- Check if fallback to GEMINI is needed ---
    disclaimer_text = ""
    if not retrieved_texts:
        disclaimer_text = "⚠️ I couldn't find relevant information in my knowledge base, I'll fetch answer using my API."
        prompt = f"{disclaimer_text}\n\nQuestion: {query}\nAnswer using your general knowledge:"
        retrieved_texts = [Document(page_content=disclaimer_text, metadata={"source": "GEMINI API", "page": "Null"})]
    else:
        # Prepare context for KB
        text_contexts = [
            f"{d.page_content.strip()} (Source: {d.metadata.get('source','uploaded.pdf')}, Page: {d.metadata.get('page','?')})"
            for d in retrieved_texts
        ]
        image_context = " ".join([f"[Image: {os.path.basename(path)}]" for path in retrieved_images])
        context = "\n".join(text_contexts) + ("\n" + image_context if image_context else "")
        prompt = f"Use the following information to answer the question clearly and professionally:\n{context}\n\nQuestion: {query}\nAnswer:"

    # --- Generate response from Gemini ---
    response = chat_model.generate_content(prompt, stream=True)
    return response, retrieved_images, retrieved_texts


# -------------------------------
# The rest of your existing Streamlit UI code remains unchanged
# (Rendering chat bubbles, file uploader, chat input, streaming Gemini responses)
# -------------------------------

st.set_page_config(page_title="TEASER", layout="wide", page_icon="🤖")

# CSS + Copy Script
st.markdown("""<style>
.chat-container { max-width:600px; margin:auto; overflow-y:auto; max-height:75vh; padding-bottom:100px; }
.chat-row { display:flex; align-items:flex-start; margin:6px 0; }
.chat-avatar { font-size:28px; margin:6px; }
.chat-bubble { padding:8px 8px; border-radius:18px; max-width:75%; word-wrap:break-word; font-size:15px; line-height:1.5; }
.user-bubble { max-width:400px; margin-left:auto; text-align:right; background-color:#333; color:#fff; }
.assistant-bubble { margin-right:0px; text-align:left; background-color:transparent; color:#fff; white-space:pre-wrap; }
.timestamp { font-size:11px; color:#aaa; margin:2px 12px; text-align:right; }
.copy-btn { font-size:12px; color:#bbb; cursor:pointer; margin:4px 0 0 12px; }
.sidebar .block-container { background:#1e1e1e; color:white; }
.stFileUploader { max-width: 50% !important; margin:0 auto 20px auto; }
</style>
<script>
function copyToClipboard(id){
  const text = document.getElementById(id).innerText;
  navigator.clipboard.writeText(text);
}
</script>
""", unsafe_allow_html=True)


for i, msg in enumerate(st.session_state.get("messages", [])):
    ts = msg.get("time", "")
    if msg["role"] == "assistant":
        msg_id = f"assistant_{i}"
        st.markdown(f"""
        <div class='chat-row' style='justify-content:flex-start;'>
          <div class='chat-avatar'>🤖</div>
          <div>
            <div id='{msg_id}' class='chat-bubble assistant-bubble'>{msg['content']}</div>
            <div class='timestamp'>{ts}</div>
            <div class='copy-btn' onclick="copyToClipboard('{msg_id}')">📋 Copy</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        # (sources/images unchanged)


# --- Sidebar ---
with st.sidebar:
    st.markdown("<h3>💬 Chats</h3>", unsafe_allow_html=True)
    search_query = st.text_input("🔍 Search Chats", value="", placeholder="Search Chats...")
    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.rerun()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    filtered_history = [c for c in st.session_state.chat_history if search_query.lower() in c["title"].lower()]
    for idx, chat in enumerate(filtered_history):
        if st.button(f"💬 {chat['title']}", key=f"chat_{idx}"):
            st.session_state.messages = chat["messages"]
            st.rerun()

st.markdown("<h2 style='text-align:center;'>🤖 Meet TEASER</h2>", unsafe_allow_html=True)
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render previous messages ---
for msg in st.session_state.messages:
    ts = msg.get("time", "")
    if msg["role"] == "user":
        safe_text = escape(msg['content'])
        st.markdown(f"""
        <div class='chat-row' style='justify-content:flex-end;'>
          <div>
            <div class='chat-bubble user-bubble'>{safe_text}</div>
            <div class='timestamp'>{ts}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        st.markdown(f"""
        <div class='chat-row' style='justify-content:flex-start;'>
          <div class='chat-avatar'>🤖</div>
          <div class='chat-bubble assistant-bubble'>{msg['content']}</div>
          <div class='timestamp'>{ts}</div>
        </div>
        """, unsafe_allow_html=True)
        if msg.get("sources"):
            st.markdown("**Sources:**")
            for src in msg["sources"]:
                try:
                    sfile = os.path.basename(src.metadata.get("source", "uploaded.pdf"))
                    spage = src.metadata.get("page", "?")
                except Exception:
                    sfile = str(getattr(src, "source", "uploaded.pdf"))
                    spage = str(getattr(src, "page", "?"))
                st.markdown(f"- 📄 {sfile} (Page {spage})")
        if msg.get("images"):
            for img in msg["images"]:
                try:
                    st.image(img, caption="Relevant Image", use_container_width=True)
                except Exception:
                    pass

# --- File uploader ---
uploaded_files = st.file_uploader("Upload", type=["pdf","png","jpg","jpeg"], label_visibility="collapsed", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            docs = extract_text_from_pdfs([file])
            if docs:
                chunks = []
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                for d in docs:
                    parts = splitter.split_text(d.page_content)
                    for part in parts:
                        chunks.append(Document(page_content=part, metadata=d.metadata))
                if chunks:
                    embeddings = embed_text([c.page_content for c in chunks])
                    dim = embeddings.shape[1] if embeddings.ndim == 2 else 1
                    if text_index is None:
                        text_index = faiss.IndexFlatL2(dim)
                        text_docs = []
                    emb_arr = np.array(embeddings).astype("float32")
                    if emb_arr.ndim == 1:
                        emb_arr = emb_arr.reshape(1, -1)
                    text_index.add(emb_arr)
                    text_docs.extend(chunks)
                    save_text_faiss(text_index, text_docs)
                    bm25 = BM25Okapi([d.page_content.split() for d in text_docs])
            st.success(f"✅ Added {file.name} to knowledge base")
        elif file.name.lower().endswith((".png",".jpg",".jpeg")):
            path, emb = embed_image(file)
            image_retriever.add_images([path], [emb])
            st.success(f"✅ Added {file.name} to image index")

# --- Chat input ---
user_query = st.chat_input("Ask Query...")

if user_query:
    current_time = datetime.datetime.now().strftime("%H:%M")

    # Save user message instantly
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "time": current_time
    })
    st.session_state.chat_history.append({
        "title": user_query[:20],
        "messages": st.session_state.messages.copy()
    })

    # Render user bubble immediately
    st.markdown(f"""
    <div class='chat-row' style='justify-content:flex-end;'>
      <div>
        <div class='chat-bubble user-bubble'>{escape(user_query)}</div>
        <div class='timestamp'>{current_time}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Show bot "Thinking..." placeholder
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(f"""
    <div class='chat-row' style='justify-content:flex-start;'>
      <div class='chat-avatar'>🤖</div>
      <div>
        <div class='chat-bubble assistant-bubble'>
          <span class="loading-dots"><span>.</span><span>.</span><span>.</span></span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stream Gemini response
    response, retrieved_imgs, retrieved_sources = rag_chat_stream(user_query, use_images=True)
    collected_text = ""
    live_placeholder = st.empty()

    for chunk in response:
        text_piece = None
        if hasattr(chunk, "text"):
            text_piece = chunk.text
        elif isinstance(chunk, dict) and "text" in chunk:
            text_piece = chunk.get("text")
        elif hasattr(chunk, "candidates") and getattr(chunk, "candidates"):
            try:
                text_piece = chunk.candidates[0].get("content") or chunk.candidates[0].get("text")
            except Exception:
                text_piece = None
        if not text_piece:
            continue

        for ch in text_piece:
            collected_text += ch
            live_placeholder.markdown(f"""
                <div class='chat-row' style='justify-content:flex-start;'>
                  <div class='chat-avatar'>🤖</div>
                  <div class='chat-bubble assistant-bubble'>{escape(collected_text)}</div>
                </div>""", unsafe_allow_html=True)
            time.sleep(0.01)

    # Clear "Thinking..."
    thinking_placeholder.empty()

    # Finalize response
    formatted_answer = format_answer(collected_text)
    current_time = datetime.datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "assistant",
        "content": formatted_answer,
        "images": retrieved_imgs,
        "sources": retrieved_sources,
        "time": current_time
    })

    # Show sources
    if retrieved_sources:
        st.markdown("**Sources:**")
        for src in retrieved_sources:
            try:
                sfile = os.path.basename(src.metadata.get("source", "uploaded.pdf"))
                spage = src.metadata.get("page", "?")
            except Exception:
                sfile = str(getattr(src, "source", "uploaded.pdf"))
                spage = str(getattr(src, "page", "?"))
            st.markdown(f"- 📄 {sfile} (Page {spage})")

    # Show images
    if retrieved_imgs:
        for img in retrieved_imgs:
            try:
                st.image(img, caption="Relevant Image", use_container_width=True)
            except Exception:
                pass

st.markdown("</div>", unsafe_allow_html=True)
