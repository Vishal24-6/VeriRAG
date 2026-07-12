"""
app.py — Streamlit UI Layer

This is a thin UI layer that delegates all RAG logic to pipeline.py.
It handles only: PDF upload, chat display, and result rendering.

Architecture:
  config.py   → hyperparameters
  pipeline.py → orchestration (calls retriever, llm, web_search, chunking)
  app.py      → UI only (this file)
"""

import streamlit as st
from pypdf import PdfReader
import hashlib
import time

import config
from retriever import build_index
from pipeline import run_query


st.set_page_config(page_title="Hybrid Multi-Doc RAG Assistant", layout="wide")

st.title(config.APP_TITLE)
st.caption(
    f"v{config.APP_VERSION} — Upload PDFs and ask questions — powered by "
    "hybrid retrieval, cross-encoder reranking, and web search fallback"
)


if "messages" not in st.session_state:
    st.session_state.messages = []


# NOTE: Caching removed — @st.cache_resource is unsafe with mutable dict inputs
# and can serve stale/empty data across reruns, causing BM25 ZeroDivisionError


# ==================================================
# SIDEBAR — Document Upload + Info
# ==================================================

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} document(s) loaded")

        with st.expander("Uploaded files"):
            for file in uploaded_files:
                st.markdown(f"• **{file.name}** ({round(file.size / 1024, 1)} KB)")

    with st.expander('🔍 How it works'):
        st.markdown('''
    1. **Chunking** — Documents are split into overlapping passages
    2. **Hybrid Search** — BM25 (keyword) + FAISS (semantic) retrieval
    3. **RRF Fusion** — Combines rankings from both search methods
    4. **Cross-Encoder** — Reranks results for precision
    5. **Confidence Gate** — Low-confidence triggers web search fallback
    6. **LLM Generation** — Groq LLaMA 3.1 generates cited answers
    ''')

    with st.expander('⚙️ Configuration'):
        st.markdown(f'''
    | Setting | Value |
    |---------|-------|
    | Bi-Encoder | `{config.BI_ENCODER_MODEL}` |
    | Cross-Encoder | `{config.CROSS_ENCODER_MODEL}` |
    | LLM | `{config.LLM_MODEL}` |
    | Chunk Size | `{config.CHUNK_SIZE}` chars |
    | Chunk Overlap | `{config.CHUNK_OVERLAP}` chars |
    | RRF k | `{config.RRF_K}` |
    | Drift Threshold | `{config.DRIFT_THRESHOLD}` |
    | Confidence Gate | mean < `{config.CONFIDENCE_MEAN_THRESHOLD}` or max < `{config.CONFIDENCE_MAX_THRESHOLD}` |
    ''')


# ==================================================
# PDF PROCESSING
# ==================================================

if uploaded_files:

    documents = {}

    for file in uploaded_files:

        try:
            reader = PdfReader(file)
        except Exception as e:
            print(f"[DEBUG] Failed to read {file.name}: {e}")
            st.warning(f"⚠️ Could not read **{file.name}** — it may be corrupted or encrypted. Skipping.")
            continue

        text = ""

        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception:
                continue  # Skip unreadable pages silently

        documents[file.name] = text

    # Filter out documents with no extractable text
    documents = {name: text for name, text in documents.items() if text.strip()}

    if not documents:
        st.error("❌ No valid text found in uploaded documents. Please upload PDFs that contain selectable text.")
        st.stop()

    # Debug logging
    print(f"[DEBUG] Documents: {len(documents)}")
    for name, text in documents.items():
        print(f"[DEBUG]   {name}: {len(text)} chars")

    # --- Safe caching via session_state (keyed on document content hash) ---
    doc_hash = hashlib.md5(
        "".join(f"{k}:{v}" for k, v in sorted(documents.items())).encode()
    ).hexdigest()

    if "index_hash" not in st.session_state or st.session_state.index_hash != doc_hash:
        print(f"[DEBUG] Building index (hash: {doc_hash[:8]}...)")
        bm25_index, chunks, sources = build_index(documents)
        st.session_state.index_hash = doc_hash
        st.session_state.index_data = (bm25_index, chunks, sources)
    else:
        print(f"[DEBUG] Using cached index (hash: {doc_hash[:8]}...)")
        bm25_index, chunks, sources = st.session_state.index_data

    if bm25_index[0] is None or not chunks:
        st.error("❌ Failed to process document. Please upload a valid text-based PDF.")
        st.stop()


    # ==================================================
    # CHAT DISPLAY
    # ==================================================

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # ==================================================
    # QUERY HANDLING — Delegates to pipeline.py
    # ==================================================

    if prompt := st.chat_input("Ask a question about the documents"):

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # ── Run the full pipeline ──────────────────────
        with st.spinner("Processing your question..."):
            result = run_query(
                index_bundle=bm25_index,
                chunks=chunks,
                sources=sources,
                question=prompt,
                chat_history=st.session_state.messages,
            )

        # ── Render Result ──────────────────────────────
        with st.chat_message("assistant"):
            if result.used_web:
                st.caption("📡 Enhanced with web search")

            if result.is_error:
                st.warning(result.answer)
            else:
                st.markdown(result.answer)

        # ── Metrics bar ────────────────────────────────
        st.markdown(
            f"**Retrieval Score:** `{result.confidence}` | "
            f"**Sources Used:** `{result.num_chunks}` | "
            f"**Response Time:** `{result.elapsed:.1f}s`"
        )

        st.divider()

        # ── Sources ────────────────────────────────────
        st.subheader("Sources")
        for source in result.sources:
            st.markdown(f"• **{source}**")

        # ── Save to history ────────────────────────────
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer
        })

else:
    st.info("👈 Upload PDF documents from the sidebar to get started")