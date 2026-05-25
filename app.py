"""Streamlit chat UI for the RAG index."""

from __future__ import annotations

import os

from datetime import datetime

import streamlit as st

# Bridge Streamlit Cloud secrets → environment variables before importing rag,
# so rag.py's env-based config works identically locally and in the cloud.
try:
    for _key, _val in st.secrets.items():
        if isinstance(_val, str) and _key not in os.environ:
            os.environ[_key] = _val
except (FileNotFoundError, Exception):
    pass

from export_pdf import generate as generate_pdf
from rag import COLLECTION_NAME, TOP_K, answer, get_collection, list_books

st.set_page_config(page_title="RAG — Cours Institut", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .stChatMessage { direction: rtl; text-align: right; }
    .stChatMessage[data-testid="stChatMessageContent"] p { text-align: right; }
    .source-tag { font-size: 0.85em; color: #888; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 RAG — Cours Institut")

with st.sidebar:
    st.header("Index")
    try:
        collection = get_collection()
        count = collection.count()
        st.metric("Indexed chunks", count)
        st.caption(f"Collection: `{COLLECTION_NAME}`")
        if count == 0:
            st.warning(
                "The index is empty. Run `python ingest.py` from a terminal to load PDFs."
            )
    except Exception as e:
        st.error(f"Could not open Chroma: {e}")

    st.header("Books")
    try:
        available_books = list_books()
    except Exception as e:
        available_books = []
        st.warning(f"Could not list books: {e}")
    if available_books:
        selected_books = st.multiselect(
            "Search in",
            options=available_books,
            default=available_books,
            help="Filter retrieval to specific books. Empty = no filter.",
        )
    else:
        selected_books = []
        st.caption("Books appear here once you ingest a PDF.")

    st.header("Settings")
    k = st.slider("Top-K passages", min_value=1, max_value=15, value=TOP_K)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.header("Export")
    msgs = st.session_state.get("messages", [])
    if msgs:
        try:
            pdf_bytes = generate_pdf(msgs)
            st.download_button(
                label="📥 Exporter en PDF",
                data=pdf_bytes,
                file_name=f"conversation-{datetime.now():%Y%m%d-%H%M}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF export failed: {e}")
    else:
        st.caption("Posez une question pour activer l'export PDF.")

    st.caption(f"Chat model: `{os.environ.get('CHAT_MODEL', 'gpt-4o-mini')}`")
    st.caption(
        f"Embedding model: `{os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')}`"
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**{os.path.basename(s.source)} — p.{s.page}** "
                        f"<span class='source-tag'>(dist={s.distance:.3f})</span>",
                        unsafe_allow_html=True,
                    )
                    st.text(s.text[:500] + ("…" if len(s.text) > 500 else ""))

prompt = st.chat_input("اطرح سؤالاً عن المحتوى… / Ask a question about the materials…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and answering…"):
            try:
                history_for_llm = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                reply, sources = answer(
                    prompt,
                    k=k,
                    books=selected_books or None,
                    history=history_for_llm,
                )
            except Exception as e:
                reply, sources = f"Error: {e}", []
        st.markdown(reply)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(
                        f"**{os.path.basename(s.source)} — p.{s.page}** "
                        f"<span class='source-tag'>(dist={s.distance:.3f})</span>",
                        unsafe_allow_html=True,
                    )
                    st.text(s.text[:500] + ("…" if len(s.text) > 500 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
