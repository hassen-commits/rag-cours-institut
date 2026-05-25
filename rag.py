"""Shared RAG primitives: Chroma collection access, retrieval, and answer generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "cours")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
HYDE_ENABLED = os.environ.get("HYDE_ENABLED", "1") not in ("0", "false", "False", "")
HYDE_MODEL = os.environ.get("HYDE_MODEL", "gpt-4o-mini")

HYDE_PROMPT = (
    "You generate a hypothetical short passage that would directly answer the user's "
    "question, as if quoted from a classical Arabic Islamic text. Write 3-5 sentences "
    "in classical Arabic, using vocabulary and turns of phrase typical of authors like "
    "al-Muhasibi, al-Ghazali, or Ibn al-Qayyim. Do not say 'I don't know' — invent a "
    "plausible passage. Do not translate. Output only the Arabic passage, no preamble."
)

REWRITE_PROMPT = (
    "You rewrite the user's latest message into a standalone question that includes "
    "all the context needed to answer it, resolving pronouns and implicit references "
    "from earlier turns. Output only the rewritten question — no preamble, no quotes. "
    "If the latest message is already standalone, return it unchanged. Match the "
    "user's language."
)

SYSTEM_PROMPT = (
    "You are a helpful study assistant for an institute's course materials "
    "(classical Arabic Islamic texts). Use the provided context excerpts to "
    "answer the user's question.\n\n"
    "Behavior:\n"
    "- If the context directly answers the question, synthesize a clear answer "
    "and cite each claim inline as [source:page].\n"
    "- If the context only touches on the topic indirectly, summarize what IS "
    "there, quote the most relevant Arabic phrases verbatim, and note that the "
    "passages don't fully address the question.\n"
    "- Only say you don't know if there is truly nothing relevant in the context.\n"
    "- Treat transliterations and translations as equivalent to their Arabic "
    "originals (e.g., 'muraqaba' = مراقبة = 'watchfulness'; 'muhasaba' = محاسبة "
    "= 'self-accounting'). Don't refuse just because the exact word form differs.\n"
    "- Reply in the language of the user's question.\n"
    "  • If the user writes in French or English, write the explanation in that "
    "language. When you quote an Arabic passage from the context, immediately "
    "follow it with a French/English translation in parentheses or after a dash. "
    "Always give the reader the meaning, never raw untranslated Arabic.\n"
    "  • If the user writes in Arabic, reply in Arabic and quote sources verbatim.\n"
    "- When the user asks about a concept, look for the concept in the context "
    "even if the wording differs, and explain what the source teaches about it."
)


@dataclass
class Retrieved:
    text: str
    source: str
    page: int
    distance: float


def _require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return key


def get_collection() -> chromadb.Collection:
    """Return the persistent Chroma collection, creating it if missing."""
    api_key = _require_api_key()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _rewrite_query(question: str, history: list[dict]) -> str:
    """Resolve pronouns / implicit references in the latest user message using prior turns."""
    if not history:
        return question
    client = OpenAI(api_key=_require_api_key())
    convo_lines = []
    for msg in history[-6:]:  # last 3 turns max — enough context, keeps cost down
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if content:
            convo_lines.append(f"{role}: {content}")
    convo_lines.append(f"User: {question}")
    convo = "\n".join(convo_lines)

    resp = client.chat.completions.create(
        model=HYDE_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": convo},
        ],
        temperature=0,
        max_tokens=200,
    )
    rewritten = (resp.choices[0].message.content or "").strip()
    return rewritten or question


def _generate_hypothetical(question: str) -> str:
    """HyDE: ask the LLM to write a plausible answer passage we'll embed instead."""
    client = OpenAI(api_key=_require_api_key())
    resp = client.chat.completions.create(
        model=HYDE_MODEL,
        messages=[
            {"role": "system", "content": HYDE_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


def _book_filter(books: list[str] | None) -> dict | None:
    if not books:
        return None
    if len(books) == 1:
        return {"book": books[0]}
    return {"book": {"$in": list(books)}}


def list_books() -> list[str]:
    """Return the unique `book` metadata values currently in the collection."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.get(include=["metadatas"])
    seen: set[str] = set()
    for meta in result.get("metadatas", []) or []:
        b = meta.get("book") if meta else None
        if b:
            seen.add(str(b))
    return sorted(seen)


def retrieve(
    question: str,
    k: int = TOP_K,
    use_hyde: bool | None = None,
    books: list[str] | None = None,
) -> list[Retrieved]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    if use_hyde is None:
        use_hyde = HYDE_ENABLED

    where = _book_filter(books)

    def _query(text: str):
        kwargs = {"query_texts": [text], "n_results": k}
        if where is not None:
            kwargs["where"] = where
        return collection.query(**kwargs)

    if use_hyde:
        hypothetical = _generate_hypothetical(question)
        merged: dict[str, tuple[str, dict, float]] = {}
        for query_text in (hypothetical, question):
            if not query_text.strip():
                continue
            result = _query(query_text)
            ids = result.get("ids", [[]])[0]
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            dists = result.get("distances", [[]])[0]
            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                prev = merged.get(cid)
                if prev is None or dist < prev[2]:
                    merged[cid] = (doc, meta, dist)
        ranked = sorted(merged.values(), key=lambda x: x[2])[:k]
        return [
            Retrieved(
                text=doc,
                source=str(meta.get("source", "?")),
                page=int(meta.get("page", -1)),
                distance=float(dist),
            )
            for doc, meta, dist in ranked
        ]

    result = _query(question)
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    return [
        Retrieved(
            text=doc,
            source=str(meta.get("source", "?")),
            page=int(meta.get("page", -1)),
            distance=float(dist),
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def format_context(chunks: Iterable[Retrieved]) -> str:
    parts = []
    for c in chunks:
        tag = f"[{os.path.basename(c.source)}:p{c.page}]"
        parts.append(f"{tag}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def answer(
    question: str,
    k: int = TOP_K,
    books: list[str] | None = None,
    history: list[dict] | None = None,
) -> tuple[str, list[Retrieved]]:
    """Retrieve context and generate an answer. Returns (answer_text, sources).

    `history` is a list of prior {"role": "user"|"assistant", "content": str} messages
    (without sources). When provided, the latest question is rewritten into a
    standalone form before retrieval, and the model sees the full conversation.
    """
    history = history or []
    retrieval_query = _rewrite_query(question, history) if history else question

    chunks = retrieve(retrieval_query, k=k, books=books)
    if not chunks:
        return (
            "The index is empty (or no chunks match the selected books). "
            "Run `python ingest.py` first to load documents.",
            [],
        )

    context = format_context(chunks)
    user_prompt = (
        f"Context excerpts retrieved for: {retrieval_query}\n\n"
        f"{context}\n\n"
        f"Latest question: {question}\n\n"
        "Answer using only the excerpts above. Cite as [filename:pN]."
    )

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-6:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})

    client = OpenAI(api_key=_require_api_key())
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content or "", chunks
