"""
Hybrid retriever: combines your existing dense FAISS search with a BM25
keyword retriever using LangChain's EnsembleRetriever.

Nothing about your existing pipeline changes — this wraps the same
faiss_index.bin + faiss_index.bin.meta.json + embed_texts() you already
have (from indexig.py / embeddings.py) inside a LangChain-compatible
retriever, then fuses it with BM25 via reciprocal-rank-style ensembling.

Install:
    pip install langchain langchain-community rank_bm25 faiss-cpu

Usage in llm.py — replace wherever you currently do:
    query_vec = embed_texts(query)
    scores, neighbors = index.search(...)
with:
    from hybrid_retriever import hybrid_search
    top_clauses = hybrid_search(query, k=5)   # list of strings
    # pass top_clauses into your existing ask_gemini(prompt, document) call
"""

import json
import faiss
import numpy as np
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from pydantic import Field

from embeddings import embed_texts  # your existing query embedding fn

INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_index.bin.meta.json"


class GeminiFaissRetriever(BaseRetriever):
    """Wraps your existing FAISS index + Gemini embed_texts as a LangChain retriever."""

    index: object = Field(default=None)
    ids: list = Field(default=None)
    texts: list = Field(default=None)
    k: int = 5

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        query_vec = embed_texts(query).reshape(1, -1)
        faiss.normalize_L2(query_vec)
        scores, neighbors = self.index.search(query_vec, self.k)
        docs = []
        for idx, score in zip(neighbors[0], scores[0]):
            docs.append(Document(
                page_content=self.texts[idx],
                metadata={"id": self.ids[idx], "dense_score": float(score)},
            ))
        return docs


def load_hybrid_retriever(k=5, bm25_weight=0.5, dense_weight=0.5):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ids, texts = meta["ids"], meta["texts"]

    dense_retriever = GeminiFaissRetriever(index=index, ids=ids, texts=texts, k=k)

    # NOTE: BM25Retriever.from_texts tokenizes all 153K clauses at startup.
    # This takes a while the first time you run it (minutes, not seconds,
    # at this corpus size) — do it once per process, not per query.
    bm25_retriever = BM25Retriever.from_texts(
        texts, metadatas=[{"id": i} for i in ids]
    )
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[bm25_weight, dense_weight],
    )


# Load once at import time — reuse this across requests instead of
# reconstructing the BM25 index on every call.
_hybrid = None


def hybrid_search(query: str, k: int = 5) -> list[str]:
    """Returns the top-k clause texts, fused from BM25 + dense FAISS search."""
    global _hybrid
    if _hybrid is None:
        _hybrid = load_hybrid_retriever(k=k)
    results = _hybrid.invoke(query)
    return [doc.page_content for doc in results]


if __name__ == "__main__":
    # quick smoke test
    q = "What is the notice period for termination?"
    for text in hybrid_search(q, k=3):
        print("-", text[:100].replace("\n", " "), "...")
