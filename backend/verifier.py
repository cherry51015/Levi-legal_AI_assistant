import re
import numpy as np
from hybrid_retriever import hybrid_search  # searches the GLOBAL 153K corpus, not the per-doc index

# Rule-based checker
def run_document_verifier_rules(text: str):
    checks = {
        "signatures": bool(re.search(r"signature|signed by", text, re.IGNORECASE)),
        "dates": bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)) or bool(re.search(r"\b\d{4}\b", text)),
        "parties": bool(re.search(r"between\s+\w+", text, re.IGNORECASE)),
        "jurisdiction": bool(re.search(r"jurisdiction|court|state of|high court|supreme court", text, re.IGNORECASE)),
    }
    score = sum(checks.values())
    return checks, score


# Main verifier -- rule checks stay local to the uploaded doc; precedent
# search now hits the GLOBAL 153K-clause corpus via hybrid BM25+FAISS,
# instead of the per-document index it was mistakenly searching before.
def run_document_verifier(doc_text, doc_chunks, doc_index, global_index, global_meta, top_k=3):
    rules, sufficiency_score = run_document_verifier_rules(doc_text)
    chunk_results = []
    for i, chunk in enumerate(doc_chunks):
        try:
            fused = hybrid_search(global_index, global_meta, chunk, k=top_k)
            similar_cases = [
                {"id": r["id"], "summary": r["text"][:300] + "...", "similarity_score": r["score"]}
                for r in fused
            ]
        except Exception:
            similar_cases = []
        chunk_results.append({
            "chunk_index": i,
            "chunk_preview": chunk[:100] + "...",
            "similar_cases": similar_cases
        })
    return {
        "sufficiency_score": sufficiency_score,
        "rule_checklist": rules,
        "chunks": chunk_results
    }
