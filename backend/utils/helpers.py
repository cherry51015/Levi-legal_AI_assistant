import random
import re
import json
import numpy as np
import faiss
from utils.embeddings import embed_texts  # your existing embedding function

# -----------------------------
# FAISS helpers
# -----------------------------
INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/faiss_index.bin.meta.json"
TOP_K = 5

def load_faiss_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def search_similar_docs(query_text, top_k=TOP_K):
    from hybrid_retriever import hybrid_search
    index, meta = load_faiss_index()
    fused = hybrid_search(index, meta, query_text, top_k)
    return [
        {"doc_id": r["id"], "snippet": r["text"][:200], "similarity": r["score"]}
        for r in fused
    ]

# -----------------------------
# Rule-based checklist
# -----------------------------
def check_rules(doc_text: str) -> dict:
    """
    Basic rule-based checks: signatures, dates, parties, jurisdiction
    """
    rules = {}

    # signatures: look for 'signed', 'signature', 'digitally signed'
    rules["signatures"] = bool(re.search(r"signed|signature|digitally signed", doc_text, re.I))

    # dates: dd/mm/yyyy or Month dd, yyyy
    rules["dates"] = bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", doc_text) or
                           re.search(r"\b(january|february|march|april|may|june|july|august|"
                                     r"september|october|november|december)\s+\d{1,2},?\s+\d{4}\b", doc_text, re.I))

    # parties: look for 'between', 'and', 'party', 'plaintiff', 'defendant'
    rules["parties"] = bool(re.search(r"\b(between|and|party|plaintiff|defendant)\b", doc_text, re.I))

    # jurisdiction: 'court', 'tribunal', 'act', 'law'
    rules["jurisdiction"] = bool(re.search(r"\b(court|tribunal|act|law)\b", doc_text, re.I))

    return rules

# -----------------------------
# ML-based sufficiency score (simple heuristic for demo)
# -----------------------------
def compute_sufficiency_score(doc_text: str) -> float:
    """
    Returns a 0-100 sufficiency score based on number of rules passed
    """
    rules = check_rules(doc_text)
    score = (sum(rules.values()) / len(rules)) * 100
    return score

# -----------------------------
# Full Document Verifier Pipeline
# -----------------------------
def run_verifier(doc_text: str) -> dict:
    """
    Runs full verifier pipeline and returns JSON-ready output
    """
    rules_result = check_rules(doc_text)
    suff_score = compute_sufficiency_score(doc_text)
    recommendations = search_similar_docs(doc_text)

    return {
        "rule_checklist": rules_result,
        "sufficiency_score": suff_score,
        "recommendations": recommendations
    }
def chunk_text(text, max_words=500, overlap=50):
    """
    Split text into chunks for processing.
    - max_words: maximum words per chunk
    - overlap: words to repeat between chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap
    return chunks
# --- Advice detection ---
advice_keywords = [
    "what should i do", "next steps", "can i", "how do i proceed", "is it okay to", 
    "should i", "do i need to", "recommend", "advice", "guidance"
]

def is_advice_request(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in advice_keywords)


# --- Friendly / casual responses ---
GREETINGS = ["hello", "hi","hii" ,"hii levi","hey", "good morning", "good evening", "good afternoon"]
THANKS = ["thanks", "thank you", "thx", "ty"]
CASUAL = [
    "how are you", "what's up", "sup", "how's it going", "good day"
]

GREET_RESPONSES = [
    "Hey there! 👋 How can I assist you with legal research today?",
    "Hi! Ready to explore some legal cases?",
    "Hello! Need help with any legal documents?",
    "Hey! What legal query shall we dive into today?"
]

THANK_RESPONSES = [
    "You're welcome! Happy to help 😎",
    "Anytime! Let me know if you have more questions.",
    "No problem! Always here for your legal queries."
]

CASUAL_RESPONSES = [
    "I'm doing great! Ready to tackle some legal cases? ⚖️",
    "All good here! How can I assist you today?",
    "Feeling helpful as always 😄. What legal question do you have?"
]

def get_friendly_response(query):
    q = query.lower().strip()
    if len(q.split()) > 10:
        return None
    
    words = re.findall(r'\b\w+\b', q)
    
    if any(g in words for g in GREETINGS):
        return random.choice(GREET_RESPONSES)
    if any(t in words for t in THANKS):
        return random.choice(THANK_RESPONSES)
    if any(c in q for c in CASUAL):
        return random.choice(CASUAL_RESPONSES)
    return None


# --- Multiline input helper ---
def read_multiline_input(prompt=""):
    print(prompt)
    print("(Paste your text. Press ENTER twice to finish.)")
    lines = []
    empty_line_count = 0
    
    while True:
        try:
            line = input()
        except EOFError:
            break
            
        if line.strip() == "":
            empty_line_count += 1
            if empty_line_count >= 2:
                break
            lines.append(line)
        else:
            empty_line_count = 0
            lines.append(line)
    
    return "\n".join(lines).strip()
def analyze_query_intent(query, last_document):
    q = query.lower().strip()

    # --- Translate intent ---
    if "translate" in q and "to " in q:
        return "translate"

    # --- Document QA intent ---
    if last_document and any(
        kw in q
        for kw in ["explain", "summarize", "meaning", "interpret", "overview", "about this document"]
    ):
        return "document_qa"

    # --- RAG search intent ---
    if any(kw in q for kw in ["what is", "who is", "define", "act", "law", "section", "article"]):
        return "rag_search"

    # --- Default ---
    return "general"
def is_out_of_context(query, faiss_results, document):
    # Mark out-of-context if no doc loaded, or search results have near-zero similarity
    if document is None:
        return True
    if not faiss_results or (faiss_results[0]['score'] < 0.1):  # Tune threshold as needed
        return True
    return False
