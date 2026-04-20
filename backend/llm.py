from utils.helpers import read_multiline_input, is_advice_request, get_friendly_response,analyze_query_intent,is_out_of_context
from verifier import run_document_verifier
from briefings import run_brief_mode
from dotenv import load_dotenv
import faiss
import json
import numpy as np
import google.generativeai as genai
import os
import re
import random
from PIL import Image
import docx
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pdf2image import convert_from_path
from langdetect import detect_langs

from utils.helpers import chunk_text

# CONFIG 
INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/faiss_index.bin.meta.json"
TOP_K = 5
GEMINI_MODEL = "gemini-1.5-flash"


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ No Gemini API key found.")
genai.configure(api_key=GEMINI_API_KEY)


# Multi-Language OCR Helpers

INDIAN_LANGUAGES = ["hin", "tam", "tel", "ben", "mar", "guj", "kan", "mal", "pan"]

def detect_languages(text_chunk):
    try:
        langs = detect_langs(text_chunk)
        langs_sorted = sorted(langs, key=lambda x: x.prob, reverse=True)
        tess_langs = []
        mapping = {"hi":"hin","ta":"tam","te":"tel","bn":"ben","mr":"mar",
                   "gu":"guj","kn":"kan","ml":"mal","pa":"pan"}
        for l in langs_sorted:
            if l.lang == "en":
                tess_langs.append("eng")
            elif l.lang in mapping:
                tess_langs.append(mapping[l.lang])
        return "+".join(set(tess_langs)) if tess_langs else "eng"
    except:
        return "eng"

def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    text = ""
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            for page in reader.pages:
                text += page.extract_text() or ""
        except:
            pass
        if not text.strip():
            images = convert_from_path(path)
            for img in images:
                tess_lang = detect_languages(pytesseract.image_to_string(img, lang="eng"))
                text += pytesseract.image_to_string(img, lang=tess_lang)
    elif ext in [".doc", ".docx"]:
        doc = docx.Document(path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif ext in [".jpg", ".jpeg", ".png"]:
        img = Image.open(path)
        tess_lang = detect_languages(pytesseract.image_to_string(img, lang="eng"))
        text = pytesseract.image_to_string(img, lang=tess_lang)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return text.strip()


# FAISS + Embeddings

def embed_texts(texts):
    if isinstance(texts, str):
        texts = [texts]
    response = genai.embed_content(
        model="models/embedding-001",
        content=texts,
        task_type="retrieval_query"
    )
    embeddings = []
    if isinstance(response, dict) and "embedding" in response:
        embeddings.append(response["embedding"])
    elif isinstance(response, list):
        for item in response:
            if isinstance(item, dict) and "embedding" in item:
                embeddings.append(item["embedding"])
            elif isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, dict) and "embedding" in sub_item:
                        embeddings.append(sub_item["embedding"])
    arr = np.array(embeddings, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(len(embeddings), -1)
    return arr

def embed_document_chunks(doc_text):
    """Split document into chunks and embed each chunk"""
    chunks = chunk_text(doc_text, max_words=500, overlap=50)
    embeddings = [embed_texts(c) for c in chunks]
    return chunks, embeddings

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def search(index, meta, query, k=TOP_K):
    q_emb = embed_texts(query)
    scores, indices = index.search(q_emb, k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        case_id = meta["ids"][idx]
        case_text = meta["texts"][idx]
        results.append({"id": case_id, "score": float(score), "text": case_text})
    return results
def ask_gemini(query, document=None, mode="chat", context_type=None):
    """
    Handles chunked documents for long input texts.
    If document is large, splits into chunks and processes each sequentially.
    Combines outputs into a single structured answer.
    """
    # Friendly / casual check 
    friendly_resp = get_friendly_response(query)
    if friendly_resp:
        return friendly_resp
    
    if not document or len(document.split()) <= 500:
        # Small doc or text → process normally
        return _ask_gemini_single(query, document, mode, context_type)
    
    # Large document → chunk processing
    from utils.helpers import chunk_text
    chunks = chunk_text(document, max_words=500, overlap=50)
    combined_answers = []

    for i, chunk in enumerate(chunks):
        answer = _ask_gemini_single(
            query,
            chunk,
            mode=mode,
            context_type=context_type
        )
        combined_answers.append({
            "chunk_index": i,
            "chunk_preview": chunk[:100] + "...",
            "answer": answer
        })

    # Optionally combine summaries for whole-doc context
    final_answer = "\n\n".join([f"Chunk {c['chunk_index']+1}:\n{c['answer']}" for c in combined_answers])
    return final_answer


def _ask_gemini_single(question, retrieved=None, mode="chat", context_type=None):
    # Single chunk Gemini call
    question_lower = question.lower()
    needs_legal_terms = False
    prompt_sections = []
    whole_doc_indicators = [
        "what is this about", "summarize", "explain this", "overview", "gist",
        "summary", "explain document", "what does this document mean",
        "key terms", "explain key terms", "terms and conditions", "legal terms",
        "interpret document", "interpret this", "document summary","summarize this document"
    ]
    if any(ind in question_lower for ind in whole_doc_indicators) or context_type == "whole_doc":
        needs_legal_terms = True
        prompt_sections.append("- First give a clear and concise plain-language summary (150 words max).")
        prompt_sections.append("- Then, in a separate section, explain ALL key legal terms or clauses present, using plain English.")
    prefix = "" if context_type != "out_of_context" else "⚠️ Note: This question is unrelated to the loaded document. I will answer briefly using my general knowledge base. \n"
    requested_language = None
    languages = ["hindi", "tamil", "telugu", "kannada", "marathi", "bengali", "french", "german", "spanish"]
    for lang in languages:
        if f"in {lang}" in question_lower or f"to {lang}" in question_lower:
            requested_language = lang
            break
    base_context = f"Document Content:\n{retrieved}\n\n" if retrieved else ""
    prompt = f"{prefix}You are a Legal AI Assistant. When asked to translate, translate the entire answer into the requested language. Always give structured, user-friendly answers.\n{chr(10).join(prompt_sections)}\nIf the user's question is not about the document, keep the answer short and indicate it's answered from general knowledge, not the document.\nNote:\nYou MUST NOT give legal advice, recommendations, or next step guidance.\nIf the user asks any question seeking advice or instructions, politely respond:\n\"I am not qualified to give legal advice. Please consult a qualified lawyer.\"\n{base_context}\nUser Question: {question}\n"
    if requested_language:
        prompt += f"\nTranslate the entire answer into {requested_language}."
    try:
        response = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {e}"


# Friendly / Casual Chat
def main():
    print("⚖️ Legal AI Assistant is starting...")
    print("🤝 Hello! I'm here to help you understand legal documents in a clear, user-friendly way.")
    print("📂 Load a document by typing its file path, or ask questions directly.")
    print("💡 Try: 'summarize this document', 'explain in Hindi', 'key terms of contract'.")

    if not os.path.exists(INDEX_PATH):
        print(f"❌ FAISS index not found at {INDEX_PATH}. Please build/load it first.")
        return

    index, meta = load_index()
    print("✅ Assistant is ready! Paste a query, document, or 'exit' to quit.")

    last_document = None
    last_answer = None

    while True:
        query = read_multiline_input("\n🔎 Enter your query, text, or file path (or 'exit'):")
        if not query:
            continue
        if query.lower().strip() == "exit":
            print("👋 Goodbye! Stay legally informed.")
            break

        # Heuristic: treat long pasted text (not a file path) as document content
        if len(query.split()) > 100 and not os.path.isfile(query.strip()):
            last_document = query
            print(f"📄 Loaded document: (pasted text) ({len(last_document.split())} words)")
            last_answer = None
            continue


        # Friendly response
        friendly = get_friendly_response(query)
        if friendly:
            print(f"\n🤖 {friendly}")
            continue

        # File path loading
        if os.path.isfile(query.strip()):
            try:
                last_document = load_document(query.strip())
                print(f"📄 Loaded document: {os.path.basename(query.strip())} ({len(last_document.split())} words)")
                last_answer = None
                continue
            except Exception as e:
                print(f"❌ Error loading file: {e}")
                continue

        # Advice detection and blocking
        if is_advice_request(query):
            print("⚠️ For specific legal advice or next steps, please consult a qualified legal professional.")
            continue

     
        # Brief Mode (chunked)
       
        if query.lower().strip() == "brief mode":
            if last_document:
                print("📑 Generating detailed structured briefings (JSON)...")
                # chunked processing
                chunks = chunk_text(last_document)
                combined_brief = []
                for i, chunk in enumerate(chunks):
                    brief = run_brief_mode(query, chunk)
                    if brief:
                        combined_brief.append({"chunk_index": i, "brief": brief})
                print("\n🤖 JSON Briefing (chunked):\n")
                print(json.dumps(combined_brief, indent=2))
            else:
                print("⚠️ No document loaded. Please load a document before using Brief Mode.")
            continue

        # Document Verifier (chunked)
 
        elif query.lower().strip() == "document verifier":
            if last_document:
                print("📋 Running Document Verifier...")
                chunks = chunk_text(last_document)
                combined_results = []
                for i, chunk in enumerate(chunks):
                    results = run_document_verifier(chunk, embedding_fn=embed_text)
                    combined_results.append({"chunk_index": i, "results": results})
                print("✅ Verification Results (chunked):")
                print(json.dumps(combined_results, indent=2))
            else:
                print("⚠️ No document loaded. Please load a document first.")
            continue

        # Detect intent
   
        intent = analyze_query_intent(query, last_document)

        # Translation
    
        if intent == "translate":
            if last_document and last_document.strip():
                answer = ask_gemini(query, last_document, mode="translate", context_type="whole_doc")
                print("\n🌐 Translation:\n", answer)
                last_answer = answer
            else:
                print("⚠️ No document loaded to translate. Load a text, PDF, or image first.")
                continue


        # Whole-document QA / Summaries (chunked)
    
        elif intent == "document_qa" and last_document:
            answer = ask_gemini(query, last_document, mode="doc_qa", context_type="whole_doc")
            print("\n🤖 Document Summary & Key Terms Explained:\n", answer)
            last_answer = answer

       
        # General RAG search / out-of-context QA
       
        else:
            retrieved = search(index, meta, query)
            out_of_context = is_out_of_context(query, retrieved, last_document)

            if out_of_context:
                answer = ask_gemini(query, None, mode="rag", context_type="out_of_context")
                print("\n⚠️ Out-of-context: This query is not directly related to the loaded document, but here is a short response from general knowledge base:\n")
                print("\n🤖 Gemini Answer:\n", answer)
                last_answer = answer
            else:
                answer = ask_gemini(query, last_document, mode="rag")
                print("\n🤖 Gemini Answer:\n", answer)
                last_answer = answer

if __name__ == "__main__":
    main()

