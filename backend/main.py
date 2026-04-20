from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
import shutil
import os
import uvicorn
import re
import json
import os

FAISS_META_PATH = "data/faiss_index.bin.meta.json"
if os.path.exists(FAISS_META_PATH):
    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        FAISS_DOCS = json.load(f)
else:
    FAISS_DOCS = {"ids": [], "texts": []}




# Import your tools

from llm import ask_gemini  # chat engine
from verifier import run_document_verifier
from briefings import run_brief_mode
from utils.helpers import chunk_text
from utils.embeddings import embed_texts
#from visualizer import run_visualizer
from utils.file_loader import load_document  # text extraction
import numpy as np
import faiss


app = FastAPI(title="Legal AI Assistant Prototype")



# Store globally
uploaded_doc_text: Optional[str] = None
uploaded_doc_chunks: Optional[list] = None
uploaded_doc_embeddings: Optional[np.ndarray] = None
uploaded_doc_index: Optional[faiss.IndexFlatL2] = None

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global uploaded_doc_text, uploaded_doc_chunks, uploaded_doc_embeddings, uploaded_doc_index

    # Save temp file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        uploaded_doc_text = load_document(temp_path)
    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Error loading document: {e}")
    finally:
        os.remove(temp_path)

    # ✅ Chunk and embed immediately
    uploaded_doc_chunks = chunk_text(uploaded_doc_text, max_words=500, overlap=50)
    uploaded_doc_embeddings = embed_texts(uploaded_doc_chunks).astype("float32")

    # ✅ Build FAISS index for this doc
    dim = uploaded_doc_embeddings.shape[1]
    uploaded_doc_index = faiss.IndexFlatL2(dim)
    uploaded_doc_index.add(uploaded_doc_embeddings)

    return {
        "message": f"✅ Document '{file.filename}' uploaded successfully!",
        "chunks": len(uploaded_doc_chunks),
        "word_count": len(uploaded_doc_text.split())
    }


# Chat endpoint (always available)

@app.post("/chat")
async def chat(query: str):
    global uploaded_doc_text
    context = uploaded_doc_text if uploaded_doc_text else None
    answer = ask_gemini(query, document=context, mode="chat")
    return {"query": query, "answer": answer}


# Document Verifier endpoint

@app.get("/verifier")
async def document_verifier():
    if not uploaded_doc_text or uploaded_doc_index is None:
        raise HTTPException(status_code=400, detail="⚠️ Upload a document first to run the verifier.")
    result = run_document_verifier(uploaded_doc_text, uploaded_doc_chunks, uploaded_doc_index, FAISS_DOCS)

    return result



# Briefings endpoint

@app.get("/briefings")
async def document_briefings():
    if not uploaded_doc_text:
        raise HTTPException(status_code=400, detail="⚠️ Upload a document first to run briefings.")
    brief_json = run_brief_mode("brief mode", uploaded_doc_text)
    return {"briefings": brief_json}





# Health check endpoint

@app.get("/health")
async def health_check():
    return {"status": "✅ Legal AI Assistant is up and running!"}

@app.get("/")
def root():
    return {"message": "Levi Legal AI API is running! Use /docs to explore endpoints."}

@app.post("/reset")
async def reset_system():
    global uploaded_doc_text, uploaded_doc_chunks, uploaded_doc_embeddings, uploaded_doc_index

    uploaded_doc_text = None
    uploaded_doc_chunks = None
    uploaded_doc_embeddings = None
    uploaded_doc_index = None

    return {"message": "✅ System reset successfully. All uploaded data cleared."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
