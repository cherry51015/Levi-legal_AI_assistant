# ⚖️ Levi — Legal AI Assistant

An AI-powered assistant for **legal document intelligence**. Levi lets lawyers, paralegals, and businesses upload legal documents and interact with them through **semantic Q&A, document verification, legal briefings, and precedent search** — grounded in a 153K+ clause legal corpus, with strict guardrails against fabricated legal advice.

The system combines **LLM reasoning (Gemini)** with a **hybrid BM25 + dense retrieval pipeline** and **rule-based compliance checks** to help legal professionals save time without sacrificing correctness.

---

## 🚀 Features

![Features Screenshot](https://raw.githubusercontent.com/cherry51015/Levi-legal_AI_assistant/main/data/samples/Screenshot%202026-02-08%20000112.png)

* **📂 Document Upload & Analysis** — Upload legal files (PDF, DOCX, TXT, or scanned images via OCR) and run AI-powered verification, clause extraction, and compliance checks.
* **💬 Chat with Your Document** — Ask document-specific questions in plain language; the AI retrieves context-aware answers grounded in the uploaded text.
* **🔎 Hybrid Precedent Search** — BM25 keyword search + dense FAISS retrieval, fused via reciprocal-rank ensembling, over a 153K+ clause legal corpus.
* **🧭 Adaptive Query Routing** — General/summary/translation queries are answered directly by the LLM for low latency; document-specific claims are routed through retrieval and grounded against source text, with an explicit refusal when nothing relevant is found. *(see [Query Routing & Guardrails](#-query-routing--guardrails) for current implementation status)*
* **🌍 Multi-Language Support** — Ask questions in different languages, including OCR support for 9 Indian languages, and receive translated responses.
* **🤖 General Knowledge Mode** — Ask general legal or non-legal questions even without an uploaded document.
* **⚖️ Safe by Design** — The model is instructed never to give legal advice or invent facts not present in source documents; enforced via prompt-level constraints and phrase-based advice detection (see [Guardrails](#-query-routing--guardrails) for what's currently rule-based vs. model-enforced).
* **📝 Brief Mode** — Generates structured JSON briefings: metadata, summary, key sections, obligations, risks.
* **🔍 Document Verifier** — Rule-based compliance checklist (signatures, dates, parties, jurisdiction) plus corpus-wide precedent matching for each clause.
* **⚡ Dual Interface** — FastAPI backend (`main.py`) for programmatic/production use, plus a CLI (`llm.py`) with fuller query-routing logic, and a Streamlit frontend (`app.py`).

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                        │
│                         app.py                                │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                          │
│                         main.py                               │
│                                                               │
│  Endpoints:                                                   │
│  • /upload                                                    │
│  • /chat                                                      │
│  • /verifier                                                  │
│  • /briefings                                                 │
└───────────────┬───────────────────────┬───────────────────────┘
                │                       │
                │                       │
                ▼                       ▼
      ┌───────────────────┐    ┌───────────────────┐
      │      llm.py       │    │    verifier.py    │
      │───────────────────│    │───────────────────│
      │ • ask_gemini()    │    │ • Rule checks     │
      │ • Gemini 1.5 Flash│    │ • Hybrid          │
      │                   │    │   precedent search│
      └─────────┬─────────┘    └─────────┬─────────┘
                │                        │
                └────────────┬───────────┘
                             │
                             │
                ┌────────────▼────────────┐
                │     briefings.py        │
                │─────────────────────────│
                │ • run_brief_mode()      │
                │ • Structured JSON       │
                └────────────┬────────────┘
                             │
                             ▼
         ┌─────────────────────────────────────┐
         │       hybrid_retriever.py           │
         │─────────────────────────────────────│
         │ • BM25 Retrieval                    │
         │ • Dense FAISS Retrieval             │
         │ • LangChain Ensemble Retriever      │
         └──────────────────┬──────────────────┘
                            │
                            ▼
         ┌─────────────────────────────────────┐
         │        Vector Index Storage         │
         │─────────────────────────────────────│
         │ data/faiss_index.bin                │
         │ 153K+ Legal Clauses                 │
         └─────────────────────────────────────┘
         
```

**Note on the CLI vs. the API:** `llm.py` run standalone (`python llm.py`) has the fullest query-routing logic — it distinguishes document-QA, translation, general-knowledge, and corpus-RAG intents via `analyze_query_intent()` and `is_out_of_context()`. The FastAPI `/chat` endpoint currently sends the entire uploaded document as context on every call, without that routing layer. Wiring the same routing into `/chat` is the next planned step — see [Roadmap](#-roadmap).

### Key Modules

| Module | Responsibility |
|---|---|
| `llm.py` | `ask_gemini(query, document)` — context-aware Q&A, chunked processing for long documents, CLI entrypoint with full intent routing |
| `main.py` | FastAPI app — `/upload`, `/chat`, `/verifier`, `/briefings`, `/health` |
| `verifier.py` | `run_document_verifier()` — rule-based compliance checklist + hybrid precedent search against the global corpus |
| `briefings.py` | `run_brief_mode()` — structured JSON legal briefings (metadata, summary, obligations, risks) |
| `hybrid_retriever.py` | `hybrid_search()` — BM25 + dense FAISS retrieval fused via LangChain's `EnsembleRetriever` |
| `utils/file_loader.py` | PDF/DOCX/TXT/image ingestion, with multilingual OCR (Tesseract + `langdetect`) |
| `utils/helpers.py` | Chunking, friendly-response handling, advice-request detection, intent classification |
| `rules.py` | Standalone rule-check functions (signatures, dates, parties, jurisdiction) |

---

## 🧠 RAG Pipeline Deep Dive

**Chunking strategy:** Documents are split via a sliding window — `max_words=500`, `overlap=50` (see `chunk_text()` in `utils/helpers.py`). The 50-word overlap preserves context across chunk boundaries so clauses split mid-sentence aren't lost during retrieval or Q&A.

**Embedding model:** Google's `models/embedding-001` (768-dimensional), used with `task_type="retrieval_document"` when indexing the corpus and `task_type="retrieval_query"` at query time — this asymmetric embedding mode is Gemini's recommended setup for retrieval tasks, rather than embedding queries and documents identically.

**Corpus:** 153K+ legal clauses, indexed with `faiss.IndexFlatIP` over L2-normalized embeddings (cosine similarity via inner product). The corpus was curated from three legal datasets: **CUAD (Contract Understanding Atticus Dataset)** for commercial contract clauses, **LexGLUE** for diverse legal NLP benchmarks spanning multiple legal domains, and **ILTUR**, providing additional legal documents and statutory text. The combined corpus covers contracts, legislation, and other legal documents across multiple legal domains, enabling broad retrieval for legal question answering.

**Hybrid retrieval:** Dense FAISS search and a BM25 keyword index (`rank_bm25`, via LangChain's `BM25Retriever`) are combined through `EnsembleRetriever` with equal weighting (0.5 / 0.5). This means an exact statutory term or case number (BM25's strength) and a semantically similar but differently worded clause (dense embeddings' strength) can both surface for the same query — neither retrieval mode alone reliably covers both cases in legal text, where exact terminology often matters as much as meaning.

---

## 🧭 Query Routing & Guardrails

Levi is deliberately **not** a single retrieve-then-generate pipeline. Legal answers carry real consequences for getting something wrong, so retrieval is applied selectively based on where correctness risk actually lives:

* **General / summary / translation queries** (e.g. "summarize this," "what does GDPR mean," "translate to Hindi") are answered directly by Gemini — no retrieval overhead, since the risk of a misleading answer on this class of query is low and latency matters more.
* **Document-specific claims** (e.g. "what's the notice period in this contract") are routed through hybrid retrieval against the source document, and the response is grounded strictly in retrieved text.
* **Advice-seeking queries** ("what should I do," "can I sue for this") are intercepted by keyword-based detection (`is_advice_request()`) and redirected to consult a licensed professional, rather than answered at all.
* **Out-of-context queries** relative to a loaded document are flagged (`is_out_of_context()`) so the response is clearly marked as general knowledge, not a claim about the user's document.

---

## 🧪 Evaluation Framework

Answer quality is measured with **rubric-based LLM-as-judge scoring**, not just spot-checking:

* Each answer is scored 1–5 on **faithfulness** (does it only state facts present in the source text?) and **completeness** (does it cover everything relevant?), plus a boolean **fabrication flag**.
* The judge model (Groq, `llama-3.3-70b-versatile`) is deliberately different from the answering model (Gemini) — using the same model to both answer and grade its own output is a documented source of self-preference bias in LLM-as-judge setups.
* **Answer relevance** (the headline metric) = `((mean(faithfulness) + mean(completeness)) / 2) / 5 × 100`, computed across a curated query set spanning faithfulness, completeness, and no-fabrication test cases.
* Latency is captured per-query client-side (`time.perf_counter()` around each generation call) — no external monitoring dependency.

Eval harness files: `eval_dataset_template.jsonl` (schema + examples), `rubric.py` (judge logic), `run_eval.py` (runner, produces `eval_report.json` + `eval_summary.md`).

---

## ⚙️ Installation

```bash
git clone https://github.com/cherry51015/Levi-legal_AI_assistant.git
cd Levi-legal_AI_assistant

python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
pip install langchain langchain-community rank_bm25 groq   # hybrid retrieval + eval
```

**Environment variables (`.env`):**
```
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key      # eval judge only, free tier
```

---

## ▶️ Running the App

**Streamlit UI:**
```bash
streamlit run app.py
```
Live demo: **[agent-translator-mgtwfgmztybxyfkw9pbddx.streamlit.app](https://agent-translator-mgtwfgmztybxyfkw9pbddx.streamlit.app/)**

**FastAPI backend (for programmatic access):**
```bash
uvicorn main:app --reload
```
Explore endpoints at `http://127.0.0.1:8000/docs`

**CLI (fullest query-routing logic):**
```bash
python llm.py
```

**Run evals:**
```bash
python run_eval.py eval_dataset.jsonl
```

### 📎 More demos & screenshots
[Full demo folder](https://drive.google.com/drive/folders/1lMOVf16aaa84_eu4Uv-0zCsvFVwEC49W?usp=sharing)

---

## 📌 Usage

1. **Upload a document** (PDF, DOCX, TXT, or image) via the sidebar or `/upload`.
2. Choose an action:
   * 🔍 **Run Verifier** → structured JSON: compliance checklist + hybrid-retrieved precedent clauses.
   * 📝 **Generate Briefing** → structured legal summary (metadata, obligations, risks).
   * 💬 **Chat** → ask questions directly about the document, or general legal/non-legal questions.
3. Review AI outputs, copy insights, or export JSON results.

---

## 🎥 Demo

**1. Upload Document**
[![Upload Demo](https://drive.google.com/uc?id=1HjetmWasqzB-6Mzava6sL9TtCqeEzmg9)](https://drive.google.com/file/d/1HjetmWasqzB-6Mzava6sL9TtCqeEzmg9/view?usp=sharing)

**2. Run Verifier**
[![Verifier Demo](https://drive.google.com/uc?id=1KWkSmLDbsruQ1lNCSdOtXh4bmWiHaNFr)](https://drive.google.com/file/d/1KWkSmLDbsruQ1lNCSdOtXh4bmWiHaNFr/view?usp=sharing)

**3. Generate Briefing**
[![Briefing Demo](https://drive.google.com/uc?id=1fuPIDwc-Wx-TgvKo_xqwtoWmodvTUTPH)](https://drive.google.com/file/d/1fuPIDwc-Wx-TgvKo_xqwtoWmodvTUTPH/view?usp=sharing)

**4. Chat with Document**
[![Chat Demo](https://drive.google.com/uc?id=1p4orvlVSL0TeBBOrdGmdO1QWRx7D_rwR)](https://drive.google.com/file/d/1p4orvlVSL0TeBBOrdGmdO1QWRx7D_rwR/view?usp=sharing)

👉 [View all demos](https://drive.google.com/drive/folders/1lMOVf16aaa84_eu4Uv-0zCsvFVwEC49W)

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI (production endpoints) + CLI
* **LLM:** Google Gemini (`gemini-1.5-flash`)
* **Retrieval:** FAISS (dense) + BM25 (`rank_bm25`) fused via LangChain `EnsembleRetriever`
* **Evaluation:** Groq (`llama-3.3-70b-versatile`) as rubric-based LLM judge
* **OCR/Multilingual:** Tesseract, `pdf2image`, `langdetect` (9 Indian languages supported)
* **Utilities:** PyPDF2, python-docx, JSON

---

## 📂 Project Structure

```
Levi-legal_AI_assistant/
│── app.py                       # Streamlit frontend
│── main.py                      # FastAPI backend (production endpoints)
│── llm.py                       # Gemini integration, CLI, full query routing
│── verifier.py                  # Compliance checks + hybrid precedent search
│── briefings.py                 # Structured legal briefing generation
│── rules.py                     # Standalone rule-check functions
│── hybrid_retriever.py          # BM25 + dense FAISS hybrid retrieval
│── run_eval.py                  # Eval harness runner
│── rubric.py                    # LLM-as-judge rubric scoring
│── eval_dataset_template.jsonl  # Eval query schema + examples
│── utils/
│   ├── file_loader.py           # Document ingestion + OCR
│   ├── helpers.py                # Chunking, intent detection, guardrail helpers
│   └── embeddings.py             # Embedding generation
│── data/
│   ├── faiss_index.bin          # 153K+ clause vector index
│   └── faiss_index.bin.meta.json # Clause IDs + text metadata
│── requirements.txt
│── README.md
```

---

## 🔧 Known Limitations / Engineering Notes

Being upfront about the current state, not just the target state:

* **BM25 index is rebuilt in memory on process startup**, not persisted — tokenizing 153K clauses takes a couple of minutes the first time a process runs. Fine for a single long-running server; would need to persist the index (pickle or a proper BM25 store) for serverless/cold-start deployments.
* **Long documents are processed in sequential chunks**, not in parallel, inside `ask_gemini()` — this is the main latency risk for large uploads (see [Evaluation](#-evaluation-framework)).

---

## 🔮 Roadmap

* [ ] Persist the BM25 index instead of rebuilding it on every process start.
* [ ] Consolidate duplicate rule-check implementations into a single source of truth.
* [ ] Export results (JSON/PDF briefings).
* [ ] Clause comparison across multiple documents.
* [ ] User authentication & document history.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📜 License

This project is licensed under the **MIT License**.

---

⚖️ Built to make legal work simpler, faster, and smarter — without ever guessing.
