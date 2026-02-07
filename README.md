
# ⚖️Levi- Legal AI Assistant

An AI-powered assistant for **legal document intelligence**. This project enables lawyers, paralegals, and businesses to upload legal documents and interact with them using advanced NLP features such as **semantic Q&A, document verification, legal briefings, and clause extraction**.

The system combines **LLM-powered reasoning** with **custom document pipelines** to help professionals save time, reduce errors, and get actionable insights from large legal texts.

Your multilingual, safe, and intelligent legal document companion — upload files, ask questions, or chat freely with general knowledge, all while ensuring no fabricated legal advice

---

## 🚀 Features

![Features Screenshot](https://raw.githubusercontent.com/cherry51015/Levi-legal_AI_assistant/main/data/samples/Screenshot%202026-02-08%20000112.png)


* **📂 Document Upload & Analysis** – Upload legal files (PDF, DOCX, TXT) and run AI-powered verification, clause extraction, and compliance checks.
* **💬 Chat with Your Document** – Ask **document-specific questions** in plain language; the AI retrieves context-aware answers.
* **🌍 Multi-Language Support** – Ask questions in different languages and receive meaningful responses.
* **🤖 General Knowledge Mode** – Even without uploading a document, you can ask **general legal or non-legal questions**, and the model will respond using its world knowledge.
* **⚖️ Safe by Design** – The model never **divises, invents, or fabricates legal advice**; it sticks to document facts or general knowledge.
* **📝 Brief Mode** – Generate concise summaries or structured briefings for faster case prep.
* **🔍 Document Verifier** – AI-powered clause and compliance verification, returning structured outputs (JSON).
* **⚡ Smooth & Interactive UI** – Modern Streamlit frontend for easy usage.

---


## 🏗️ Architecture

```
Streamlit UI
   |
   |--> File Loader (PDF/TXT/DOCX → Text)
   |--> Verifier Pipeline (run_document_verifier)
   |--> Brief Mode Pipeline (run_brief_mode)
   |--> Chat (ask_gemini with context + doc_text)
   |
   --> Backend Models (Google Gemini / FAISS index / Custom pipelines)
```

### Key Modules:

* **`llm.py`** → `ask_gemini(prompt, document)` handles context-aware legal Q&A.
* **`verifier.py`** → `run_document_verifier(text)` runs compliance + legal checks.
* **`briefings.py`** → `run_brief_mode(mode, text)` generates concise legal briefings.
* **`utils/file_loader.py`** → Handles PDF, DOCX, and TXT ingestion.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/legal-ai-assistant.git
cd legal-ai-assistant
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On macOS/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

To launch the Streamlit UI:

```bash
streamlit run app.py
```

Open in browser at **[https://agent-translator-mgtwfgmztybxyfkw9pbddx.streamlit.app/]**(https://agent-translator-mgtwfgmztybxyfkw9pbddx.streamlit.app/)

---

## 📌 Usage

1. **Upload a document** (PDF, DOCX, TXT) from the sidebar.
2. Choose an action:

   * 🔍 Run Verifier → Get structured JSON results of compliance checks.
   * 📝 Generate Briefing → AI-generated legal summary.
   * 💬 Chat → Ask questions directly about the document.
3. Review AI outputs, copy insights, or export JSON results.

---

## 🎥 Demo

### 1. Upload Document

[![Upload Demo](https://drive.google.com/uc?id=1HjetmWasqzB-6Mzava6sL9TtCqeEzmg9)](https://drive.google.com/file/d/1HjetmWasqzB-6Mzava6sL9TtCqeEzmg9/view?usp=sharing)

---

### 2. Run Verifier

[![Verifier Demo](https://drive.google.com/uc?id=1KWkSmLDbsruQ1lNCSdOtXh4bmWiHaNFr)](https://drive.google.com/file/d/1KWkSmLDbsruQ1lNCSdOtXh4bmWiHaNFr/view?usp=sharing)

---

### 3. Generate Briefing

[![Briefing Demo](https://drive.google.com/uc?id=1fuPIDwc-Wx-TgvKo_xqwtoWmodvTUTPH)](https://drive.google.com/file/d/1fuPIDwc-Wx-TgvKo_xqwtoWmodvTUTPH/view?usp=sharing)

---

### 4. Chat with Document

[![Chat Demo](https://drive.google.com/uc?id=1p4orvlVSL0TeBBOrdGmdO1QWRx7D_rwR)](https://drive.google.com/file/d/1p4orvlVSL0TeBBOrdGmdO1QWRx7D_rwR/view?usp=sharing)

---

### 🎬 Full Demo Folder

👉 [View All Demos](https://drive.google.com/drive/folders/1lMOVf16aaa84_eu4Uv-0zCsvFVwEC49W)

---


## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Backend:** Python (FastAPI-ready endpoints)
* **LLM Integration:** Google Gemini API
* **Embeddings / Indexing:** FAISS
* **Utilities:** PyPDF2, python-docx, JSON, Pandas

---

## 📂 Project Structure

```
legal-ai-assistant/
│── app.py                 # Streamlit frontend
│── llm.py                 # LLM integration (Gemini, embeddings)
│── verifier.py            # Document verification pipeline
│── briefings.py           # Legal briefing pipeline
│── utils/
│   └── file_loader.py     # Document ingestion utilities
│── data/
│   └── faiss_index.bin    # Vector index for semantic search
│── assets/
│   ├── screenshots/       # Static UI screenshots
│   └── demo/              # Demo GIFs
│── requirements.txt
│── README.md
```

---

## 🔮 Roadmap

* [ ] Export results (JSON/PDF briefings).
* [ ] Add **clause comparison across multiple documents**.
* [ ] Integrate **legal precedent search** with multi-doc context.
* [ ] Deploy as a **FastAPI backend + Streamlit frontend** for production use.
* [ ] User authentication & document history.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

⚖ Built with AI to make legal work **simpler, faster, and smarter**.

---

