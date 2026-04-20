import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Levi Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Levi Legal AI Assistant")
st.subheader("Upload, Analyze & Understand Legal Documents Easily")


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.sidebar.header("Controls")
if st.sidebar.button("🔄 Reset System"):
    try:
        res = requests.post(f"{API_URL}/reset")
        if res.status_code == 200:
            st.sidebar.success(res.json().get("message", "System reset successfully!"))
        else:
            st.sidebar.error("Failed to reset system")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error: {e}")

-
st.sidebar.header("Features")
mode = st.sidebar.radio(
    "Choose Mode",
    ["Upload Document", "Chat / QA", "Document Verifier", "Briefings"]
)

if mode == "Upload Document":
    st.header("📄 Upload Your Legal Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Images (JPG/PNG)",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"]
    )
    if uploaded_file:
        with st.spinner("Uploading document..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            try:
                response = requests.post(f"{API_URL}/upload", files=files)
                result = response.json()
                st.success(result.get("message", "File uploaded successfully!"))
                st.info(f"Chunks: {result.get('chunks')}, Word count: {result.get('word_count')}")
            except Exception as e:
                st.error(f"⚠️ Upload failed: {e}")

elif mode == "Chat / QA":
    st.header("💬 Ask Questions About Your Document")
    query = st.text_area("Enter your question", height=100)
    if st.button("Ask"):
        if not query.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    response = requests.post(f"{API_URL}/chat", params={"query": query})
                    answer = response.json().get("answer")
                    st.markdown(f"**Answer:**\n\n{answer}")
                except Exception as e:
                    st.error(f"⚠️ Error connecting to API: {e}")


elif mode == "Document Verifier":
    st.header("📝 Document Verifier")
    if st.button("Run Verifier"):
        with st.spinner("Analyzing document..."):
            try:
                response = requests.get(f"{API_URL}/verifier")
                result = response.json()
                st.json(result)
            except Exception as e:
                st.error(f"⚠️ Error connecting to API: {e}")


elif mode == "Briefings":
    st.header("📑 Generate Document Briefings")
    if st.button("Generate Briefings"):
        with st.spinner("Generating briefings..."):
            try:
                response = requests.get(f"{API_URL}/briefings")
                briefings = response.json().get("briefings")
                st.json(briefings)
            except Exception as e:
                st.error(f"⚠️ Error connecting to API: {e}")


st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:gray;">
    ⚖️ Levi Legal AI Assistant — Hackathon Prototype
    </div>
    """,
    unsafe_allow_html=True,
)
