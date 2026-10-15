
import os
import streamlit as st
import pymupdf
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about its content.")


# -----------------------------
# 2. Load embedding model once
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# 3. Read configuration
# -----------------------------
HF_TOKEN = (
    st.secrets.get("HF_TOKEN", "")
    or os.getenv("HF_TOKEN", "")
)

HF_MODEL = (
    st.secrets.get("HF_MODEL", "")
    or os.getenv("HF_MODEL", "")
)


# -----------------------------
# 4. Extract PDF text
# -----------------------------
def extract_text(pdf_bytes):
    pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if text:
                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

    return "\n\n".join(pages)


# -----------------------------
# 5. Create text chunks
# -----------------------------
def create_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# -----------------------------
# 6. RAG question answering
# -----------------------------
def answer_question(pdf_bytes, question):
    if not HF_TOKEN:
        return "Please configure HF_TOKEN in Streamlit Secrets."

    if not HF_MODEL:
        return "Please configure HF_MODEL in Streamlit Secrets."

    text = extract_text(pdf_bytes)

    if not text.strip():
        return (
            "No readable text was found in this PDF. "
            "It may be scanned and require OCR."
        )

    chunks = create_chunks(text)

    if not chunks:
        return "No usable text chunks were found in the PDF."

    model = load_embedding_model()

    # Create embeddings for PDF chunks
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Embed the question
    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # Retrieve relevant chunks
    k = min(3, len(chunks))
    scores, ids = index.search(query_vector, k)

    context = "\n\n".join(
        chunks[i] for i in ids[0] if i >= 0
    )

    if not context.strip():
        return "No relevant passages were found."

    # Call the hosted language model
    client = InferenceClient(
        model=HF_MODEL,
        token=HF_TOKEN
    )

    response = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied PDF context. "
                    "If the answer is missing, say you could not "
                    "find it in the PDF. Do not invent facts."
                )
            },
            {
                "role": "user",
                "content": (
                    f"PDF context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        max_tokens=400,
        temperature=0.1
    )

    return response.choices[0].message.content


# -----------------------------
# 7. Streamlit user interface
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

question = st.text_input(
    "Ask a question",
    placeholder="What is this document about?"
)

if st.button("Submit", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a PDF file.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Reading PDF and generating your answer..."
            ):
                pdf_bytes = uploaded_file.getvalue()
                answer = answer_question(
                    pdf_bytes,
                    question.strip()
                )

            st.subheader("Answer")
            st.markdown(answer)

        except Exception:
            st.error(
                "The request failed. Please check the app logs, "
                "PDF format, Hugging Face token, model availability "
                "and available memory."
            )
            st.exception(
                __import__("sys").exc_info()[1]
            )
