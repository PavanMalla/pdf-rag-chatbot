
import os
import fitz
import gradio as gr
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Load the embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Configure the hosted language model
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL = os.environ.get("HF_MODEL")

client = InferenceClient(
    model=HF_MODEL,
    token=HF_TOKEN
)


def extract_text(pdf_path):
    """Extract text from a PDF file."""
    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if text:
                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

    return "\n\n".join(pages)


def create_chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def answer_question(pdf_file, question):
    """Retrieve relevant PDF text and generate an answer."""
    if not pdf_file:
        return "Please upload a PDF file."

    if not question or not question.strip():
        return "Please enter a question."

    if not HF_TOKEN or not HF_MODEL:
        return "Configure HF_TOKEN and HF_MODEL in Space Settings."

    try:
        # 1. Extract text from the PDF
        text = extract_text(pdf_file)

        if not text.strip():
            return (
                "No extractable text was found. "
                "This PDF may be scanned and need OCR."
            )

        # 2. Split text into chunks
        chunks = create_chunks(text)

        # 3. Create embeddings
        embeddings = embedding_model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        # 4. Build FAISS vector index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # 5. Embed the user's question
        query_vector = embedding_model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        # 6. Retrieve the most relevant chunks
        k = min(3, len(chunks))
        scores, ids = index.search(query_vector, k)

        context = "\n\n".join(
            chunks[i] for i in ids[0] if i >= 0
        )

        # 7. Generate an answer using the LLM
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied PDF context. "
                        "If the answer is not in the context, say "
                        "you could not find it in the PDF. "
                        "Do not invent facts."
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

    except Exception as error:
        print(f"Application error: {error}")
        return (
            "The request failed. Check the Space logs, "
            "API token, model availability and PDF format."
        )


demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.File(
            label="Upload your PDF",
            file_types=[".pdf"],
            type="filepath"
        ),
        gr.Textbox(
            label="Ask a question",
            placeholder="What is this document about?"
        )
    ],
    outputs=gr.Markdown(label="Answer"),
    title="PDF RAG Chatbot",
    description=(
        "Upload a PDF and ask questions based on its content."
    )
)

if __name__ == "__main__":
    demo.launch()
