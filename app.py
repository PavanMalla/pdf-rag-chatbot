import streamlit as st

st.title("PDF RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

question = st.text_input("Ask a question")

if st.button("Submit"):
    if uploaded_file and question.strip():
        # Call your existing PDF retrieval and answer function here
        st.write("Your answer will appear here.")
    else:
        st.warning("Please upload a PDF and enter a question.")
