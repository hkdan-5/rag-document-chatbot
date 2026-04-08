import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Seitenlayout
st.set_page_config(
    page_title="AI Document Chatbot",
    layout="wide"
)

# Titel
st.title("📄 AI Document Chatbot")
st.caption("Retrieval Augmented Generation (RAG) with FAISS and Sentence Transformers")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This demo shows a Retrieval Augmented Generation (RAG) system. "
        "It searches company reports and retrieves relevant passages."
    )

    st.subheader("Documents")
    st.write("- Apple Annual Report")
    st.write("- Bertelsmann Annual Report")
    st.write("- Deutsche Telekom Annual Report")

# Prüfen ob Vectorstore existiert
if not os.path.exists("vectorstore/vectorstore.index") or not os.path.exists("vectorstore/chunks.pkl"):
    st.error("Vectorstore not found. Please run ingest.py and vectorstore.py first.")
    st.stop()

# Vectorstore laden
with open("vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

index = faiss.read_index("vectorstore/vectorstore.index")

# Embedding Modell
model = SentenceTransformer("all-MiniLM-L6-v2")

# Layout mit zwei Spalten
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Ask a question")

    query = st.text_input("Your question about the reports")

    st.write("Example questions:")

    example_questions = [
        "What risks are mentioned in the report for Apple?",
        "What sustainability initiatives are described?",
        "What are the main financial highlights?",
        "What strategy does the company describe for the future of Bertelsmann?"
    ]

    for q in example_questions:
        if st.button(q):
            query = q

with col2:
    st.header("Answer")

    if query:

        # Query embedding
        query_vec = model.encode(query)

        # Ähnlichkeitssuche
        k = 5
        D, I = index.search(np.array([query_vec]).astype("float32"), k=k)

        combined_text = ""

        for idx in I[0]:
            combined_text += chunks[idx].page_content + "\n"

        # Antwortbox
        st.success(combined_text[:800] + "...")

        st.subheader("Sources")

        # Quellen anzeigen
        for rank, (idx, score) in enumerate(zip(I[0], D[0])):

            chunk = chunks[idx]

            source = chunk.metadata.get("source", "Unknown document")
            page = chunk.metadata.get("page", "Unknown page")

            with st.expander(f"Source {rank+1} | Score: {score:.2f} | {source} | Page {page}"):

                preview = chunk.page_content[:500]

                # Highlighting der Suchbegriffe
                for word in query.split():
                    preview = preview.replace(word, f"**{word}**")

                st.markdown(preview + "...")