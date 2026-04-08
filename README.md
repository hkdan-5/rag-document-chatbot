# AI Document Chatbot (RAG)

This project demonstrates a Retrieval Augmented Generation (RAG) system
that answers questions about company reports using vector search.

## Architecture

Documents (PDF)

      ↓
      
Chunking

      ↓
      
Embeddings (Sentence Transformers)

      ↓
      
FAISS Vector Search

      ↓
      
Relevant Text Chunks

      ↓
      
Displayed in Streamlit Web App

## Project Structure

rag-chatbot

│

├── data

├── vectorstore

├── ingest.py

├── vectorstore.py

├── app.py

└── requirements.txt

## How it works

1. `ingest.py`
   - Loads PDFs
   - Splits text into chunks

2. `vectorstore.py`
   - Converts chunks into embeddings
   - Creates FAISS vector index

3. `app.py`
   - Streamlit interface
   - User asks question
   - System retrieves relevant text chunks

## Run the project

Install dependencies


pip install -r requirements.txt


Create vector database


python ingest.py
python vectorstore.py


Run the app


streamlit run app.py


## Example Questions

- What risks are mentioned in the report?
- What sustainability initiatives are described?
- What are the main financial highlights?
