import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ingest import all_chunks

if not all_chunks:
    raise ValueError("No chunks found. Make sure ingest.py loaded English PDFs correctly.")

# Modell initialisieren
model = SentenceTransformer('all-MiniLM-L6-v2')

# Alle Chunks in Vektoren umwandeln
print("Encoding chunks into embeddings...")
vectors = [model.encode(chunk.page_content) for chunk in all_chunks]
print(f"Created embeddings for {len(vectors)} chunks")

# FAISS Index erstellen
dim = len(vectors[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(vectors).astype("float32"))

print("FAISS Vectorstore ready")

# Ordner für Vectorstore prüfen oder erstellen
os.makedirs("vectorstore", exist_ok=True)

# Speichern im vectorstore-Ordner
faiss.write_index(index, "vectorstore/vectorstore.index")
with open("vectorstore/chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("Saved vectorstore and chunks in 'vectorstore/'")