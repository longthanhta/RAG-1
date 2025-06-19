# load_pdf_and_build_index.py

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path

# === Step 1: Extract PDF text ===
pdf_path = Path(__file__).parent / "doc/7106511906_RE225BE_QIG_REV1.0.0.pdf"
with pdfplumber.open(pdf_path) as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# === Step 2: Chunk text ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(text)

# === Step 3: Embed chunks ===
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# === Step 4: Save FAISS index and metadata ===
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings))

index_path = Path(__file__).parent / "faiss_index.index"
faiss.write_index(index, str(index_path))  # ← Fix: convert Path to string

# Save chunks (for mapping back during retrieval)
chunk_path = Path(__file__).parent /"text_chunks.pkl"
with open(chunk_path, "wb") as f:
    pickle.dump(chunks, f)

print("✅ Index built and saved.")
