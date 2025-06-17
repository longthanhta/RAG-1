from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
vecs = model.encode(["Paris", "London", "Berlin"])

for i, v in enumerate(vecs):
    print(f"Vector {i+1} norm: {np.linalg.norm(v):.6f}")
