from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
import numpy as np

# Config
COLLECTION_NAME = "squad_rag"
TOP_K = 3

# 1. Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# 2. Load collection
collection = Collection(COLLECTION_NAME)
collection.load()

# 3. Input your question
query_text = input("Enter your question: ")

# 4. Encode using the same embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
query_vector = embedder.encode(query_text).tolist()

# 5. Search Milvus
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=TOP_K,
    output_fields=["context"]
)

# 6. Show results
print(f"\nTop {TOP_K} retrieved contexts for: \"{query_text}\"\n")
for i, hit in enumerate(results[0]):
    print(f"Result {i+1} (distance: {hit.distance:.4f}):")
    print(hit.entity.get("context")[:500], "\n" + "-"*80)
