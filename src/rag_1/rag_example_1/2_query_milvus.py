from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from sklearn.preprocessing import normalize
import numpy as np

# -----------------------------
# ✅ Configurations
# -----------------------------
COLLECTION_NAME = "squad_rag"   # Milvus collection name
TOP_K = 3                       # Number of top results to retrieve
METRIC_TYPE = "L2"              # "IP" for cosine similarity, "L2" for Euclidean

# -----------------------------
# ✅ Step 1: Connect to Milvus
# -----------------------------
connections.connect("default", host="localhost", port="19530")

# -----------------------------
# ✅ Step 2: Load the collection
# -----------------------------
collection = Collection(COLLECTION_NAME)
collection.load()

# -----------------------------
# ✅ Step 3: User input query
# -----------------------------
query_text = input("Enter your question: ")

# -----------------------------
# ✅ Step 4: Encode the query
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
query_vector_raw = embedder.encode([query_text])  # shape: (1, dim)

# -----------------------------
# ✅ Step 5: Normalize only if using IP (cosine simulation)
# -----------------------------
if METRIC_TYPE == "IP":
    print("⚙️  Using metric IP: applying normalization for cosine similarity...")
    query_vector = normalize(query_vector_raw)[0]
else:
    print("⚙️  Using metric L2: skipping normalization...")
    query_vector = query_vector_raw[0]

# Print vector norm
vector_norm = np.linalg.norm(query_vector)
print(f"📏 Final query vector norm: {vector_norm:.6f}")

# -----------------------------
# ✅ Step 6: Search in Milvus
# -----------------------------
search_params = {
    "metric_type": METRIC_TYPE,
    "params": {"nprobe": 10}
}

results = collection.search(
    data=[query_vector.tolist()],
    anns_field="embedding",
    param=search_params,
    limit=TOP_K,
    output_fields=["context"]
)

# -----------------------------
# ✅ Step 7: Display results
# -----------------------------
print(f"\n🔍 Top {TOP_K} retrieved contexts for: \"{query_text}\"\n")
for i, hit in enumerate(results[0]):
    print(f"Result {i+1} (score: {hit.distance:.4f})")
    print(hit.entity.get("context")[:500].strip())  # show first 500 characters
    print("-" * 80)
