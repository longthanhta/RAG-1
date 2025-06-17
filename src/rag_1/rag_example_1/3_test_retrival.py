from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.preprocessing import normalize
import hashlib
from tqdm import tqdm

# ----------------------------
# ✅ Configuration
# ----------------------------
COLLECTION_NAME = "squad_rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
METRIC_TYPE = "IP"     # "L2" for Euclidean, "IP" (with normalization) for Cosine
TOP_K = 3
NUM_SAMPLES = 100      # Number of SQuAD samples to evaluate

# ----------------------------
# ✅ Milvus Connection & Load
# ----------------------------
connections.connect("default", host="localhost", port="19530")
collection = Collection(COLLECTION_NAME)
collection.load()

# ----------------------------
# ✅ Load Dataset and Model
# ----------------------------
dataset = load_dataset("squad", split=f"train[:{NUM_SAMPLES}]")
embedder = SentenceTransformer(EMBEDDING_MODEL)

# ----------------------------
# ✅ Helper: Hash context to skip duplicates
# ----------------------------
def short_hash(text):
    return hashlib.md5(text.encode()).hexdigest()[:8]

# ----------------------------
# ✅ Evaluation Loop
# ----------------------------
correct = 0

for i, sample in tqdm(enumerate(dataset)):
    question = sample["question"]
    answer = sample["answers"]["text"][0].lower().strip()

    # Embed the question
    query_vec = embedder.encode([question])

    # Normalize for cosine similarity (only for IP)
    if METRIC_TYPE == "IP":
        from sklearn.preprocessing import normalize
        query_vec = normalize(query_vec)

    query_vec = query_vec[0].tolist()  # convert to flat list

    # Set search parameters based on metric type
    search_params = {
        "metric_type": METRIC_TYPE,
        "params": {"nprobe": 10}
    }

    # Perform vector search
    results = collection.search(
        data=[query_vec],
        anns_field="embedding",
        param=search_params,
        limit=TOP_K,
        output_fields=["context"]
    )

    # Print Q&A info
    print(f"\n🔎 Question {i+1}: {question}")
    print(f"🎯 Gold Answer: {sample['answers']['text'][0]}\n")

    match_found = False
    printed_hashes = set()

    for rank, hit in enumerate(results[0]):
        context = hit.entity.get("context")
        ctx_hash = short_hash(context)

        if ctx_hash not in printed_hashes:
            contains = answer in context.lower()
            print(f"Result {rank + 1} (score: {hit.distance:.4f}) — Contains answer: {'✅ YES' if contains else '❌ NO'}")
            print(f"(Context hash: {ctx_hash})")
            print(context)
            printed_hashes.add(ctx_hash)

            if contains:
                match_found = True

    if match_found:
        correct += 1

# ----------------------------
# ✅ Final Evaluation
# ----------------------------
recall = correct / NUM_SAMPLES
print(f"\n📊 Recall@{TOP_K} using {METRIC_TYPE}: {correct}/{NUM_SAMPLES} = {recall:.2%}")
