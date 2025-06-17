# retriever_milvus.py - Build vector store in Milvus using SQuAD train set

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from tqdm import tqdm
from sklearn.preprocessing import normalize

# -----------------------
# ✅ Configuration
# -----------------------
COLLECTION_NAME = "squad_rag"
EMBEDDING_DIM = 384  # Dimension of MiniLM embeddings
BATCH_SIZE = 10

# Change this to either "L2" or "IP"
METRIC_TYPE = "L2"  # use "IP" for cosine similarity (requires normalization), or "L2" for Euclidean

# -----------------------
# ✅ Step 1: Connect to Milvus
# -----------------------
connections.connect("default", host="localhost", port="19530")

# -----------------------
# ✅ Step 2: Define and create collection
# -----------------------
# Drop existing collection (if any)
if utility.has_collection(COLLECTION_NAME):
    Collection(COLLECTION_NAME).drop()

# Define fields: auto-increment ID, context text, and embedding vector
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
]
schema = CollectionSchema(fields, description="SQuAD RAG document collection")
collection = Collection(name=COLLECTION_NAME, schema=schema)

# -----------------------
# ✅ Step 3: Load embedding model
# -----------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# ✅ Step 4: Load dataset (e.g., first 30,000 samples)
# -----------------------
print("Loading dataset...")
dataset = load_dataset("squad", split="train[:30000]")

# -----------------------
# ✅ Step 5: Embed context and prepare data
# -----------------------
print("Encoding contexts...")
contexts = [sample["context"] for sample in tqdm(dataset, desc="Extracting contexts")]
embeddings = embedder.encode(contexts, batch_size=BATCH_SIZE, show_progress_bar=True)

# Normalize if using IP (to simulate cosine similarity)
if METRIC_TYPE == "IP":
    print("Normalizing embeddings for cosine similarity...")
    embeddings = normalize(embeddings)

# Prepare data for insertion into Milvus
print("Preparing data for insertion...")
milvus_data = [
    contexts,                          # context list
    [emb.tolist() for emb in embeddings]  # vector list
]

# -----------------------
# ✅ Step 6: Insert into Milvus
# -----------------------
print("Inserting into Milvus...")
collection.insert(milvus_data)
collection.flush()

# -----------------------
# ✅ Step 7: Create index and load
# -----------------------
print("Creating index and loading collection...")
collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": METRIC_TYPE,        # "L2" or "IP"
        "params": {"nlist": 128}
    }
)
collection.load()

print(f"✅ Inserted {len(contexts)} documents into Milvus collection '{COLLECTION_NAME}' using metric '{METRIC_TYPE}'.")
