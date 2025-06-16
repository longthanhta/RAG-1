# retriever_milvus.py - Build vector store in Milvus using SQuAD train set

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from tqdm import tqdm
import numpy as np

# Config
COLLECTION_NAME = "squad_rag"
EMBEDDING_DIM = 384  # MiniLM embedding size
BATCH_SIZE = 10

# 1. Connect to Milvus (assumes standalone running on localhost:19530)
connections.connect("default", host="localhost", port="19530")

# 2. Create collection if not exists
if utility.has_collection(COLLECTION_NAME):
    Collection(COLLECTION_NAME).drop()

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
]
schema = CollectionSchema(fields, description="SQuAD RAG document collection")
collection = Collection(name=COLLECTION_NAME, schema=schema)

# 3. Init embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Load SQuAD train set (1000 samples)
print("Loading dataset...")
dataset = load_dataset("squad", split="train[:30000]")
#dataset = load_dataset("squad", split="train")

# 5. Insert documents
print("Encoding contexts...")
contexts = [sample["context"] for sample in tqdm(dataset, desc="Extracting contexts")]
embeddings = embedder.encode(contexts, batch_size=BATCH_SIZE, show_progress_bar=True)

# Convert to list of lists
print("Preparing data for insertion...")
milvus_data = [
    [contexts[i] for i in range(len(contexts))],
    [embeddings[i].tolist() for i in range(len(embeddings))],
]

print("Inserting into Milvus...")
collection.insert(milvus_data)
collection.flush()

# 6. Create index (required for search) and load collection
print("Creating index and loading collection...")
collection.create_index(
    field_name="embedding",
    index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
)
collection.load()

print(f"Inserted {len(contexts)} documents into Milvus collection '{COLLECTION_NAME}' and loaded it for search.")

