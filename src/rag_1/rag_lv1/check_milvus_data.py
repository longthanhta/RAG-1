import sys
import json

print("Number of records:", len(milvus_data))
print("Size of first record (MB):", sys.getsizeof(json.dumps(milvus_data[0])) / 1024**2)
print("Estimated total size (MB):", sys.getsizeof(json.dumps(milvus_data)) / 1024**2)
