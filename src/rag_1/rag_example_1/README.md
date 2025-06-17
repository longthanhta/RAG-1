L2_result.txt is output from 3_test_retrival.py when the milvus data store the data and calculate based on L2

IP_result.txt is output from 3_test_retrival.py when the milvus data store the data and calculate based on L2

collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": METRIC_TYPE,        # "L2" or "IP"
        "params": {"nlist": 128}
    }
)

but the result so there is not much different in accuracy


Additional information:

Source: https://github.com/UKPLab/sentence-transformers

Models like "all-MiniLM-L6-v2" belong to the all-* family of "universal sentence encoders" optimized for cosine similarity:

"These models are optimized for cosine-similarity by training on normalized embeddings."

This includes:

all-MiniLM-L6-v2

all-mpnet-base-v2

These models are trained with triplet loss using cosine similarity, so they tend to produce unit-norm embeddings naturally, even without explicitly normalizing.