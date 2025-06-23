import pickle
from pathlib import Path

# Load the saved chunks
chunk_path = Path(__file__).parent / "text_chunks.pkl"

with open(chunk_path, "rb") as f:
    chunks = pickle.load(f)

# Display chunks in a readable format
for i, chunk in enumerate(chunks):
    print(f"===== Chunk {i + 1} =====")
    print(chunk)
    print("\n" + "=" * 40 + "\n")

# Optional: limit how many chunks to preview
# You can use slices like chunks[:10] to preview only a few
