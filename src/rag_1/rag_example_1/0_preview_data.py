from datasets import load_dataset
import json

# Load SQuAD dataset
dataset = load_dataset("squad", split="train[:3]")  # Just first 3 examples

# Print each example in pretty JSON format
for i, example in enumerate(dataset):
    print(f"\n📘 Example {i+1}:\n")
    print(json.dumps(example, indent=2, ensure_ascii=False))

