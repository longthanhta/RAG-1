from datasets import load_dataset

# Load only 100 samples
dataset = load_dataset("squad", split="validation[:100]")
for sample in dataset:
    print("Context:", sample['context'])
    print("Question:", sample['question'])
    print("Answer:", sample['answers']['text'])
