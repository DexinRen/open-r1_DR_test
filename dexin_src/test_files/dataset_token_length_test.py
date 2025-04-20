from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset and tokenizer
dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def get_total_length(example):
    # Combine all relevant fields
    total_text = example["problem"] + " " + example["solution"] + " " + example["answer"]
    return len(tokenizer(total_text)["input_ids"])

# Get all lengths
lengths = [get_total_length(example) for example in dataset]

# Find the max
max_len = max(lengths)
max_idx = lengths.index(max_len)
longest_example = dataset[max_idx]

# Print results
print(f"📏 Max tokenized length: {max_len}")
print(f"\n🧮 Problem:\n{longest_example['problem']}\n")
print(f"🧠 Solution:\n{longest_example['solution']}\n")
print(f"✅ Answer: {longest_example['answer']}")
