from datasets import load_dataset
from pathlib import Path

# Config
SAVE_DIR = Path("/home/adonis/proj/open-r1_DR_test/dexin_src/test_datasets")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def save_subset(dataset_name, subset_name, count, filename, split="train"):
    print(f"Loading {dataset_name} ({subset_name})...")
    dataset = load_dataset(dataset_name, name=subset_name, split=split)
    mini = dataset.select(range(min(count, len(dataset))))
    path = SAVE_DIR / filename
    print(f"Saving {len(mini)} rows to {path}")
    mini.save_to_disk(str(path))

# Save mini test sets
save_subset("Idavidrein/gpqa", "gpqa_diamond", 8, "gpqa_diamond_8")
save_subset("Idavidrein/gpqa", "gpqa_experts", 8, "gpqa_experts_8")
save_subset("Idavidrein/gpqa", "gpqa_extended", 8, "gpqa_extended_8")
save_subset("HuggingFaceH4/MATH-500", None, 8, "math500_8", split="test")
save_subset("open-r1/OpenR1-Math-220k", None, 8, "math220k_8", split="train")

print("✅ All mini datasets saved.")
