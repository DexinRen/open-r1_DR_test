import random
import json
from datasets import load_dataset, Dataset
from pathlib import Path
import numpy as np

# WORKING DIR: under /open-r1_DR_test
LOCAL_DATASET_PATH = Path(".local_dataset")
SPLITED_DATA_DIR = Path(".local_dataset/math220k-splitted")
BOOTSTRAP_PATH = Path(".local_dataset/bootstrapped")

def validate_split_columns(split_dir: Path, reference_dataset, train_file: str = "train.json", test_file: str = "test.json"):
    """
    Ensures that both train and eval files contain the same columns as the reference HuggingFace dataset.
    """
    from datasets import Dataset

    train_path = split_dir / train_file
    eval_path = split_dir / test_file

    train_ds = Dataset.from_json(str(train_path))
    eval_ds = Dataset.from_json(str(eval_path))

    reference_columns = set(reference_dataset.column_names)
    train_columns = set(train_ds.column_names)
    eval_columns = set(eval_ds.column_names)

    missing_train = reference_columns - train_columns
    missing_eval = reference_columns - eval_columns

    if missing_train:
        raise ValueError(f"[Validation Error] Train set missing columns: {missing_train}")
    if missing_eval:
        raise ValueError(f"[Validation Error] Eval set missing columns: {missing_eval}")

    print(f"[Dataset] validation passed: all expected columns from '{reference_dataset.info.builder_name}' are present.")


def split_math220k():
    target_path = LOCAL_DATASET_PATH / "math220k-splitted"
    train_file = target_path / "train.json"
    test_file = target_path / "test.json"

    if train_file.exists() and test_file.exists():
        print("[Dataset] Fixed split already exists.")
        return

    print("[Dataset] Generating split for math220k...")
    target_path.mkdir(parents=True, exist_ok=True)

    raw = load_dataset("open-r1/OpenR1-Math-220k", name="default", split="train")
    split = raw.train_test_split(test_size=10000, seed=42)
    
    print(f"[Dataset] Train size: {len(split['train'])}, Eval size: {len(split['test'])}")
    # Save JSON files
    split["train"].to_json(train_file)
    split["test"].to_json(test_file)

    # Validate column structure
    validate_split_columns(
        split_dir=target_path,
        reference_dataset=raw
    )

def sample_training_data():
    dataset = Dataset.from_json(str(SPLITED_DATA_DIR / "train.json"))
    sampled = random.sample(list(dataset), 10000)
    for ex in sampled:
        ex["dataset_type"] = "math220k"
    out_path = SPLITED_DATA_DIR / "train_holdout.json"
    with open(out_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"[INFO] Saved 10k sample to {out_path}")
    return check_sample_consistency(str(SPLITED_DATA_DIR / "train.json"), str(SPLITED_DATA_DIR / "train_holdout.json"))


def check_sample_consistency(reference, new_dataset_path):
    new_data = Dataset.from_json(str(new_dataset_path))
    ref_cols = set(reference.column_names)
    new_cols = set(new_data.column_names)
    missing = ref_cols - new_cols
    extra = new_cols - ref_cols - {"dataset_type"}
    print("[CHECK] Missing columns:", missing)
    print("[CHECK] Unexpected extra columns:", extra)
    if not missing and not extra:
        print("Column check passed.")
    return (not missing and not extra)

def bootstrap_to_size(dataset: Dataset, size: int, name: str, seed: int = 42) -> Path:
    BOOTSTRAP_PATH.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), size=size, replace=True)
    bootstrapped = dataset.select(indices)
    
    out_path = BOOTSTRAP_PATH / f"{name}_bootstrapped_{size}.json"
    bootstrapped.to_json(out_path, lines=True,orient="records")
    if check_sample_consistency(dataset, out_path):
        return out_path
    else:
        print(f"Dataset {name} bootstrap failed, exit")
        exit(1)

def bootstrap_all_to_size(size):
    math220k_test = Dataset.from_json(".local_dataset/math220k-splitted/test.json")
    math220k_train = Dataset.from_json(".local_dataset/math220k-splitted/train.json")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    gpqa_diamond = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    gpqa_extended = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train")
    gpqa_mix = Dataset.from_list(list(gpqa_diamond) + list(gpqa_extended))

    bootstrap_to_size(math220k_test, size, "math220k_test")
    bootstrap_to_size(math220k_train, size, "math220k_train")
    bootstrap_to_size(math500, size, "math500")
    bootstrap_to_size(gpqa_mix, size, "gpqa_mix")


size = 1000
split_math220k()
sample_training_data()
bootstrap_all_to_size(size)