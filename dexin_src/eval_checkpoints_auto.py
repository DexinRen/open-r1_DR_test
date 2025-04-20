import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import json
import csv
import re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dexin_src.utils.formatter import Formatter

# Paths
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
CSV_FILE_VANILLA = Path("dexin_src/eval_output/eval_summary.csv")
CSV_FILE_BOOTSTRAP = Path("dexin_src/eval_output/eval_summary_bootstrap.csv")

EVAL_DATA_PATH = {
    "vanilla": Path("local_dataset/math220k-splitted/test.json"),
    "bootstrap": Path("open-r1_DR_test/local_dataset/bootstrapped/math220k_test_bootstrapped_1000.json")
    }

TRAIN_DATA_PATH = {
    "vanilla": Path("local_dataset/math220k-splitted/train_holdout.json"),
    "bootstrap": Path("local_dataset/bootstrapped/math220k_train_bootstrapped_1000.json")
    }

MATH500_DATA_PATH = {
    "vanilla": "HuggingFaceH4/MATH-500",
    "bootstrap": Path("local_dataset/bootstrapped/math500_bootstrapped_1000.json")
}

GPQA_MIX_DATA_PATH = {
    "bootstrap": Path("local_dataset/bootstrapped/gpqa_mix_bootstrapped_1000.json")
}

UNPARSABLE_LOG_PATH = Path("dexin_src/eval_output/generations/unparsable.log")
CHECKPOINT_DIR = Path("data/qwen2.5-0.5B-math220k-default")

GEN_OUTPUT_DIR = Path("dexin_src/eval_output/generations")
GEN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATH220k_DIR = Path("local_dataset/math220k-splitted")

def read_csv_data(bootstrap=False):
    # Setup CSV
    csv_name = CSV_FILE_VANILLA
    if bootstrap: csv_name = CSV_FILE_BOOTSTRAP
    csv_name.parent.mkdir(parents=True, exist_ok=True)

    # Load existing CSV if exists
    existing_data = {}
    if csv_name.exists():
        with open(csv_name, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_data[row["checkpoint"]] = row
    return existing_data

def extract_singlechoice_answer(text: str):
    """
    Extracts the final answer from the model's generation using either \\boxed{} or 'Answer: $LETTER' format.

    - If both boxed and 'Answer:' are present, use boxed.
    - Skip all prompt/instructional matches like \\boxed{$Your_Answer} or 'Answer: $LETTER'.
    - If no valid answer is found, log the generation and return None.
    """
    # === [Step 1: Extract boxed answer if present, but exclude instructional pattern] ===
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    for boxed in boxed_matches[::-1]:  # Check from last to first
        boxed = boxed.strip()
        if boxed in ["A", "B", "C", "D"]:
            return boxed, 0
        if boxed in ["$LETTER", "$Your_Answer"]:
            continue  # skip instructional lines

    # === [Step 2: Extract from 'Answer:' ONLY IF it's not from the prompt] ===
    lines = text.strip().splitlines()
    for line in reversed(lines):  # Check from last to first
        if "Answer:" in line:
            if "$LETTER" in line or "$Your_Answer" in line:
                continue  # skip prompt/instruction
            after = line.split("Answer:")[-1].strip()
            token = after.split()[0].strip("().,:;") if after else ""
            if token in ["A", "B", "C", "D"]:
                return token, 0
            
    # === [Step 3: Extract from Other form of answer that is not following the given format] ===
    soft_match = re.search(r"(?:Answer:\s*\n|answer is|final answer:|answer to this question is)\s*([A-D])\b|option\s+([A-D])\s+is\s+correct", text, re.IGNORECASE)
    if soft_match:
        # Search for A–D immediately following
        letter_search = re.search(r"\b([A-D])\b", soft_match.group(0), re.IGNORECASE)
        if letter_search:
            return letter_search.group(1).upper(), 1

    # === [Step 4: Log unparsable generation and return None] ===
    with open(UNPARSABLE_LOG_PATH, "a") as f:
        f.write(text + "\n\n===========ENDOF EXAMPLE==============\n\n")

    return None, 0

# Formatter for prompt construction
def get_formatter(tokenizer):
    return Formatter(tokenizer, max_length=4096)

# Extract \boxed{} answer
def extract_boxed(text):
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    return match.group(1).strip() if match else ""

def acc_calculator(preds, refs, anss, prompts, gen, save_name):
    correct = 0
    wrong_log = ""
    
    def clean(x):
        return str(x).strip().strip("=., ").lower()
    
    for i, (p,r,a) in enumerate(zip(preds, refs, anss)):
        sol_match = (clean(p) == clean(r))
        ans_match = (clean(p) == clean(a))
        grade = int(ans_match or sol_match)
        correct += grade
        if grade == 0:
            wrong_log += f"--- Question {i} ---\n"
            wrong_log += f"Prompt:\n{prompts[i]}\n"
            wrong_log += f"Generated:\n{gen[i]}\n"
            wrong_log += f"|pred: {p} | ref: {r} | ans: {a}|\n\n"

    accuracy = correct / len(preds)
    with open(GEN_OUTPUT_DIR / f"{save_name}_wrong_preds.txt", "a") as f:
        f.write(wrong_log)
    
    return accuracy

# Evaluate on math220k test set
def evaluate_math_accuracy(checkpoint_path_or_model, save_name, bootstrap=False):
    if isinstance(checkpoint_path_or_model, Path):
        checkpoint_path_or_model = str(checkpoint_path_or_model)
    
    if bootstrap:
        ds_test  = Dataset.from_json(str(EVAL_DATA_PATH["bootstrap"])) # 1k bootstrapped data 
        ds_train = Dataset.from_json(str(TRAIN_DATA_PATH["bootstrap"]))# 1k bootstrapped data
        ds_math500 = Dataset.from_json(str(MATH500_DATA_PATH["bootstrap"])) # 1k bootstrapped data
    else:
        ds_test  = Dataset.from_json(str(EVAL_DATA_PATH["vanilla"]))  # 10k data
        ds_train = Dataset.from_json(str(TRAIN_DATA_PATH["vanilla"])) # 10k data
        ds_math500 = load_dataset("HuggingFaceH4/MATH-500", split="test") # 500 data
    
    ds_test    = ds_test.add_column("dataset_type", ["math220k"] * len(ds_test))
    ds_train   = ds_train.add_column("dataset_type", ["math220k"] * len(ds_train))
    ds_math500 = ds_math500.add_column("dataset_type", ["math500"] * len(ds_math500))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path_or_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    formatter = get_formatter(tokenizer)

    prompts_test  =  ds_test.map(lambda x: {"prompt": formatter.format_prompt(x)}, remove_columns=ds_test.column_names)["prompt"]
    prompts_train = ds_train.map(lambda x: {"prompt": formatter.format_prompt(x)}, remove_columns=ds_train.column_names)["prompt"]
    prompts_math500  =  ds_math500.map(lambda x: {"prompt": formatter.format_prompt(x)}, remove_columns=ds_math500.column_names)["prompt"]

    llm = LLM(model=str(checkpoint_path_or_model), tokenizer=str(checkpoint_path_or_model), dtype="bfloat16")
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=768, stop=None)
    
    outputs_test = llm.generate(prompts_test, sampling_params)
    gen_test = [o.outputs[0].text.strip() for o in outputs_test]    
    preds_test = [extract_boxed(gen) for gen in gen_test]
    refs_test = [extract_boxed(ref) for ref in ds_test["solution"]] # SHOULD BE SOLUTION
    anss_test = [ans for ans in ds_test["answer"]]

    outputs_train = llm.generate(prompts_train, sampling_params)
    gen_train = [o.outputs[0].text.strip() for o in outputs_train]
    preds_train = [extract_boxed(gen) for gen in gen_train]
    refs_train = [extract_boxed(ref) for ref in ds_train["solution"]] # SHOULD BE SOLUTION
    anss_train = [ans for ans in ds_train["answer"]]
    
    outputs_math500 = llm.generate(prompts_math500, sampling_params)
    gen_math500 = [o.outputs[0].text.strip() for o in outputs_math500]    
    preds_math500 = [extract_boxed(gen) for gen in gen_math500]
    refs_math500 = [extract_boxed(ref) for ref in ds_math500["solution"]] # SHOULD BE SOLUTION
    anss_math500 = [ans for ans in ds_math500["answer"]]

    accuracy = {}
    accuracy["math220k_test"] = acc_calculator(preds_test, refs_test, anss_test, prompts_test, gen_test, save_name+"math220k-test")
    accuracy["math220k_train"] = acc_calculator(preds_train, refs_train, anss_train, prompts_train, gen_train, save_name+"math220k-train")
    accuracy["math500"] = acc_calculator(preds_math500, refs_math500, anss_math500, prompts_math500, gen_math500, save_name+"math500")
    return accuracy


def evaluate_gpqa_accuracy(checkpoint_path_or_model, subset, save_name, bootstrap=False):
    if bootstrap:
        dataset  = Dataset.from_json(str(GPQA_MIX_DATA_PATH["bootstrap"])) # 1k bootstrapped data 
    else:
        dataset = load_dataset("Idavidrein/gpqa", subset, split="train")
    
    dataset = dataset.add_column("dataset_type", ["gpqa"] * len(dataset))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path_or_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    formatter = get_formatter(tokenizer)

    prompts = [formatter.format_prompt(x) for x in dataset]

    llm = LLM(model=str(checkpoint_path_or_model), tokenizer=str(checkpoint_path_or_model), dtype="bfloat16")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["You are an AI assistant"],
        repetition_penalty=1.1
    )

    outputs = llm.generate(prompts, sampling_params)
    generations = [o.outputs[0].text.strip() for o in outputs]
    
    question_num = len(prompts)
    clean_answers = []
    dirty_answers = []
    blank_count = 0
    for i, gen in enumerate(generations):
        ans, unformated = extract_singlechoice_answer(gen)
        if ans is None:
            blank_count += 1
        if unformated:
            dirty_answers.append(ans)
        else:
            clean_answers.append(ans)

    print(f"\n====[DEBUG]==== Blank or unparsable answers: {blank_count} / {len(generations)}\n")
    
    correct_clean_num = sum(answer == "A" for answer in clean_answers)
    correct_dirty_num = sum(answer == "A" for answer in dirty_answers)
    # Save generation with wrong answer
    with open(GEN_OUTPUT_DIR / f"{subset}_wrong_answer_{save_name}.log", "w") as f:
        for i, (prompt, gen, pred) in enumerate(zip(prompts, generations, clean_answers)):
            if pred != "A":
                f.write(f"--- Incorrect Example {i} ---\n")
                f.write(f"Prompt:\n{prompt}\n")
                f.write(f"Generated:\n{gen}\n")
                f.write(f"Answer Extracted: {pred}\n\n")
    out = {
        "acc_all": (correct_clean_num + correct_dirty_num) / question_num, 
        "acc_clean":correct_clean_num / question_num,
        }
    return out

def eval_gpqa(bootstrap=False):
    csv_name = CSV_FILE_VANILLA
    if bootstrap: csv_name = CSV_FILE_BOOTSTRAP
    existing_data = read_csv_data()
    # Evaluate missing GPQA results
    checkpoints = ["untrained"] + [d.name for d in CHECKPOINT_DIR.glob("checkpoint-*") if d.is_dir()]
    # checkpoints.sort(key=lambda x: (float("inf") if x == "untrained" else int(x.split("-")[-1])))

    with open(csv_name, "w", newline="") as f:
        fieldnames = ["checkpoint", "train_accuracy", "train_loss", "test_accuracy", "math500_accuracy", "gpqa_diamond_accuracy_all", "gpqa_diamond_accuracy_clean",  "gpqa_extended_accuracy_all", "gpqa_extended_accuracy_clean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name in checkpoints:
            row = existing_data.get(name, {"checkpoint": name})
            message = f"\n\n#### Evaluation Result for Checkpoint: {name} ####\n"
            if name == "untrained":
                model_path = BASE_MODEL
            else:
                model_path = CHECKPOINT_DIR / name
            
            if "gpqa_diamond_accuracy" not in row:
                print(f"\n====[Eval]==== GPQA diamond for {name}\n")
                gpqa_diamond_acc = evaluate_gpqa_accuracy(model_path, "gpqa_diamond", name)
                row["gpqa_diamond_accuracy_all"] = gpqa_diamond_acc["acc_all"]
                row["gpqa_diamond_accuracy_clean"] = gpqa_diamond_acc["acc_clean"]
                message += f"####\t| gpqa-diamond acc all    = {gpqa_diamond_acc['acc_all']}\n"
                message += f"####\t| gpqa-diamond acc clean  = {gpqa_diamond_acc['acc_clean']}\n"

            if "gpqa_extended_accuracy" not in row:
                print(f"\n====[Eval]==== GPQA extended for {name}\n")
                gpqa_extended_acc = evaluate_gpqa_accuracy(model_path, "gpqa_extended", name)
                row["gpqa_extended_accuracy_all"] = gpqa_extended_acc["acc_all"]
                row["gpqa_extended_accuracy_clean"] = gpqa_extended_acc["acc_clean"]
                message += f"####\t| gpqa-extended acc all   = {gpqa_extended_acc['acc_all']}\n"
                message += f"####\t| gpqa-extended acc clean = {gpqa_extended_acc['acc_clean']}\n"

            message += "\n\n"
            print(message)
            writer.writerow(row)


def eval_all(bootstrap=False):
    # Setup CSV
    csv_name = CSV_FILE_VANILLA
    if bootstrap: csv_name = CSV_FILE_BOOTSTRAP
    csv_name.parent.mkdir(parents=True, exist_ok=True)
    
    if bootstrap:
        column_names = ["checkpoint", "method", 
                        "train_accuracy", "test_accuracy", "math500_accuracy", 
                        "gpqa_mix_accuracy_clean", "gpqa_mix_accuracy_all"]
    else:
        column_names = ["checkpoint", "method", 
                        "train_accuracy", "test_accuracy", "math500_accuracy", 
                        "gpqa_diamond_accuracy_clean", "gpqa_diamond_accuracy_all",
                        "gpqa_extended_accuracy_clean", "gpqa_extended_accuracy_all"]
    
    if not csv_name.exists():
        with open(csv_name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=column_names)
            writer.writeheader()

    # Evaluate all checkpoints
    checkpoints = [d for d in CHECKPOINT_DIR.glob("checkpoint-*") if d.is_dir()]
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    checkpoints.insert(0, BASE_MODEL)

    print(f"[DEBUG] Found {len(checkpoints)} checkpoints")

    for c in checkpoints:
        if isinstance(c, str): # BASE MODEL
            name = "untrained"
        else: name = c.name

        math_acc = evaluate_math_accuracy(c, name, bootstrap=bootstrap)
        if bootstrap:
            gpqa_mix_acc = evaluate_gpqa_accuracy(c, "gpqa_mix", name, bootstrap=bootstrap)
        else:
            gpqa_diamond_acc = evaluate_gpqa_accuracy(c, "gpqa_diamond", name, bootstrap=bootstrap)
            gpqa_extended_acc = evaluate_gpqa_accuracy(c, "gpqa_extended", name, bootstrap=bootstrap)
        
        if bootstrap:
            checkpoint_status = {
                "checkpoint": name,
                "method": "bootstrap (size=1000)",
                "train_accuracy": math_acc["math220k_train"],
                "test_accuracy": math_acc["math220k_test"],
                "math500_accuracy": math_acc["math500"],
                "gpqa_mix_accuracy_clean": gpqa_mix_acc["acc_clean"],
                "gpqa_mix_accuracy_all": gpqa_mix_acc["acc_all"],
            }

        else:    
            checkpoint_status = {
                "checkpoint": name,
                "method": "vanilla dataset", 
                "train_accuracy": math_acc["math220k_train"],
                "test_accuracy": math_acc["math220k_test"],
                "math500_accuracy": math_acc["math500"],
                "gpqa_diamond_accuracy_clean": gpqa_diamond_acc["acc_clean"],
                "gpqa_diamond_accuracy_all": gpqa_diamond_acc["acc_all"],
                "gpqa_extended_accuracy_clean":gpqa_extended_acc["acc_clean"],
                "gpqa_extended_accuracy_all":gpqa_extended_acc["acc_all"],
            }

        with open(csv_name, "a", newline="") as acc_log:
            writer = csv.DictWriter(acc_log, fieldnames=column_names)
            writer.writerow(checkpoint_status)

        # print out current checkpoint status:
        print("\n=======================")
        print(f"EVAL FINISHED:\n\t | Model Name: {name}\n")
        for i in checkpoint_status.keys():
            print(f"\t | {i}:{checkpoint_status[i]}\n")
        print("\n=======================\n")

eval_all(bootstrap=True)