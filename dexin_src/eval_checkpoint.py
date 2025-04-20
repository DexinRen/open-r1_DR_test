import argparse
from pathlib import Path
from datasets import Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from dexin_src.utils.formatter import Formatter
import json

EVAL_DATA_PATH = Path("open-r1_DR_test/local_dataset/math220k-splitted/test.json")
OUTPUT_DIR = Path("open-r1_DR_test/eval_outputs")
import re

def extract_boxed(text):
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    return match.group(1).strip() if match else ""

def main(checkpoint_path: str):
    checkpoint_name = Path(checkpoint_path).name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Eval] Loading tokenizer from: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("[Eval] Loading test set...")
    raw_eval = Dataset.from_json(str(EVAL_DATA_PATH))
    raw_eval = raw_eval.add_column("dataset_type", ["math220k"] * len(raw_eval))

    print("[Eval] Formatting prompts...")
    formatter = Formatter(tokenizer, max_length=4096)
    prompts = raw_eval.map(lambda x: {"prompt": formatter.format_prompt(x)}, remove_columns=raw_eval.column_names)["prompt"]

    print("[Eval] Measuring prompt token lengths...")
    prompt_lengths = [len(tokenizer(prompt)["input_ids"]) for prompt in prompts]

    print(f"[Eval] Prompt token length stats:")
    print(f"  Max length:    {max(prompt_lengths)}")
    print(f"  Min length:    {min(prompt_lengths)}")
    print(f"  Avg length:    {sum(prompt_lengths) / len(prompt_lengths):.2f}")
    print(f"  95th percentile: {sorted(prompt_lengths)[int(0.95 * len(prompt_lengths))]}")
    print(f"  99th percentile: {sorted(prompt_lengths)[int(0.99 * len(prompt_lengths))]}")
    print("[Eval] Launching VLLM...")

    llm = LLM(
        model=checkpoint_path, 
        tokenizer=checkpoint_path, 
        dtype="bfloat16",
        )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=None
    )

    print(f"[Eval] Generating {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)
    generations = [o.outputs[0].text.strip() for o in outputs]

    token_lengths = [len(tokenizer(gen)["input_ids"]) for gen in generations]

    print("[Eval] Generated answer token stats:")
    print(f"  Max length:    {max(token_lengths)}")
    print(f"  Min length:    {min(token_lengths)}")
    print(f"  Avg length:    {sum(token_lengths) / len(token_lengths):.2f}")
    print(f"  95th percentile: {sorted(token_lengths)[int(0.95 * len(token_lengths))]}")
    print(f"  99th percentile: {sorted(token_lengths)[int(0.99 * len(token_lengths))]}")

    preds = [extract_boxed(gen) for gen in generations]
    refs = [extract_boxed(ref) for ref in raw_eval["solution"]]
    correct = sum(p == r for p, r in zip(preds, refs))
    accuracy = correct / len(preds)
    print(f"[Eval] Boxed Accuracy: {accuracy:.4f}")

    # Save generations
    txt_file = OUTPUT_DIR / f"{checkpoint_name}.txt"
    with open(txt_file, "w") as f:
        for i, (prompt, gen) in enumerate(zip(prompts, generations)):
            f.write(f"--- Example {i} ---\n")
            f.write(f"Prompt:\n{prompt}\n")
            f.write(f"Generated:\n{gen}\n\n")

    # Save metrics
    json_file = OUTPUT_DIR / f"{checkpoint_name}.json"
    with open(json_file, "w") as f:
        json.dump({
            "checkpoint": checkpoint_name,
            "accuracy": accuracy,
            "num_samples": len(generations)
        }, f, indent=2)

    print(f"[Eval:{checkpoint_name}] Output saved to {txt_file} and {json_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    main(args.checkpoint)
