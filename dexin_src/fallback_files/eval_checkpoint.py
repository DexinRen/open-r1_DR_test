import re
import os
import numpy as np
import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# evaluator doc: https://huggingface.co/docs/evaluate/en/base_evaluator
from typing import List, Dict, Union
from transformers import pipeline
# QWen2.5 is causalLM
# The model is generating markdown+latex format answer instead of plain text, have to "Translate" this
import random

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


GOLD_IDX = 0
LOG_FILE = "data/performance_log.txt"
CHECK_POINT = "/home/adonis/proj/open-r1_DR_test/data/Qwen2.5-0.5B-Open-R1-Distill/checkpoint-118500"

class DatasetCard:
    def __init__(self, repo_address, dataset_name, split, subset_name=None):
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.repo_address = repo_address
        self.split = split
        self.data_processor()
        self.num = len(self.raw_data)
    
    def data_processor(self):
        self.raw_data = load_dataset(self.repo_address, name=self.subset_name, split=self.split)
        if self.dataset_name in ["gpqa_extended","gpqa_diamond","gpqa_experts","gpqa"]:
            self.prompts = [gpqa_data_prep(line) for line in self.raw_data]
        else: self.prompts = self.raw_data

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

def gpqa_data_prep(line:str)->str:
    """This function is a customized version from Project Open_R1 scr/openr1/evaluate.py., The only difference is the golden index is fixed at 0 (option A)"""
    """Prompt template adapted from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14"""

    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(GOLD_IDX, line["Correct Answer"])
    # vanilla query_templete = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    format_spec = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering"
    query_template = format_spec + "\nQuestion: {Question}\n A : {A}; B : {B}; C : {C}; D: {D}"
    query = query_template.format(A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"])    
    return query

def answer_compare(text:str, label:str) -> int:
    matches = re.findall(r"Answer\s*([A-D])", text)
    if matches:
        return int(matches[-1] == label)
    return 0

def accuracy(grades:list) -> float:
    answer_count = len(grades)
    correct_count = sum(grades)
    
    return correct_count/answer_count

def gpqa_grader(raw_answer:List[List[Dict]], prompts:List[str])->Dict[str, Union[List[int], List[str]]]:
    grade = 0
    grades = []
    answer_log = []
    correct_answer = {0:"A", 1:"B", 2:"C", 3:"D"}[GOLD_IDX]
    # Remove all symbols and question head
    # to simplify answer extraction process
    for i in range(len(raw_answer)):
        question_len = len(prompts[i])
        matches = re.findall(r"\\boxed{([A-D])}", raw_answer[i])
        if not matches:
            text_head_removed = raw_answer[i][0]['generated_text'][question_len:]
            text_symb_removed = re.sub(r'[^a-zA-Z0-9\s]', '', text_head_removed)
            matches = re.findall(r"Answer\s*([A-D])", text_symb_removed)
        if matches:
            grade = int(matches[-1] == "A")
        grades.append(grade)
        answer_log.append(text_symb_removed)
        grade = 0
    return {"grades":grades, "answer_log":answer_log}

def generation_with_log_gpqa(pipe:pipeline, dataset:DatasetCard, batch_size:int, max_new_tokens:int)->Dict[str, Union[List[int], List[str], float]]:
    i = 0
    curr_acc = 0
    grades = []
    answer_log = []
    report_gap = 4 * batch_size

    while i < dataset.num:
        clear_terminal()
        progress = i / dataset.num
        print(f"Evaluation In Progress:\n\t| Dataset: {dataset.dataset_name} \n\t| Query answered:{i}/{dataset.num} \n\t| Current Accuracy: {curr_acc}\n\t| Batch size: {batch_size}, max_new_token:{max_new_tokens}")
        print("\t| Progress: [ " + "".join([">" for _ in range(0,int(progress*50))])+"".join("=" for _ in range(int(progress*50), 50))+" ] "+str(progress*100)+"%\n")
        
        if i+report_gap < dataset.num:
            output  = gpqa_grader(pipe(dataset.prompts[i:i+report_gap],  max_new_tokens=max_new_tokens, batch_size=batch_size), dataset.prompts[i:i+report_gap])
        else:
            output  = gpqa_grader(pipe(dataset.prompts[i:],  max_new_tokens=max_new_tokens, batch_size=batch_size), dataset.prompts[i:])
        
        grades.extend(output["grades"])
        answer_log.extend(output["answer_log"])
        
        curr_acc = accuracy(grades)
        i += report_gap
    
    return {"grades": grades, "answer_log":answer_log, "accuracy": curr_acc}

# def GenText2Doc(dataset, outputs):
#     docs = []
#     for i, row in enumerate(dataset):
#         prompt = row
#         docs.append(Doc(
#             task_name="gpqa",
#             query=prompt,
#             choices=["A", "B", "C", "D"],
#             gold_index=0,  # or whatever the correct answer index is
#         ))
#     for i in range(len(docs)):
#         docs[i].response = outputs[i][0]["generated_text"]
#     return docs

# gpqa_metric = multilingual_extractive_match_metric(
#     language=Language.ENGLISH,
#     gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
#     pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
#     precision=5
# )

def main():
    # Load Tokenizer, code borrowed from scr/sft.py
    report = {}

    report[CHECK_POINT] = {}

    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        CHECK_POINT, trust_remote_code=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 
    
    pipe = pipeline("text-generation", model=CHECK_POINT, tokenizer=tokenizer)
    gpqa = ["gpqa_extended","gpqa_diamond","gpqa_experts","gpqa"]
    ##### Loading Dataset #####
    print("Loading Dataset")
    datasets_gpqa = [
        DatasetCard("Idavidrein/gpqa", "gpqa_diamond", "train", subset_name = "gpqa_diamond"),
        DatasetCard("Idavidrein/gpqa", "gpqa_diamond", "train", subset_name = "gpqa_extended"),
        #DatasetCard("open-r1/OpenR1-Math-220k", "Math_220k", "train"),
        #DatasetCard("HuggingFaceH4/MATH-500", "Math_500", "train"),
    ]
    
    for dataset in datasets_gpqa:
        if dataset.dataset_name in gpqa:
            batch_size = 8
            max_new_tokens = 2048
            outputs  = generation_with_log_gpqa(pipe, dataset, batch_size, max_new_tokens)
            report[CHECK_POINT][dataset.dataset_name] = outputs["accuracy"]
    #docs_gpqa_extended = GenText2Doc(gpqa_extended_formated, raw_answer_gpqa_extended)
    #gpqa_diamond_acc = gpqa_accuracy(raw_answer_gpqa_diamond, "A")
    #gpqa_extended_acc = gpqa_accuracy(raw_answer_gpqa_extended, "A")

    # print("============== Accuracy ==============")
    # print(f"gpqa_diamond_acc  = {gpqa_diamond_acc}")
    # print(f"gpqa_extended_acc = {gpqa_extended_acc}")

def prompt_fn(line, task_name: str = None):
    """Assumes the model is either prompted to emit \\boxed{answer} or does so automatically"""
    return Doc(
        task_name=task_name,
        query=line["problem"],
        choices=[line["solution"]],
        gold_index=0,
    )

latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)

def eval_by_evalutate():
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "/home/adonis/proj/open-r1_DR_test/data/Qwen2.5-0.5B-Open-R1-Distill/checkpoint-118500", 
        device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Load evaluation dataset (e.g., hellaswag)
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="train")

    # Load metric
    metric = latex_gold_metric

    correct = 0
    for line in dataset:
        prompt = line["question"] + "\n solution: "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=32)
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Dummy match
        if line["endings"][0] in predicted_text:
            correct += 1


if __name__ == "__main__":
    #main()
    eval_by_evalutate()