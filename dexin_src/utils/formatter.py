# Derived from the Open-R1 project (Apache License 2.0)
# Original code © Hugging Face Inc. (2025)
# Modifications and extensions by Dexin (2025)
# This version rewrites core functionality while maintaining structural consistency

class Formatter:
    def __init__(self, tokenizer, max_length=8192):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, example):
        # this is called by trainer, train=True by default
        prompt = self.format_prompt(example, train=True)
        if example["dataset_type"] == "gpqa":
            tokenized = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=self.max_length)
            return tokenized
        
        if example["dataset_type"] == "math500": # this is for evaluation, no answer and solution provided
            tokenized = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=self.max_length)
            return tokenized
        
        if example["dataset_type"] == "math220k":
            tokenized = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=self.max_length)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
    def format_prompt(self, example, train=False): 
        # This will be called directly by eval_checkpoint, not for training by defualt
        """Return the raw prompt string (without tokenization)"""
        dataset_type = example["dataset_type"]

        if dataset_type!="math220k" and train:
            raise ValueError(f"Dataset cannot be used for training purpose: {dataset_type}")
        
        if dataset_type == "math220k":
            prompt  = "Answer the following question. "
            prompt += f"Question: {example['problem']}\n"
            prompt += "Let's think step by step. Be sure to conclude with your final answer at the end using 'Answer: X' or \\boxed{X}, replacing X with your answer\n"
            if train:
                prompt += f"Solution: \n{example['solution']}\n"
                prompt += f"Answer: {example['answer']}"
            return prompt
        
        elif dataset_type == "math500":
            prompt  = "Answer the following question. "
            prompt += f"Question: {example['problem']}\n"
            prompt += "Let's think step by step. Be sure to conclude with your final answer at the end using 'Answer: X' or \\boxed{X}, replacing X with your answer\n"
            return prompt
        
        elif dataset_type == "gpqa":
            prompt  = "Answer the following multiple choice questions. "
            prompt += f"Question: {example['Question']}\n"
            prompt += f"A. {example['Correct Answer']}\n"
            prompt += f"B. {example['Incorrect Answer 1']}\n"
            prompt += f"C. {example['Incorrect Answer 2']}\n"
            prompt += f"D. {example['Incorrect Answer 3']}\n"
            prompt += "At the end, write your final answer using the format 'Answer: X' or \\boxed{X}, where X is your chosen option letter (A,B,C, or D).\n"
            return prompt
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")