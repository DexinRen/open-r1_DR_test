
import logging
import os
import sys

import datasets
import torch
import inspect
import transformers
import lighteval
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForCausalLM
from open_r1.configs import SFTConfig
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
import lighteval.metrics
logger = logging.getLogger(__name__)

def model_signature(model):
    model_to_inspect = model
    if hasattr(model, "get_base_model"):
        model_to_inspect = model.get_base_model()
    else:
        # PeftMixedModel do not provide a `get_base_model` method
        model_to_inspect = model.base_model.model
    signature = inspect.signature(model.forward) 
    return signature

def main():
    #model = AutoModelForCausalLM.from_pretrained("/home/adonis/proj/open-r1_DR_test/data/Qwen2.5-0.5B-Open-R1-Distill/checkpoint-116000")
    #signature = inspect.signature(model.forward) 
    # dataset_gpqa_diamond  = load_dataset("Idavidrein/gpqa", name="gpqa_diamond")
    # dataset_gpqa_extended = load_dataset("Idavidrein/gpqa", name="gpqa_extended")
    # dataset_math_500      = load_dataset("HuggingFaceH4/MATH-500")
    # dataset_math_220k = load_dataset("open-r1/OpenR1-Math-220k")
    #print(f"model signature keys : {signature.parameters.keys()}\n")
    # print(f"MATH-220K columns    : {dataset_math_220k.column_names}\n")
    # print(f"MATH-500 columns     : {dataset_math_500.column_names}\n")
    # print(f"gpqa_diamond columns : {dataset_gpqa_diamond.column_names}\n")
    # print(f"gpqa_extended columns: {dataset_gpqa_extended.column_names}\n")
    #  odict_keys(['input_ids', 'attention_mask', 'position_ids', 'past_key_values', 
    #              'inputs_embeds', 'labels', 'use_cache', 'output_attentions', 
    #              'output_hidden_states', 'return_dict'])
    print(dir(lighteval.metrics))
main()