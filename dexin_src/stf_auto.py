import logging
import os
import sys

from pathlib import Path
from datasets import Dataset
from dexin_src.dataset_spliter import ensure_split_math220k

import datasets
import torch
import transformers
from datasets import load_dataset
from datasets import load_from_disk
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from dexin_src.utils.formatter import Formatter
from dexin_src.utils.callbacks import get_callbacks
from open_r1.configs import SFTConfig
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
from utils.evaluate import gpqa_metric, latex_gold_metric  # use your local version
from transformers import EvalPrediction

CUR_EVAL_DATATYPE = 'math'
TEST_FLAG = False
PACKING_EVAL = False
EVAL_LIMIT = 200
LOCAL_SPLIT_PATH = Path("open-r1_DR_test/local_dataset/math220k-splitted")

def check_local_dataset():
    train_path = LOCAL_SPLIT_PATH / "train.json"
    test_path  = LOCAL_SPLIT_PATH / "test.json"
    if not (train_path.exists() and test_path.exists()):
        print("[Data] Fixed split not found — generating it.")
        ensure_split_math220k()
    else:
        print("[Data] Fixed split found — loading from disk.")


def strip_labels(dataset):
    for col in ['labels', 'label', 'label_ids']:
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)
    return dataset

def compute_dynamic_metrics(eval_preds: EvalPrediction, tokenizer):
    preds, _ = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    if CUR_EVAL_DATATYPE == "gpqa":
        references = ["A"] * len(decoded_preds) # correct answer is always "A" 
        return gpqa_metric.compute(predictions=decoded_preds, references=references)
    else:
        return latex_gold_metric.compute(decoded_preds)

logger = logging.getLogger(__name__)

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    train_path = LOCAL_SPLIT_PATH / "train.json"
    test_path  = LOCAL_SPLIT_PATH / "test.json"
    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")
    if TEST_FLAG:
        logger.warning(" TEST_FLAG = TRUE | you are under test mode, this can only be used for debugging stage")
        logger.warning(" TEST_FLAG = TRUE | please change TEST_FLAG = True before normal training")
    
    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Check for local pre-splitted dataset math220k
    check_local_dataset()

    #################
    # Load Tokenizer
    #################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ############################
    # Initialize Data Formatter
    ############################
    train_formatter = Formatter(tokenizer, max_length=training_args.max_seq_length)
    # eval_formatter = Formatter(tokenizer, max_length=2048 if not PACKING_EVAL else training_args.max_seq_length)

    ################
    # Load datasets
    ################

    # Causion, math220k use "\\boxed{}" to indicate the answer
    # answer extraction should identify "\\boxed{Answer}"
    if not TEST_FLAG: 
        ds_train_math220k = Dataset.from_json(str(train_path))
        # ds_test_math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
        # ds_test_gpqa_diamond  = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        # ds_test_gpqa_extended = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train")
    else:
        # Load mini dataset for testing the whole program
        ds_train_math220k = Dataset.from_json(str(train_path)).select(range(2000))
        # ds_test_math220k = Dataset.from_json(str(test_path)).select(range(2000))
        # ds_test_math500 = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(10))
        # ds_test_gpqa_diamond  = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train").select(range(8))
        # ds_test_gpqa_extended = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train").select(range(8))
        
    # add dataset label inside the dataset (important for formatter to distinguish different dataset) 

    # ds_test_math500 = ds_test_math500.map(lambda x: {"dataset_type": "math500"})
    # ds_test_gpqa_diamond = ds_test_gpqa_diamond.add_column("dataset_type", ["gpqa"] * len(ds_test_gpqa_diamond))
    # ds_test_gpqa_extended = ds_test_gpqa_extended.add_column("dataset_type", ["gpqa"] * len(ds_test_gpqa_extended))
    
    ds_train_math220k = ds_train_math220k.add_column("dataset_type", ["math220k"] * len(ds_train_math220k))
    # ds_test_math220k = ds_test_math220k.add_column("dataset_type", ["math220k"] * len(ds_test_math220k))
    ##############################
    # Tokenize Evaluation Dataset
    ##############################

    # Call formatter to format & tokenize data
    # remove all original data columns

    # GPQA will be set as an extra dataset in Trainer
    # which by defualt will not be tokenized and formated
    # So we need to format and tokenize this dataset before set it as extra dataset
    tokenized_train_math220k = ds_train_math220k.map(
        train_formatter,
        remove_columns=ds_train_math220k.column_names,
        desc="Tokenizing training data"
    )

    # tokenized_eval_math220k = ds_test_math220k.map(
    #     eval_formatter,
    #     remove_columns=ds_test_math220k.column_names,
    #     desc="Tokenizing test data"
    # )

    # if EVAL_LIMIT is not None:
    #     tokenized_eval_math220k = tokenized_eval_math220k.select(range(EVAL_LIMIT))
    
    # eval_data = tokenized_eval_math220k if training_args.do_eval else None
    eval_data = None # eval in other file
    # tokenized_test_math500 = ds_test_math500.map(
    #     formatter,
    #     remove_columns=ds_test_math500.column_names,
    # )

    # tokenized_gpqa_diamond = ds_test_gpqa_diamond.map(
    #     formatter,
    #     remove_columns=ds_test_gpqa_diamond.column_names,
    #     # only attributs returned by tokenuizer get preserved
    # ) 

    # tokenized_gpqa_extended = ds_test_gpqa_extended.map(
    #     formatter,
    #     remove_columns=ds_test_gpqa_extended.column_names,
    #     # only attributs returned by tokenuizer get preserved
    # )

    # Strip labels in all evaluation dataset, or the trainer will try to compute 
    # loss on those dataset with empty label and cause CUDA memory problem
    
    # tokenized_test_math500  = strip_labels(tokenized_test_math500) 
    # tokenized_gpqa_diamond  = strip_labels(tokenized_gpqa_diamond)
    # tokenized_gpqa_extended = strip_labels(tokenized_gpqa_extended)

    # tokenized_math500 = ds_test_math500.map(
    #     formatter,.remove_col
    #     remove_columns=ds_test_math500.column_names
    # ) 

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset = tokenized_train_math220k,
        eval_dataset  = eval_data,
        # using process class will let SFTTrainer mistakenly believe
        # we were going to use chat-templete (this is a shit mountain API, sooo stupid)
        # processing_class = formatter, # commented and use formatting_func 
        # to process pre-format and tokenize the dataset(so it do formatting func is doing nothing 
        # but cheat the API that this is not in "Chat Mode" (So Stupid)
        formatting_func = lambda x: x,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        compute_metrics = lambda eval_preds: compute_dynamic_metrics(eval_preds, tokenizer)
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # === Evaluate on GPQA ===

    # We have to use a global var to break the Trainer compute_matics
    # why do this? Lighteval and evaluation is broken, have to use trainer.evaluate
    # trainer evaluate does not take extra flags and argument
    # But we set compute_matrics as lambda so its dynamic and 
    # can read global var every time it get called 
    # (god I hate this, this broken repo and env just force me to use dirty tricks)
    
    # CUR_EVAL_DATATYPE = 'gpqa'
    logger.info("*** Post-training Evaluation: GPQA ***")
    # torch.cuda.empty_cache()
    # preds_gpqa_diamond  = trainer.predict(tokenized_gpqa_diamond)
    # preds_gpqa_diamond_decoded = tokenizer.batch_decode(preds_gpqa_diamond.predictions, skip_special_tokens=True)
    # acc_gpqa_diamond = gpqa_metric.compute(predictions=preds_gpqa_diamond_decoded, references=["A"] * len(preds_gpqa_diamond_decoded))

    # torch.cuda.empty_cache()
    # preds_gpqa_extended = trainer.predict(tokenized_gpqa_extended)
    # preds_gpqa_extended_decoded = tokenizer.batch_decode(preds_gpqa_extended.predictions, skip_special_tokens=True)
    # acc_gpqa_extended = gpqa_metric.compute(predictions=preds_gpqa_extended_decoded, references=["A"] * len(preds_gpqa_extended_decoded))
    
    # === Add Step Num and Sample Len to Matrics ===
    metrics_math220k_train = train_result.metrics
    
    # metrics_gpqa_diamond = {
    #     "accuracy": acc_gpqa_diamond,
    #     "eval_samples": len(tokenized_gpqa_diamond),
    #     "step": trainer.state.global_step,
    # }
    # metrics_gpqa_extended = {
    #     "accuracy": acc_gpqa_extended,
    #     "eval_samples": len(tokenized_gpqa_extended),
    #     "step": trainer.state.global_step,
    # }

    metrics_math220k_train["step"] = trainer.state.global_step
    metrics_math220k_train["train_samples"] = len(ds_train_math220k)

    trainer.log_metrics("train", metrics_math220k_train)
    # trainer.log_metrics("eval_gpqa_diamond", metrics_gpqa_diamond)
    # trainer.log_metrics("eval_gpqa_extended", metrics_gpqa_extended)

    trainer.save_metrics("train", metrics_math220k_train)
    # trainer.save_metrics("eval_gpqa_diamond", metrics_gpqa_diamond)
    # trainer.save_metrics("eval_gpqa_extended", metrics_gpqa_extended)
    
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)