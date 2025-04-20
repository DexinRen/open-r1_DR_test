NUM_GPUS=1
# the generation size (MAX_NEW_TOEKNS + context size) of Qwen2.5 0.5B model is self.max_length=32768, 
# if MAX_NEW_TOEKNS + context exceed that, vllm will automatically set context = 0, 
# if context size = 0, model will generate without any prompt or context (AVOID THIS!!!)
MAX_NEW_TOKENS=24576 
TEMP=0
TOP_P=1
CHECKPOINT_NAME=checkpoint-118500
MODEL=/home/adonis/proj/open-r1_DR_test/data/Qwen2.5-0.5B-Open-R1-Distill/$CHECKPOINT_NAME
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.7,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:$TEMP,top_p:$TOP_P}"

# MODEL=jdqqjr/Qwen2.5-0.5B-Open-R1-Distill
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.7,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:$TEMP,top_p:$TOP_P}"

OUTPUT_DIR=test_files/bin/$CHECKPOINT_NAME
TASK_NAME=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK_NAME|0|0" \
    --use-chat-template \
    --save-details \
    --custom-tasks /home/adonis/proj/open-r1_DR_test/src/open_r1/evaluate.py \
    --output-dir $OUTPUT_DIR