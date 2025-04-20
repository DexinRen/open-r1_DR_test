#!/bin/bash
# Run with: bash open-r1_DR_test/dexin_src/run_sft.sh
SFT_CONFIG_FILE="/home/adonis/proj/open-r1_DR_test/dexin_src/config/run_config.yaml"
SFT_SCRIPT_PATH="/home/adonis/proj/open-r1_DR_test/dexin_src/stf_auto.py"
ACCELERATE_CONF="/home/adonis/proj/open-r1_DR_test/recipes/accelerate_configs/zero3.yaml"
# Activate UV environment FIRST
# source openr1/bin/activate

# Run script with config file and accelerate
accelerate launch --config_file "$ACCELERATE_CONF" \
  --num_processes 1 \
  "$SFT_SCRIPT_PATH" \
  --config "$SFT_CONFIG_FILE"
