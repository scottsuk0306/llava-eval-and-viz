#!/bin/bash

# Ensure the logs directory exists
mkdir -p logs

# Start each command in the background and redirect output to a log file
CUDA_VISIBLE_DEVICES=1 python -m src.run_inference_vlm --model_name llava-hf/llava-1.5-7b-hf > logs/llava-1.5-7b-hf.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m src.run_inference_vlm --model_name llava-hf/llava-1.5-13b-hf > logs/llava-1.5-13b-hf.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -m src.run_inference_vlm --model_name llava-hf/llava-v1.6-mistral-7b-hf > logs/llava-v1.6-mistral-7b-hf.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m src.run_inference_vlm --model_name llava-hf/llava-v1.6-34b-hf > logs/llava-v1.6-34b-hf.log 2>&1 &

echo "All processes started. Check the logs directory for output."
