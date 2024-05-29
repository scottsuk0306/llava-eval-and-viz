#!/bin/bash

# Ensure the logs directory exists
mkdir -p logs

# Start each command in the background and redirect output to a log file
python -m src.run_inference_vlm_api --model_name claude-3-haiku-20240307 > logs/claude-3-haiku-20240307.log 2>&1 &
python -m src.run_inference_vlm_api --model_name claude-3-sonnet-20240229 > logs/claude-3-sonnet-20240229.log 2>&1 &
python -m src.run_inference_vlm_api --model_name claude-3-opus-20240229 > logs/claude-3-opus-20240229.log 2>&1 &
python -m src.run_inference_vlm_api --model_name gemini/gemini-pro-vision > logs/gemini-pro-vision.log 2>&1 &
python -m src.run_inference_vlm_api --model_name gpt-4-vision-preview > logs/gpt-4-vision-preview.log 2>&1 &

echo "All processes started. Check the logs directory for output."
