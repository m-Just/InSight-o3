#!/bin/bash

# Use self-hosted Qwen3-VL-8B-Instruct
MODEL='Qwen/Qwen3-VL-8B-Instruct'
API_BASE_URL='http://localhost:8000/v1'
API_KEY='None'

# Use the model itself as the judge model
# This is only for demonstration; in practice, you may want to use a more consistent judge model like `gpt-5-nano`
JUDGE_MODEL="$MODEL"
JUDGE_API_BASE_URL="$API_BASE_URL"
JUDGE_API_KEY="$API_KEY"

# Serve the model with vLLM running in the background
# If this fails, you may want to ensure that the model can be successfully served in the foreground first
TENSOR_PARALLEL_SIZE=8   # use 8 GPUs for tensor parallelism
source ./insight_o3/scripts/vllm_serve_bg.sh "$MODEL" "$TENSOR_PARALLEL_SIZE"

# Evaluate the model on O3-Bench
python -m insight_o3.scripts.evaluate \
  --eval_name "$MODEL/o3_bench" \
  --model "$MODEL" \
  --api_base_url "$API_BASE_URL" \
  --api_key "$API_KEY" \
  --judge_model "$JUDGE_MODEL" \
  --judge_api_base_url "$JUDGE_API_BASE_URL" \
  --judge_api_key "$JUDGE_API_KEY" \
  --ann_file "data/O3-Bench/test/metadata.jsonl" \
  --img_dir "data/O3-Bench/test"

# Gather evaluation results
python -m insight_o3.scripts.gather_eval_results \
  --settings "$MODEL" \
  --datasets "o3_bench"

# Kill the background vLLM process
echo "Evaluation complete. Killing background vLLM process..."
if [[ -n "${vllm_pid:-}" ]]; then
    kill "$vllm_pid"
    wait "$vllm_pid"
fi