#!/bin/bash

# Use self-hosted Qwen3-VL-8B-Instruct
MODEL='Qwen/Qwen3-VL-32B-Instruct'
# MODEL='Qwen/Qwen3.5-9B'
API_BASE_URL='http://localhost:8000/v1'
API_KEY='None'

# Use the model itself as the judge model
# This is only for demonstration; in practice, you may want to use a more reliable judge model like `gpt-5-nano`
JUDGE_MODEL="gpt-5-nano"
JUDGE_API_BASE_URL="$OPENAI_BASE_URL"
JUDGE_API_KEY="$OPENAI_API_KEY"

# Serve the model with vLLM running in the background
# If this fails, you may want to ensure that the model can be successfully served in the foreground first
TENSOR_PARALLEL_SIZE=8
source ./insight_o3/scripts/vllm_serve_bg.sh "$MODEL" "$TENSOR_PARALLEL_SIZE"

IMG_RESCALE_RATIOS=("0.25" "0.375" "0.5" "1.0")
# IMG_RESCALE_RATIOS=("1.0")

# Evaluate the model on O3-Bench
settings=()
for img_rescale_ratio in "${IMG_RESCALE_RATIOS[@]}"; do
  setting="$MODEL/img_rescale_ratio_$img_rescale_ratio"
  settings+=("$setting")
  python -m insight_o3.scripts.evaluate \
    --eval_name "$setting/o3_bench" \
    --model "$MODEL" \
    --api_base_url "$API_BASE_URL" \
    --api_key "$API_KEY" \
    --judge_model "$JUDGE_MODEL" \
    --judge_api_base_url "$JUDGE_API_BASE_URL" \
    --judge_api_key "$JUDGE_API_KEY" \
    --ann_file "data/O3-Bench/test/metadata.jsonl" \
    --img_dir "data/O3-Bench/test" \
    --img_rescale_ratio $img_rescale_ratio \
    --img_max_pixels $((10000*10000))
done

# Gather evaluation results
python -m insight_o3.scripts.gather_eval_results \
  --settings "${settings[@]}" \
  --datasets "o3_bench"

# Kill the background vLLM process
echo "Evaluation complete. Killing background vLLM process..."
if [[ -n "${vllm_pid:-}" ]]; then
    kill "$vllm_pid"
    wait "$vllm_pid"
fi