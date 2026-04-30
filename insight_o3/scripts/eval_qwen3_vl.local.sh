#!/bin/bash

# Use self-hosted Qwen3-VL-8B-Instruct
# MODEL='Qwen/Qwen3-VL-8B-Instruct'
# TENSOR_PARALLEL_SIZE=1

MODEL='Qwen/Qwen3-VL-32B-Instruct'
TENSOR_PARALLEL_SIZE=4

API_BASE_URL='http://localhost:8000/v1'
API_KEY='None'

# Use the model itself as the judge model
# This is only for demonstration; in practice, you may want to use a more reliable judge model like `gpt-5-nano`
JUDGE_MODEL="gpt-5-nano"
JUDGE_API_BASE_URL="$OPENAI_BASE_URL"
JUDGE_API_KEY="$OPENAI_API_KEY"

declare -A ANN_FILES=(
    ["o3_bench"]="data/O3-Bench/test/metadata.jsonl"
    ["arxiv_long_100"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/export/arxiv_0307_sample_filtered_cs_reduced_pages_min_50_max_100_sample_100/qa_samples_dpi_200.jsonl"
    ["arxiv_long_100_images_maybe_sliced"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/export/arxiv_0307_sample_filtered_cs_reduced_pages_min_50_max_100_sample_100/qa_samples_dpi_200_images_maybe_sliced.jsonl"
    ["arxiv_0307_veqa_batch_0350_mveqa_batch_0352"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_aug_noaug/manifest_sample_50.jsonl"
    ["arxiv_0307_veqa_batch_0350_mveqa_batch_0352_regions"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_regions_incap/manifest_sample_50.jsonl"
)

declare -A IMG_DIRS=(
    ["o3_bench"]="data/O3-Bench/test"
    ["arxiv_long_100"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/export/arxiv_0307_sample_filtered_cs_reduced_pages_min_50_max_100_sample_100/images_dpi_200"
    ["arxiv_long_100_images_maybe_sliced"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/export/arxiv_0307_sample_filtered_cs_reduced_pages_min_50_max_100_sample_100/images_dpi_200"
    ["arxiv_0307_veqa_batch_0350_mveqa_batch_0352"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_aug_noaug/pdf_image"
    ["arxiv_0307_veqa_batch_0350_mveqa_batch_0352_regions"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_regions_incap/pdf_image"
)

# DATASETS=("arxiv_0307_veqa_batch_0350_mveqa_batch_0352")
# IMG_RESCALE_RATIOS=("0.25" "0.375" "0.5" "1.0")
# YARN_FACTORS=("1.0" "2.0")
# YARN_FACTORS=("2.0")

DATASETS=("arxiv_0307_veqa_batch_0350_mveqa_batch_0352_regions")
IMG_RESCALE_RATIOS=("1.0")
YARN_FACTORS=("1.0")

settings=()
for yarn_factor in "${YARN_FACTORS[@]}"; do
  # Serve the model with vLLM running in the background
  SKIP_SERVE=false
  if [ "$SKIP_SERVE" == "false" ]; then
    IMAGE_LIMIT=500
    if [ "$yarn_factor" == "1.0" ]; then
      MAX_MODEL_LEN=$(( 256 * 1024 ))
      source ./insight_o3/scripts/vllm_serve_bg_v2.sh "$MODEL" "$TENSOR_PARALLEL_SIZE" "$MAX_MODEL_LEN" "$IMAGE_LIMIT"
    elif [ "$yarn_factor" == "2.0" ]; then
      MAX_MODEL_LEN=$(( 512 * 1024 ))
      VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 source ./insight_o3/scripts/vllm_serve_bg_v2.sh "$MODEL" "$TENSOR_PARALLEL_SIZE" "$MAX_MODEL_LEN" "$IMAGE_LIMIT" --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings": 262144,"mrope_section":[24,20,20],"mrope_interleaved": true}}'
    else
      echo "Invalid yarn factor: $yarn_factor"
      exit 1
    fi
  fi

  for img_rescale_ratio in "${IMG_RESCALE_RATIOS[@]}"; do
    model_setting="${MODEL}"
    if [ "$yarn_factor" != "1.0" ]; then
      model_setting="${model_setting}_yarn_${yarn_factor}x"
    fi
    setting="${model_setting}/img_rescale_ratio_$img_rescale_ratio"
    settings+=("$setting")
    for dataset in "${DATASETS[@]}"; do
      python -m insight_o3.scripts.evaluate \
        --eval_name "$setting/$dataset" \
        --model "$MODEL" \
        --api_base_url "$API_BASE_URL" \
        --api_key "$API_KEY" \
        --judge_model "$JUDGE_MODEL" \
        --judge_api_base_url "$JUDGE_API_BASE_URL" \
        --judge_api_key "$JUDGE_API_KEY" \
        --ann_file "${ANN_FILES[$dataset]}" \
        --img_dir "${IMG_DIRS[$dataset]}" \
        --img_rescale_ratio "$img_rescale_ratio" \
        --img_max_pixels $((10000*10000))
    done
  done

  # Kill the background vLLM process
  echo "Evaluation complete. Killing background vLLM process..."
  if [[ -n "${vllm_pid:-}" ]]; then
      kill "$vllm_pid"
      wait "$vllm_pid"
  fi
done

# Gather evaluation results
python -m insight_o3.scripts.gather_eval_results \
  --settings "${settings[@]}" \
  --datasets "${DATASETS[@]}"