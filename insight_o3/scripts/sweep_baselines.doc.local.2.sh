#!/bin/bash

# ------------------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------------------

# MODELS=('gpt-4o' 'gpt-5-nano' 'gpt-5-mini' 'gemini-2.5-flash')
# MODELS=('gpt-5-mini' 'gemini-3-flash-preview')
MODELS=('gpt-5.4-mini')
# MODELS=('gemini-3-flash-preview')

API_BASE_URL=$OPENAI_BASE_URL
API_KEY=$OPENAI_API_KEY

JUDGE_MODEL='gpt-5-nano'
JUDGE_API_BASE_URL=$OPENAI_BASE_URL
JUDGE_API_KEY=$OPENAI_API_KEY

# DATASETS=('vstar' 'hr_bench_4k' 'treebench' 'visual_probe_hard' 'mme_realworld_lite' 'o3_bench')
# DATASETS=('arxiv_0307_veqa_batch_0350_mveqa_batch_0352')
# DATASETS=('arxiv_0307_veqa_batch_0350_mveqa_batch_0352_maxp40')
DATASETS=('arxiv_0307_veqa_batch_0350_mveqa_batch_0352_regions')
IMG_RESCALE_RATIOS=("1.0")
# IMG_RESCALE_RATIOS_GPT=("0.25" "0.375" "0.5")
# IMG_RESCALE_RATIOS=("0.1")
IMG_RESCALE_RATIOS_GPT=("1.0")

declare -A ANN_FILES=(
    ["vstar"]="./data/vstar_bench/vstar_bench.jsonl"
    ["hr_bench_4k"]="./data/HR-Bench/hrbench_4k.jsonl"
    ["treebench"]="./data/TreeBench/treebench.jsonl"
    ["visual_probe_hard"]="./data/VisualProbe_Hard/visual_probe_hard.jsonl"
    ["mme_realworld_lite"]="./data/MME-RealWorld-Lite/MME_RealWorld_Lite.jsonl"
    ["o3_bench"]="./data/O3-Bench/test/metadata.jsonl"
    ["arxiv_long_100"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/export/arxiv_0307_sample_filtered_cs_reduced_pages_min_50_max_100_sample_100/qa_samples_dpi_200.jsonl"
    ["arxiv_0307_veqa_batch_0350_mveqa_batch_0352"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_aug_noaug/manifest_sample_50.jsonl"
    ["arxiv_0307_veqa_batch_0350_mveqa_batch_0352_maxp40"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_aug_noaug_maxp40/manifest_sample_50.jsonl"
    ["arxiv_0307_veqa_batch_0350_mveqa_batch_0352_regions"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_regions_incap/manifest_sample_50.jsonl"
)

declare -A IMG_DIRS=(
    ["vstar"]="./data/vstar_bench"
    ["hr_bench_4k"]="./data/HR-Bench/images_4k"
    ["treebench"]="./data/TreeBench/images"
    ["visual_probe_hard"]="./data"
    ["mme_realworld_lite"]="./data/MME-RealWorld-Lite/imgs"
    ["o3_bench"]="./data/O3-Bench/test"
    ["arxiv_long_100"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/export/arxiv_0307_sample_filtered_cs_reduced_pages_min_50_max_100_sample_100/images_dpi_200"
    ["arxiv_0307_veqa_batch_0350_mveqa_batch_0352"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_aug_noaug/pdf_image"
    ["arxiv_0307_veqa_batch_0350_mveqa_batch_0352_maxp40"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_aug_noaug_maxp40/pdf_image"
    ["arxiv_0307_veqa_batch_0350_mveqa_batch_0352_regions"]="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/veqa_batch_0350_mveqa_batch_0352/dpi200_regions_incap/pdf_image"
)

NUM_TRIALS=1


# ------------------------------------------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------------------------------------------

settings=()
for model in "${MODELS[@]}"; do
    if [[ "$model" == *"gpt"* ]]; then
        img_rescale_ratios=("${IMG_RESCALE_RATIOS_GPT[@]}")
    else
        img_rescale_ratios=("${IMG_RESCALE_RATIOS[@]}")
    fi

    for img_rescale_ratio in "${img_rescale_ratios[@]}"; do
        # Model-specific settings
        if [[ "$model" == *"gpt"* ]]; then
            img_max_pixels=$((1280*1280))
        else
            img_max_pixels=$((3500*3500))
        fi

        if [[ "$model" == *"gemini"* ]]; then
            extra_args=("--separate_trial_requests")
        fi

        setting="$model/img_rescale_ratio_$img_rescale_ratio"
        settings+=("$setting")

        # Run evaluation on each dataset
        for dataset in "${DATASETS[@]}"; do
            ann_file="${ANN_FILES[$dataset]}"
            image_dir="${IMG_DIRS[$dataset]}"
            eval_name="${setting}/${dataset}"

            echo "=================================================="
            echo "Running evaluation for $eval_name"
            echo "=================================================="

            python -m insight_o3.scripts.evaluate \
                --eval_name "$eval_name" \
                --model "$model" \
                --api_base_url "$API_BASE_URL" \
                --api_key "$API_KEY" \
                --judge_model "$JUDGE_MODEL" \
                --judge_api_base_url "$JUDGE_API_BASE_URL" \
                --judge_api_key "$JUDGE_API_KEY" \
                --ann_file "$ann_file" \
                --img_dir "$image_dir" \
                --img_rescale_ratio "$img_rescale_ratio" \
                --img_max_pixels "$img_max_pixels" \
                --num_trials "$NUM_TRIALS" \
                "${extra_args[@]}"

            echo "Evaluation completed for $eval_name"
        done
    done
done

echo "Sweep completed. Gathering evaluation results..."
python -m insight_o3.scripts.gather_eval_results \
    --settings "${settings[@]}" \
    --datasets "${DATASETS[@]}"