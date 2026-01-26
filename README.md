<p align="center">
  <img alt="InSight-o3" src="assets/banner.png" width="650" style="max-width: 100%;">
</p>

<p align="center">
  <strong>Empowering Multimodal Foundation Models with Generalized Visual Search</strong>
</p>


<div align="center">

🤗 **[Models and Datasets](https://huggingface.co/collections/m-Just/insight-o3)** |
📄 **[Paper](https://arxiv.org/abs/2512.18745)**

</div>

## What's new
- [x] [2026/1/26] Our paper has been accepted by ICLR 2026! 🥳
- [x] [2026/1/16] InSight-o3 training data released! See [VisCoT_VStar_Collage](https://huggingface.co/datasets/m-Just/VisCoT_VStar_Collage) and [InfoVQA_RegionLocalization](https://huggingface.co/datasets/m-Just/InfoVQA_RegionLocalization).
- [x] [2026/1/12] [InSight-o3 vSearcher model](https://huggingface.co/m-Just/InSight-o3-vS), training \& evaluation code released!
- [x] [2025/12/24] [O3-Bench](https://huggingface.co/datasets/m-Just/O3-Bench) and evaluation code released!

---

**The ability for AI agents to *"think with images"* requires a sophisticated blend of *reasoning* and *perception*.**
In our work, "[InSight-o3: Empowering Multimodal Foundation Models with Generalized Visual Search](https://arxiv.org/abs/2512.18745)":
- We introduce **O3-Bench**, a new benchmark for multimodal reasoning with interleaved attention to visual details. It tests how well an AI agent can truly "think with images".
- We propose **InSight-o3**, a multi-agent framework consisting of a visual reasoning agent (__*vReasoner*__) and a visual search agent (__*vSearcher*__), to address the challenge presented by O3-Bench through task decomposition and specialization.

> The name, "InSight-o3", reflects its dual role: providing deeper *insight* into multimodal semantics while bringing the target region *in sight* through precise localization with vSearcher.

Here is an example of **InSight-o3** (w/ GPT-5-mini as vReasoner) solving a problem in **O3-Bench**:
<p align="center">
  <img alt="O3-Bench illustration" src="assets/teaser.png" style="max-width: 100%;">
</p>

The vSearcher of InSight-o3 aims to solve the task of __*generalized visual search*__—locating *relational*, *fuzzy*, or *conceptual* regions described in *free-form* language, e.g., "the area to the left of the parking lot," and "the chart showing the company's revenue in the last decade," beyond just simple objects or figures in natural images.
We train a multimodal LLM (Qwen2.5-VL-7B) specifically for this task via RL.

⭐ **Performance.** Our vSearcher (named **InSight-o3-vS**) empowers frontier multimodal foundation models (which serve as vReasoners), significantly improving their performance on a wide range of benchmarks including [V*-Bench](https://huggingface.co/datasets/craigwu/vstar_bench) (**73.8%** ➡️ **86.9%** for GPT-5-mini, and **80.1%** ➡️ **87.6%** for Gemini-2.5-Flash) and [O3-Bench](https://huggingface.co/datasets/m-Just/O3-Bench) (**39.0%** ➡️ **61.5%** for GPT-5-mini, and **60.4%** ➡️ **69.7%** for Gemini-2.5-Flash).

## Benchmark
O3-Bench consists of two domains:&nbsp; 📊 __*composite charts*__&nbsp; and &nbsp;🗺️ __*digital maps*__.
They are designed with two key principles in mind:
- **High resolution & high information density.** Images are large, high-resolution, cluttered, and *information-dense*, making evidence gathering *genuinely non-trivial*.
- **Multi-hop solution paths.** Solutions require piecing together *subtle* visual evidence from *distinct* image areas through *multi-step*, *interleaved* reasoning.

Both domains are challenging for current frontier multimodal models/systems, e.g., OpenAI o3's accuracy on O3-Bench is **40.8%** by our evaluation via the official API; in comparison, an *average* human can easily achieve **>95%** accuracy.

The full benchmark results are shown below.
<p align="center">
  <img alt="Benchmark results" src="assets/benchmark.png" width="650" style="max-width: 100%;">
</p>

To account for sampling randomness, the results above are averaged over **3** random trials.
All models/systems are given a **16K** tokens/repsonse budget including reasoning tokens (i.e., `max_completion_tokens=16384`).
The performance gap between GPT and Gemini is partly because OpenAI restricts the input image resolution of GPT models to roughly **1280×1280px** (as per [OpenAI API](https://platform.openai.com/docs/guides/images-vision#calculating-costs)).
For models other than GPT, we use a much higher, **3500×3500px** image resolution.

To reproduce the results or evaluate your own models on O3-Bench, please follow the guide below.

## Evaluation
Our evaluation code uses OpenAI chat completion API to generate responses.
To use our code, make sure you have the following packages installed: `openai`, `numpy`, `pandas`, `pillow`, `tqdm`.

### O3-Bench
You can download O3-Bench with the following command if you have `git-lfs` installed in your system:
```sh
git submodule init data/O3-Bench
git submodule update --remote data/O3-Bench
```
You can visualize the downloaded data using [`notebooks/visualize_o3_bench.ipynb`](notebooks/visualize_o3_bench.ipynb).

To evaluate a model on O3-Bench, run the following command:
```sh
MODEL='<model>'
API_BASE_URL='<api base url>'
API_KEY='<api key>'

JUDGE_MODEL='<judge model>'
JUDGE_API_BASE_URL='<judge api base url>'
JUDGE_API_KEY='<judge api key>'

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
```

You can also manually download the dataset from [HuggingFace](https://huggingface.co/datasets/m-Just/O3-Bench).
Just remember to change `--ann_file` and `--img_dir` accordingly if you place the data at a different location.

**For consistency with our results, we recommend using `gpt-5-nano` as the judge model.**
By default, a *single* trial will be run. You can run multiple trials by specifying `--num_trials <num_trials>`.
Completed trials will be *skipped* instead of *overwritten*.
The evaluation results will be saved under `outputs/eval/<eval_name>`.

**See [`insight_o3/scripts/examples/evaluate_qwen3_vl.sh`](insight_o3/scripts/examples/evaluate_qwen3_vl.sh) for a more concrete example based on self-hosted Qwen3-VL-8B-Instruct using vLLM.**

To visualize the evaluation outputs, see [`notebooks/visualize_output.ipynb`](notebooks/visualize_output.ipynb).

### Other benchmarks
Our code also supports evaluation on other datasets (including all the benchmarks we used in our paper).
If you want to evaluate on those datasets, please follow [`data/README.md`](data/README.md) to prepare the data first.
To reproduce the baseline results in our paper, see [`insight_o3/scripts/examples/sweep_baselines.sh`](insight_o3/scripts/examples/sweep_baselines.sh).


## InSight-o3
We use a modified [verl](https://github.com/volcengine/verl) (with [vLLM](https://github.com/vllm-project/vllm) as the inference engine) for both training and evaluation of InSight-o3.
To get started, grab [the modified verl codebase](https://github.com/m-Just/verl-public/tree/insight_o3) first:
```sh
git submodule init verl
git submodule update --remote verl
```

Then, follow the [installation guide](https://verl.readthedocs.io/en/latest/start/install.html#install-dependencies) to install verl and its dependencies.
We recommend installing the following packages **in these versions**:
```
torch==2.8.0+cu126
vllm==0.10.2
flash_attn==2.8.3
transformers==4.57.3
ray==2.53.0
qwen-vl-utils==0.0.10
openai==2.14.0
```
using the following commands:
```sh
uv pip install vllm==0.10.2 --torch-backend=cu126
uv pip install flash-attn==2.8.3 --no-build-isolation   # this may take a while
uv pip install transformers==4.57.3 ray==2.53.0 qwen-vl-utils==0.0.10 openai==2.14.0
```
Please note that other versions are not tested.

### Data preparation
You can download the training data with the following command:
```sh
git submodule init data/VisCoT_VStar_Collage data/InfoVQA_RegionLocalization
git submodule update --remote data/VisCoT_VStar_Collage data/InfoVQA_RegionLocalization
```

Then, use [`verl/recipe/vsearch/create_parquet_dataset.py`](https://github.com/m-Just/verl-public/blob/insight_o3/recipe/vsearch/create_parquet_dataset.py) to pack the downloaded datasets into [verl-compatible](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html) parquet files.
See **example usages** commented in `create_parquet_dataset.py` for the exact commands to pack the training/evaluation datasets.

Successfully packed datasets should have these columns: `data_source`, `prompt`, `images`, `reward_model`, `extra_info`, and `agent_name`.
In particular, the `images` column stores the file paths of the images, e.g., each row of `images` should look like
```
[{'image': 'file:///path/to/images/0.jpg'}]   # currently only support single-image QA
```

The `reward_model` columns stores the ground truth answer for each row, e.g.,
```
{'ground_truth': 'C', 'style': 'rule'}
```

The `agent_name` indicates the main agent on which the data is to be used.
For training, depending on whether the data is used for the in-loop or out-of-loop subagent RL, `agent_name` should be set to `vreasoner` or `vsearcher`, respectively.

If your dataset has ground-truth bounding boxes, put them as a list of `(x1, y1, x2, y2)` under `bboxes` of `extra_info`. This is (only) **required** for out-of-loop RL.


### Training
After preparing the data, you can start training with the following code snippet:
```sh
# Start at the project root dir and add it to PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Go inside verl and set things up
cd verl
export VERL_PROJ_DIR="$(pwd)"
export MODEL_PATH='Qwen/Qwen2.5-VL-7B-Instruct'

export WORK_DIR='<root path for saving logs, checkpoints, etc.>'
export PROJECT_NAME='my_project'
export EXP_NAME='my_experiment'
# outputs will be saved under "$WORK_DIR/ckpts/$PROJECT_NAME/$EXP_NAME" by default

export API_MODEL_FOR_AGENT='<vReasoner model>'  # e.g., gpt-5-mini
# agent configuration at `recipe/vsearch/config/agent_${API_MODEL_FOR_AGENT}.yaml`
export JUDGE_MODEL='<judge model>'              # e.g., gpt-5-nano
export OPENAI_BASE_URL='<api base url>'
export OPENAI_API_KEY='<api key>'
# `OPENAI_*` settings will be applied to both the vReasoner and the judge model

export TRAIN_FILES='<path(s) to training dataset file(s) (in parquet format)>'
export VAL_FILES='<path(s) to validation dataset file(s) (in parquet format)>'
# multiple dataset files can be concatenated as follows: '[/path/to/dateset_A,/path/to/dateset_B]'

bash recipe/vsearch/train.sh
```

The current training script uses the two training datasets introduced in our paper.
They are mixed in 1:1 ratio as can be seen from the following part of `recipe/vsearch/train.sh`:
```
  +data.batch_sampler.weights.info_vqa_region_localization=0.5 \
  +data.batch_sampler.weights.visual_cot_vstar_collage=0.5
```
To use your own training datasets, you need to replace the name after `+data.batch_sampler.weights.` with the name you put in the `data_source` field of your training data parquet files.

Auto-resuming is enabled by default. You can always run the same command to resume training.
To resume from a specific training step (instead of the latest checkpointed step), use `export RESUME_FROM_STEP=<step>`.

More detailed configurations for training and evaluation can be found in `recipe/vsearch/_base.sh` and `recipe/vsearch/config/qwen_2_5_vl_7b_async.yaml`. Feel free to open issues if there's anything unclear!

### Evaluation
For evaluation, simply change the above snippet for training as follows:
- Change the launching script to `recipe/vsearch/val.sh`.
- Change `MODEL_PATH` if needed (e.g., to [`m-Just/InSight-o3-vS`](https://huggingface.co/m-Just/InSight-o3-vS) for our vSearcher model).
- Change `VAL_FILES` if needed.
- Optionally, add `export NUM_VAL_TRIALS='<number of evaluation trials to run>'`.

There are three key result metrics:
- `accuracy_reward`: this is the accuracy on the evaluation dataset.
- `critical_failure_ratio`: this is the ratio of failed queries due to API errors.
- `has_answer`: this is the ratio of queries for which the agent has successfully generated an answer.

Since the evaluation is partly based on API, **there will be randomness in the results** (the fluctuation can be huge sometimes).
We recommend setting `NUM_VAL_TRIALS` to at least 3 and computing the average for more reliable results.

By default, verl will look for checkpoints under `trainer.default_local_dir` (set to `$WORK_DIR/ckpts/$PROJECT_NAME/$EXP_NAME`) and try to load the latest checkpoint.
If this is not the desired behavior, e.g., you just want to evaluate the original model specified by `MODEL_PATH`, you can either change `trainer.default_local_dir` or turn auto-resuming off.

More detailed configurations for training and evaluation can be found in `recipe/vsearch/_base.sh` and `recipe/vsearch/config/qwen_2_5_vl_7b_async.yaml`. Feel free to open issues if there's anything unclear!

## Citation

If you find our work useful, please consider citing:
```
@inproceedings{li2026insight_o3,
  title={InSight-o3: Empowering Multimodal Foundation Models with Generalized Visual Search},
  author={Kaican Li and Lewei Yao and Jiannan Wu and Tiezheng Yu and Jierun Chen and Haoli Bai and Lu Hou and Lanqing Hong and Wei Zhang and Nevin L. Zhang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
