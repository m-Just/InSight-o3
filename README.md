<p align="center">
  <img alt="InSight-o3" src="assets/banner.png" width="650" style="max-width: 100%;">
</p>

<p align="center">
  <strong>Empowering Multimodal Foundation Models with Generalized Visual Search</strong>
</p>


<div align="center">

🤗 **[O3-Bench](https://huggingface.co/datasets/m-Just/O3-Bench)** |
📄 **[Paper](https://arxiv.org/abs/2512.18745)**

</div>

---

**The ability for AI agents to *"think with images"* requires a sophisticated blend of *reasoning* and *perception*.**
In our work, "[InSight-o3: Empowering Multimodal Foundation Models with Generalized Visual Search](https://arxiv.org/abs/2512.18745)":
- We introduce **O3-Bench**, a new benchmark for multimodal reasoning with interleaved attention to visual details. It tests how well an AI agent can truly "think with images".
- We propose **InSight-o3**, a multi-agent framework consisting of a visual reasoning agent (__*vReasoner*__) and a visual search agent (__*vSearcher*__), to address the challenge presented by O3-Bench through task decomposition and specialization.

> The name, "InSight-o3", reflects its dual role: providing deeper *insight* into multimodal semantics while bringing the target region *in sight* through precise localization with vSearcher.

Here is an example of **InSight-o3** (w/ GPT-5-mini as vReasoner) solving a problem in **O3-Bench**:
<p align="center">
  <img alt="O3-Bench illustration" src="assets/teaser.svg" style="max-width: 100%;">
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

The downloaded data will be placed under `data/O3-Bench`. Then you can run the following command to evaluate your model on O3-Bench:
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

> This code currently only supports evaluating models/systems accessible via a single API call.
The evaluation code for InSight-o3 (which involves interactions between two models) is in preparation.
The training code for InSight-o3 will come soon as well. Please stay tuned!

## Citation

If you find our work useful, please consider citing:
```
@article{li2025insighto3,
  title={InSight-o3: Empowering Multimodal Foundation Models with Generalized Visual Search},
  author={Kaican Li and Lewei Yao and Jiannan Wu and Tiezheng Yu and Jierun Chen and Haoli Bai and Lu Hou and Lanqing Hong and Wei Zhang and Nevin L. Zhang},
  journal={arXiv preprint arXiv:2512.18745},
  year={2025}
}
```
