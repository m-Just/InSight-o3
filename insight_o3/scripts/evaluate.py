import argparse
import asyncio
from dataclasses import dataclass, asdict
from pprint import pprint
from pathlib import Path
from collections import defaultdict
import os
import re
import json
from time import perf_counter

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN
import pandas as pd
import numpy as np
from tqdm import tqdm

import insight_o3.prompts as prompts
from insight_o3.inference import query_api_vqa, InferenceResult
from insight_o3.inference_qwen_agent import query_qwen_agent_vqa
from insight_o3.utils.api import create_async_openai_client, query_api


@dataclass
class EvalResult:
    sample_index: int
    success: bool
    fail_reason: str | None = None
    inference_result: InferenceResult | None = None
    extracted_answer: str | None = None
    is_correct: bool = False
    profile: dict[str, float] | None = None


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive float, got {value}")
    return parsed


def get_eval_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir).resolve() / args.eval_name


def get_eval_trial_dir(trial_id: int, args: argparse.Namespace) -> Path:
    return get_eval_dir(args) / f"trial_{trial_id}"


def resolve_sample_image_paths(sample: dict, img_dir: str) -> list[str]:
    if 'images' in sample:
        images = sample['images']
        if not isinstance(images, list) or not images:
            raise ValueError("sample['images'] must be a non-empty list")
        return [str(Path(img_dir) / image_path) for image_path in images]

    image_path = sample['image'] if 'image' in sample else sample['file_name']
    return [str(Path(img_dir) / image_path)]


def init_token_usage() -> dict[str, int]:
    return {
        'input_tokens': 0,
        'cached_input_tokens': 0,
        'output_tokens': 0,
        'reasoning_tokens': 0,
        'api_calls_with_usage': 0,
    }


def add_token_usage(total: dict[str, int], delta: dict[str, int] | None) -> None:
    if not delta:
        return
    for key in total:
        total[key] += int(delta.get(key, 0))


def finalize_token_usage_summary(token_usage: dict[str, int]) -> dict[str, int | float]:
    api_calls_with_usage = token_usage['api_calls_with_usage']
    input_tokens = token_usage['input_tokens']
    output_tokens = token_usage['output_tokens']
    cached_input_tokens = token_usage['cached_input_tokens']
    reasoning_tokens = token_usage['reasoning_tokens']
    return {
        **token_usage,
        'scope': 'main_model_requests_executed_in_this_invocation',
        'cached_input_ratio': (cached_input_tokens / input_tokens) if input_tokens else 0.0,
        'reasoning_ratio': (reasoning_tokens / output_tokens) if output_tokens else 0.0,
        'avg_input_tokens_per_request': (input_tokens / api_calls_with_usage) if api_calls_with_usage else 0.0,
        'avg_output_tokens_per_request': (output_tokens / api_calls_with_usage) if api_calls_with_usage else 0.0,
    }


def attribute_token_usage_to_record(token_usage: dict[str, int] | None, num_records: int) -> dict[str, int | float] | None:
    if not token_usage:
        return None
    if num_records <= 0:
        raise ValueError(f"Expected a positive number of records, got {num_records}")

    return {
        'scope': 'main_model_request_usage_attributed_to_this_eval_record',
        'request_shared_across_trial_records': (num_records > 1),
        'num_trial_records_sharing_request': num_records,
        'input_tokens': token_usage['input_tokens'] / num_records,
        'cached_input_tokens': token_usage['cached_input_tokens'] / num_records,
        'output_tokens': token_usage['output_tokens'] / num_records,
        'reasoning_tokens': token_usage['reasoning_tokens'] / num_records,
        'api_calls_with_usage': token_usage['api_calls_with_usage'] / num_records,
    }


def flatten_numeric_metrics(data: dict, prefix: str = "") -> dict[str, float]:
    metrics = {}
    for key, value in data.items():
        metric_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            metrics.update(flatten_numeric_metrics(value, metric_name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[metric_name] = float(value)
    return metrics


def summarize_numeric_values(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    return {
        'count': int(arr.size),
        'total': float(arr.sum()),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


def summarize_profiling(records: list[dict], wall_time_seconds: float) -> dict:
    metrics = defaultdict(list)
    for record in records:
        eval_result = record['eval_result']
        eval_profile = eval_result.get('profile') or {}
        inference_profile = (eval_result.get('inference_result') or {}).get('profile') or {}
        profile = {
            'eval': eval_profile,
            'inference': inference_profile,
        }
        for metric_name, value in flatten_numeric_metrics(profile).items():
            metrics[metric_name].append(value)

    return {
        'num_records': len(records),
        'wall_time_seconds': wall_time_seconds,
        'records_per_second': (len(records) / wall_time_seconds) if wall_time_seconds else 0.0,
        'timing_stats': {
            metric_name: summarize_numeric_values(values)
            for metric_name, values in sorted(metrics.items())
        },
    }


def format_rate(value: float | dict[str, float]) -> str:
    if isinstance(value, dict):
        return f"{value['mean'] * 100:.1f}% ± {value['std'] * 100:.1f}%"
    return f"{value * 100:.1f}%"


def format_scalar(value: float | dict[str, float], suffix: str = "") -> str:
    if isinstance(value, dict):
        return f"{value['mean']:.3f}{suffix} ± {value['std']:.3f}{suffix}"
    return f"{value:.3f}{suffix}"


def normalize_message_content(content: object) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text = item.get('text')
                if isinstance(text, str):
                    text_parts.append(text)
        return ''.join(text_parts) if text_parts else None
    return str(content)


def print_trial_metrics(trial_id: int, metrics: dict) -> None:
    print(f"Evaluation results (trial {trial_id}):")
    print(
        "  Overall:"
        f" completion={format_rate(metrics['overall_completion_rate'])}"
        f", success={format_rate(metrics['overall_success_rate'])}"
        f", accuracy={format_rate(metrics['overall_accuracy'])}"
    )

    overall_stats = metrics['overall_stats']
    print(
        "  Counts:"
        f" correct={overall_stats['correct']}"
        f", wrong={overall_stats['wrong']}"
        f", incomplete={overall_stats['incomplete']}"
        f", invalid={overall_stats['invalid']}"
        f", total={overall_stats['total']}"
    )

    print("  Categories:")
    for category in sorted(metrics['category_accuracy']):
        category_stats = metrics['category_stats'][category]
        print(
            f"    {category}:"
            f" completion={format_rate(metrics['category_completion_rate'][category])}"
            f", success={format_rate(metrics['category_success_rate'][category])}"
            f", accuracy={format_rate(metrics['category_accuracy'][category])}"
            f", counts={category_stats}"
        )

    profiling = metrics.get('profiling')
    if not profiling:
        return

    print("  Profiling:")
    print(
        f"    wall_time={format_scalar(profiling['wall_time_seconds'], 's')},"
        f" throughput={format_scalar(profiling['records_per_second'], ' rec/s')}"
    )
    timing_stats = profiling.get('timing_stats', {})
    metric_order = [
        'inference.image_preprocess.total_seconds',
        'inference.request_wait_seconds',
        'inference.response_parse_seconds',
        'eval.answer_parse_seconds',
        'eval.answer_extraction_request_wait_seconds',
        'eval.judge_request_wait_seconds',
        'eval.total_seconds',
    ]
    for metric_name in metric_order:
        if metric_name not in timing_stats:
            continue
        stats = timing_stats[metric_name]
        print(
            f"    {metric_name}:"
            f" mean={stats['mean']:.3f}s,"
            f" median={stats['median']:.3f}s,"
            f" max={stats['max']:.3f}s,"
            f" total={stats['total']:.3f}s"
        )


def print_summary_metrics(summary_metrics: dict) -> None:
    print(f"Summary metrics over {summary_metrics['num_trials']} trial(s):")
    print(
        "  Overall:"
        f" completion={format_rate(summary_metrics['overall_completion_rate'])}"
        f", success={format_rate(summary_metrics['overall_success_rate'])}"
        f", accuracy={format_rate(summary_metrics['overall_accuracy'])}"
    )

    print("  Categories:")
    for category in sorted(summary_metrics['category_accuracy']):
        print(
            f"    {category}:"
            f" completion={format_rate(summary_metrics['category_completion_rate'][category])}"
            f", success={format_rate(summary_metrics['category_success_rate'][category])}"
            f", accuracy={format_rate(summary_metrics['category_accuracy'][category])}"
        )

    profiling = summary_metrics.get('profiling')
    if profiling:
        print("  Profiling:")
        print(
            f"    wall_time={format_scalar(profiling['wall_time_seconds'], 's')},"
            f" throughput={format_scalar(profiling['records_per_second'], ' rec/s')}"
        )
        timing_stats = profiling.get('timing_stats', {})
        metric_order = [
            'inference.image_preprocess.total_seconds',
            'inference.request_wait_seconds',
            'inference.response_parse_seconds',
            'eval.answer_parse_seconds',
            'eval.answer_extraction_request_wait_seconds',
            'eval.judge_request_wait_seconds',
            'eval.total_seconds',
        ]
        for metric_name in metric_order:
            if metric_name not in timing_stats or 'mean' not in timing_stats[metric_name]:
                continue
            mean_stats = timing_stats[metric_name]['mean']
            print(f"    {metric_name}: {format_scalar(mean_stats, 's')}")

    token_usage = summary_metrics.get('main_model_token_usage')
    if not token_usage:
        return

    print("  Main model token usage:")
    if token_usage['api_calls_with_usage'] == 0:
        print("    No usage data returned by the API.")
        return
    print(
        f"    input={token_usage['input_tokens']}"
        f" ({token_usage['cached_input_ratio']:.1%} cached),"
        f" output={token_usage['output_tokens']}"
        f" ({token_usage['reasoning_ratio']:.1%} reasoning)"
    )
    print(
        f"    avg/request:"
        f" input={token_usage['avg_input_tokens_per_request']:.0f},"
        f" output={token_usage['avg_output_tokens_per_request']:.0f},"
        f" requests={token_usage['api_calls_with_usage']}"
    )


async def process_sample(
    sample: dict,
    args: argparse.Namespace,
    trial_ids: list[int],
    client_main: AsyncOpenAI | None,
    client_judge: AsyncOpenAI,
) -> tuple[list[EvalResult], dict[str, int]]:
    index = sample['sample_index']
    image_paths = resolve_sample_image_paths(sample, args.img_dir)
    question = sample['question']
    options = sample.get('options', None)
    user_prompt = f"{question}\n{options}" if options else question
    ground_truth = sample['answer']

    if args.sys_prompt == 'model_default':
        system_prompt = None
    elif args.sys_prompt == 'think':
        system_prompt = prompts.SIMPLE_SYSTEM_PROMPT_THINKING
    elif args.sys_prompt == 'qwen_agent_analysis':
        system_prompt = prompts.QWEN_AGENT_ANALYSIS_PROMPT
    else:
        raise ValueError(f"Invalid system prompt: {args.sys_prompt}")

    if args.backend == 'qwen_agent':
        generate_cfg = {**args.qwen_agent_generate_cfg}
        mct = args.chat_completion_kwargs.get('max_completion_tokens')
        if mct is not None:
            generate_cfg.setdefault('max_completion_tokens', mct)

        inference_results, token_usage = await query_qwen_agent_vqa(
            image_paths=image_paths,
            user_prompt=user_prompt,
            model=args.model,
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            image_rescale_ratio=args.img_rescale_ratio,
            image_format=args.img_format,
            image_max_pixels=args.img_max_pixels or None,
            system_prompt=system_prompt,
            model_type=args.qwen_agent_model_type,
            tools=args.qwen_agent_tools,
            generate_cfg=generate_cfg,
            max_retries=args.qwen_agent_max_retries,
        )
    else:
        chat_completion_kwargs = {
            'n': len(trial_ids),
            **args.chat_completion_kwargs,
        }
        inference_results, token_usage = await query_api_vqa(
            image_paths=image_paths,
            user_prompt=user_prompt,
            model=args.model,
            client=client_main,
            image_rescale_ratio=args.img_rescale_ratio,
            image_format=args.img_format,
            image_max_pixels=args.img_max_pixels or None,
            image_url_extra_settings=args.image_url_extra_settings,
            system_prompt=system_prompt,
            **chat_completion_kwargs,
        )

    # Extract answers from the inference results and check if they are correct
    eval_results = []
    for result in inference_results:
        eval_start = perf_counter()

        if not result.success:
            eval_results.append(EvalResult(
                sample_index=index,
                success=False,
                fail_reason='inference_failed',
                inference_result=result,
                profile={
                    'answer_parse_seconds': 0.0,
                    'answer_extraction_request_wait_seconds': 0.0,
                    'judge_request_wait_seconds': 0.0,
                    'total_seconds': perf_counter() - eval_start,
                },
            ))
            continue

        if result.finish_reason != 'stop':
            eval_results.append(EvalResult(
                sample_index=index,
                success=False,
                fail_reason=f'finish_reason:{result.finish_reason}',
                inference_result=result,
                profile={
                    'answer_parse_seconds': 0.0,
                    'answer_extraction_request_wait_seconds': 0.0,
                    'judge_request_wait_seconds': 0.0,
                    'total_seconds': perf_counter() - eval_start,
                },
            ))
            continue

        # Try rule-based answer extraction first
        answer_parse_start = perf_counter()
        if not isinstance(result.last_message_content, str):
            eval_results.append(EvalResult(
                sample_index=index,
                success=False,
                fail_reason='empty_model_response',
                inference_result=result,
                profile={
                    'answer_parse_seconds': perf_counter() - answer_parse_start,
                    'answer_extraction_request_wait_seconds': 0.0,
                    'judge_request_wait_seconds': 0.0,
                    'total_seconds': perf_counter() - eval_start,
                },
            ))
            continue
        matches = re.findall(r'<answer>(.*?)</answer>', result.last_message_content, re.DOTALL)
        raw_answer_str = matches[-1].strip() if matches else result.last_message_content
        matches = re.findall(r'\\boxed\{(.*?)\}', raw_answer_str, re.DOTALL)
        raw_answer_str = matches[-1].strip() if matches else raw_answer_str
        answer_parse_seconds = perf_counter() - answer_parse_start
        answer_extraction_request_wait_seconds = 0.0
        judge_request_wait_seconds = 0.0

        # Correct if the extracted answer matches the ground truth verbatim
        if raw_answer_str == ground_truth:
            eval_results.append(EvalResult(
                sample_index=index,
                success=True,
                inference_result=result,
                extracted_answer=raw_answer_str,
                is_correct=True,
                profile={
                    'answer_parse_seconds': answer_parse_seconds,
                    'answer_extraction_request_wait_seconds': answer_extraction_request_wait_seconds,
                    'judge_request_wait_seconds': judge_request_wait_seconds,
                    'total_seconds': perf_counter() - eval_start,
                },
            ))
            continue
        
        # Extract answer using judge model
        if len(raw_answer_str) > args.max_judge_ans_context_len:
            crop_marker = ' ... '
            kept_chars = max(1, args.max_judge_ans_context_len - len(crop_marker))
            prefix_chars = kept_chars // 2
            suffix_chars = kept_chars - prefix_chars
            raw_answer_str = (
                f"{raw_answer_str[:prefix_chars]}"
                f"{crop_marker}"
                f"{raw_answer_str[-suffix_chars:]}"
            )
            
        if options:
            answer_extraction_prompt = prompts.GPT_EVAL_MCQA_PROMPT.format(
                question=question,
                options=options,
                model_answer=raw_answer_str,
            )
        else:
            answer_extraction_prompt = prompts.GPT_EVAL_OPEN_QA_PROMPT.format(
                question=question,
                model_response=raw_answer_str,
            )

        try:
            answer_extraction_request_start = perf_counter()
            _, response = await query_api(
                query=answer_extraction_prompt,
                model=args.judge_model,
                client=client_judge,
                max_completion_tokens=2048,
            )
            answer_extraction_request_wait_seconds = perf_counter() - answer_extraction_request_start
            extracted_answer = normalize_message_content(response.choices[0].message.content)
        except Exception as e:
            answer_extraction_request_wait_seconds = perf_counter() - answer_extraction_request_start
            print(f"WARNING: Failed to extract answer: {e}")
            eval_results.append(EvalResult(
                sample_index=index,
                success=False,
                fail_reason='answer_extraction_failed',
                inference_result=result,
                profile={
                    'answer_parse_seconds': answer_parse_seconds,
                    'answer_extraction_request_wait_seconds': answer_extraction_request_wait_seconds,
                    'judge_request_wait_seconds': judge_request_wait_seconds,
                    'total_seconds': perf_counter() - eval_start,
                },
            ))
            continue

        if extracted_answer is None:
            print("WARNING: Failed to extract answer: empty judge response content")
            eval_results.append(EvalResult(
                sample_index=index,
                success=False,
                fail_reason='answer_extraction_empty',
                inference_result=result,
                profile={
                    'answer_parse_seconds': answer_parse_seconds,
                    'answer_extraction_request_wait_seconds': answer_extraction_request_wait_seconds,
                    'judge_request_wait_seconds': judge_request_wait_seconds,
                    'total_seconds': perf_counter() - eval_start,
                },
            ))
            continue

        # If the question is an MCQ, check if the extracted answer matches the ground truth verbatim
        if options:
            extracted_answer = extracted_answer.strip()
            eval_results.append(EvalResult(
                sample_index=index,
                success=True,
                inference_result=result,
                extracted_answer=extracted_answer,
                is_correct=(extracted_answer == ground_truth),
                profile={
                    'answer_parse_seconds': answer_parse_seconds,
                    'answer_extraction_request_wait_seconds': answer_extraction_request_wait_seconds,
                    'judge_request_wait_seconds': judge_request_wait_seconds,
                    'total_seconds': perf_counter() - eval_start,
                },
            ))
            continue

        # Otherwise (the question is not an MCQ), check with the judge model
        judge_prompt = prompts.GPT_JUDGE_ANSWER_PROMPT.format(
            question=question,
            gt_answer=ground_truth,
            model_answer=extracted_answer,
        )

        try:
            judge_request_start = perf_counter()
            _, judge_response = await query_api(
                query=judge_prompt,
                model=args.judge_model,
                client=client_judge,
                max_completion_tokens=2048,
            )
            judge_request_wait_seconds = perf_counter() - judge_request_start
            judge_content = normalize_message_content(judge_response.choices[0].message.content)
        except Exception as e:
            judge_request_wait_seconds = perf_counter() - judge_request_start
            print(f"WARNING: Failed to judge answer: {e}")
            eval_results.append(EvalResult(
                sample_index=index,
                success=False,
                fail_reason='judge_failed',
                inference_result=result,
                profile={
                    'answer_parse_seconds': answer_parse_seconds,
                    'answer_extraction_request_wait_seconds': answer_extraction_request_wait_seconds,
                    'judge_request_wait_seconds': judge_request_wait_seconds,
                    'total_seconds': perf_counter() - eval_start,
                },
            ))
            continue

        if judge_content is None:
            print("WARNING: Failed to judge answer: empty judge response content")
            eval_results.append(EvalResult(
                sample_index=index,
                success=False,
                fail_reason='judge_empty',
                inference_result=result,
                profile={
                    'answer_parse_seconds': answer_parse_seconds,
                    'answer_extraction_request_wait_seconds': answer_extraction_request_wait_seconds,
                    'judge_request_wait_seconds': judge_request_wait_seconds,
                    'total_seconds': perf_counter() - eval_start,
                },
            ))
            continue

        eval_results.append(EvalResult(
            sample_index=index,
            success=True,
            inference_result=result,
            extracted_answer=extracted_answer,
            is_correct=("correct" in judge_content.lower()),
            profile={
                'answer_parse_seconds': answer_parse_seconds,
                'answer_extraction_request_wait_seconds': answer_extraction_request_wait_seconds,
                'judge_request_wait_seconds': judge_request_wait_seconds,
                'total_seconds': perf_counter() - eval_start,
            },
        ))

    return eval_results, token_usage


async def main(args: argparse.Namespace, trial_ids: list[int], client_main: AsyncOpenAI | None, client_judge: AsyncOpenAI):
    main_start = perf_counter()
    token_usage_total = init_token_usage()
    # Load samples
    if not Path(args.ann_file).exists():
        raise FileNotFoundError(f"Annotation file not found: {args.ann_file}")
    samples = pd.read_json(args.ann_file, lines=True).to_dict(orient='records')
    for idx, sample in enumerate(samples):
        sample['sample_index'] = idx

    # Evaluate samples concurrently
    trials = defaultdict(list)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def handle_sample(sample: dict) -> tuple[int, tuple[list[EvalResult], dict[str, int]]]:
        async with semaphore:
            results = await process_sample(sample, args, trial_ids, client_main, client_judge)
        return sample['sample_index'], results

    tasks = [asyncio.create_task(handle_sample(sample)) for sample in samples]

    for i, task in enumerate(tqdm(asyncio.as_completed(tasks), total=len(samples), desc="Evaluating")):
        sample_index, result_bundle = await task
        eval_results, token_usage = result_bundle
        add_token_usage(token_usage_total, token_usage)
        sample = samples[sample_index]
        per_record_token_usage = attribute_token_usage_to_record(token_usage, len(trial_ids))

        for trial_id, eval_result in zip(trial_ids, eval_results, strict=True):
            record = {**sample, 'eval_result': asdict(eval_result)}
            if per_record_token_usage is not None:
                record['main_model_token_usage'] = per_record_token_usage
            trials[trial_id].append(record)

        display_result = next((er for er in eval_results if er.success), None)
        if display_result and i % 10 == 0:
            print(f"Sample index: {sample.get('sample_index')}")
            print(f"Question id: {sample.get('question_id')}")
            if conversations := display_result.inference_result.conversations:
                print(f"Conversation(s):")
                for conversation in conversations:
                    print(conversation)
            print(f"Ground truth: {sample.get('answer')}, Predicted: {display_result.extracted_answer}")
            print(f"Is correct: {display_result.is_correct}")
            print("-" * 50)

    # Restore original sample order
    for trial_id, records in trials.items():
        records.sort(key=lambda x: x['sample_index'])
        for record in records:
            del record['sample_index']

    # Calculate statistics and save results
    keys = ('correct', 'wrong', 'incomplete', 'invalid', 'total')

    def compute_success_rate(stats: dict) -> float:
        return (stats['correct'] + stats['wrong']) / stats['total']

    def compute_completion_rate(stats: dict) -> float:
        valid = stats['correct'] + stats['wrong'] + stats['incomplete']
        return (stats['correct'] + stats['wrong']) / valid if valid else 0

    def compute_accuracy(stats: dict) -> float:
        return stats['correct'] / stats['total']
    
    trial_wall_time_seconds = perf_counter() - main_start
    for trial_id, records in trials.items():
        category_stats = defaultdict(lambda: {k: 0 for k in keys})
        for record in records:
            category = record.get('category', record.get('subset', 'unknown'))
            if record['eval_result']['success']:
                category_stats[category]['correct' if record['eval_result']['is_correct'] else 'wrong'] += 1
            elif record['eval_result']['inference_result']['success']:
                finish_reason = record['eval_result']['inference_result']['finish_reason']
                category_stats[category]['invalid' if finish_reason == 'stop' else 'incomplete'] += 1
            else:
                category_stats[category]['invalid'] += 1
            category_stats[category]['total'] += 1
        overall_stats = {k: sum(stats[k] for stats in category_stats.values()) for k in keys}

        accuracy_results = {
            'overall_success_rate': compute_success_rate(overall_stats),
            'overall_completion_rate': compute_completion_rate(overall_stats),
            'overall_accuracy': compute_accuracy(overall_stats),
            'overall_stats': overall_stats,
            'category_success_rate': {c: compute_success_rate(s) for c, s in category_stats.items()},
            'category_completion_rate': {c: compute_completion_rate(s) for c, s in category_stats.items()},
            'category_accuracy': {c: compute_accuracy(s) for c, s in category_stats.items()},
            'category_stats': dict(category_stats),
            'profiling': summarize_profiling(records, trial_wall_time_seconds),
        }

        print_trial_metrics(trial_id, accuracy_results)

        trial_dir = get_eval_trial_dir(trial_id, args)

        with open(trial_dir / 'eval_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(accuracy_results, f, ensure_ascii=False, indent=4)
        print(f"Evaluation metrics (trial {trial_id}) saved to {trial_dir / 'eval_metrics.json'}")

        with open(trial_dir / 'eval_records.jsonl', 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"Evaluation records (trial {trial_id}) saved to {trial_dir / 'eval_records.jsonl'}")

        trial_done_file = trial_dir / 'done'
        trial_done_file.touch()
        print(f"Trial {trial_id} completed")

    return finalize_token_usage_summary(token_usage_total)


def summarize_over_trials(args: argparse.Namespace) -> dict:
    """Summarize metrics across trials.

    - Overall metrics are averaged across trials.
    - Category metrics are averaged per-category across trials.
    """
    if args.num_trials <= 0:
        raise ValueError(f"Number of trials must be greater than 0, got {args.num_trials}")

    overall_success_rates = []
    overall_completion_rates = []
    overall_accuracies = []
    category_success_rates = defaultdict(list)
    category_completion_rates = defaultdict(list)
    category_accuracies = defaultdict(list)
    profiling_wall_times = []
    profiling_records_per_second = []
    profiling_timing_stats = defaultdict(lambda: defaultdict(list))

    for trial_id in range(args.num_trials):
        trial_dir = get_eval_trial_dir(trial_id, args)
        metrics_path = trial_dir / 'eval_metrics.json'
        if not metrics_path.exists():
            raise FileNotFoundError(f"Evaluation metrics not found for trial {trial_id}")

        with open(metrics_path, 'r', encoding='utf-8') as f:
            trial_metrics = json.load(f)

        overall_success_rates.append(trial_metrics['overall_success_rate'])
        overall_completion_rates.append(trial_metrics['overall_completion_rate'])
        overall_accuracies.append(trial_metrics['overall_accuracy'])

        for category, value in trial_metrics['category_success_rate'].items():
            category_success_rates[category].append(value)
        for category, value in trial_metrics['category_completion_rate'].items():
            category_completion_rates[category].append(value)
        for category, value in trial_metrics['category_accuracy'].items():
            category_accuracies[category].append(value)
        if profiling := trial_metrics.get('profiling'):
            profiling_wall_times.append(profiling['wall_time_seconds'])
            profiling_records_per_second.append(profiling['records_per_second'])
            for metric_name, metric_stats in profiling.get('timing_stats', {}).items():
                for stat_name, value in metric_stats.items():
                    profiling_timing_stats[metric_name][stat_name].append(value)

    def compute_stats(values: list[float]) -> dict:
        return {'mean': float(np.mean(values)), 'std': float(np.std(values))}

    summary: dict = {'num_trials': args.num_trials}
    summary['overall_success_rate'] = compute_stats(overall_success_rates)
    summary['overall_completion_rate'] = compute_stats(overall_completion_rates)
    summary['overall_accuracy'] = compute_stats(overall_accuracies)
    summary['category_success_rate'] = {c: compute_stats(values) for c, values in category_success_rates.items()}
    summary['category_completion_rate'] = {c: compute_stats(values) for c, values in category_completion_rates.items()}
    summary['category_accuracy'] = {c: compute_stats(values) for c, values in category_accuracies.items()}
    if profiling_wall_times:
        summary['profiling'] = {
            'wall_time_seconds': compute_stats(profiling_wall_times),
            'records_per_second': compute_stats(profiling_records_per_second),
            'timing_stats': {
                metric_name: {
                    stat_name: compute_stats(values)
                    for stat_name, values in metric_stats.items()
                }
                for metric_name, metric_stats in profiling_timing_stats.items()
            },
        }

    return summary


async def run_trials(args: argparse.Namespace, trials_to_run: list[int]):
    if args.backend == 'api':
        client_main = create_async_openai_client(
            api_key=args.api_key,
            base_url=args.api_base_url,
            timeout=NOT_GIVEN if args.client_timeout is None else args.client_timeout,
        )
    else:
        client_main = None

    client_judge = create_async_openai_client(
        api_key=args.judge_api_key,
        base_url=args.judge_api_base_url,
        timeout=NOT_GIVEN if args.client_timeout is None else args.client_timeout,
    )

    token_usage_total = init_token_usage()
    try:
        if args.separate_trial_requests:
            for trial_id in trials_to_run:
                print(f"Running trial {trial_id}")
                token_usage = await main(args, [trial_id], client_main, client_judge)
                add_token_usage(token_usage_total, token_usage)
        else:
            print(f"Running trials {trials_to_run} together")
            token_usage = await main(args, trials_to_run, client_main, client_judge)
            add_token_usage(token_usage_total, token_usage)
    finally:
        clients_to_close = [client_judge]
        if client_main is not None:
            clients_to_close.append(client_main)
        await asyncio.gather(*(c.close() for c in clients_to_close))
    return finalize_token_usage_summary(token_usage_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, required=True, help='Name of the evaluation')
    parser.add_argument('--ann_file', type=str, required=True, help='Path to the annotation file (JSONL)')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')

    parser.add_argument('--model', type=str, required=True, help='Model to use')
    parser.add_argument('--api_base_url', type=str, required=True, help='API base URL')
    parser.add_argument('--api_key', type=str, required=True, help='API key')
    parser.add_argument('--backend', type=str, default='api', choices=['api', 'qwen_agent'], help='Inference backend: "api" for direct OpenAI-compatible API, "qwen_agent" for Qwen Agent with tool use')
    parser.add_argument('--img_rescale_ratio', type=positive_float, default=1.0, help='Uniformly rescale each input image by this ratio before applying --img_max_pixels')
    parser.add_argument('--img_format', type=str, default='png', choices=['png', 'jpg'], help='Mime subtype and encoded format to use for image data URLs')
    parser.add_argument('--img_max_pixels', type=int, default=3500 * 3500, help='Resize images to be at most this many pixels before sending to the API; set to 0 to remove the limit')
    parser.add_argument('--sys_prompt', type=str, default='model_default', choices=['model_default', 'think', 'qwen_agent_analysis'], help='System prompt to use')
    parser.add_argument('--chat_completion_kwargs', type=json.loads, default={"max_completion_tokens": 16384}, help='Additional kwargs for chat completion API (default: \'{"max_completion_tokens": 16384}\')')
    parser.add_argument('--client_timeout', type=int, default=None, help='OpenAI client timeout in seconds; if not provided, the default timeout will be used')
    parser.add_argument('--image_url_extra_settings', type=json.loads, default={"detail": "high"}, help='Extra settings for image_url (default: \'{"detail": "high"}\')')

    parser.add_argument('--qwen_agent_model_type', type=str, default='qwenvl_oai', help='Qwen Agent LLM model_type (only used with --backend qwen_agent)')
    parser.add_argument('--qwen_agent_tools', type=json.loads, default=["image_zoom_in_tool"], help='Qwen Agent tool list (default: \'["image_zoom_in_tool"]\')')
    parser.add_argument('--qwen_agent_max_calls', type=int, default=6, help='Max LLM calls per agent run (sets QWEN_AGENT_MAX_LLM_CALL_PER_RUN)')
    parser.add_argument('--qwen_agent_max_short_side_length', type=int, default=None, help='Max short side length for Qwen VL internal processing (sets QWEN_VL_OAI_MAX_SHORT_SIDE_LENGTH)')
    parser.add_argument('--qwen_agent_generate_cfg', type=json.loads, default={"top_p": 0.8, "top_k": 20, "temperature": 0.7, "repetition_penalty": 1.0, "presence_penalty": 1.5}, help='Qwen Agent generation config JSON; max_completion_tokens from --chat_completion_kwargs is merged as fallback')
    parser.add_argument('--qwen_agent_max_retries', type=int, default=3, help='Max retries per sample for Qwen Agent backend')

    parser.add_argument('--judge_model', type=str, default='gpt-5-nano', help='Judge model to use')
    parser.add_argument('--judge_api_base_url', type=str, required=True, help='Judge model API base URL')
    parser.add_argument('--judge_api_key', type=str, required=True, help='Judge model API key')

    parser.add_argument('--output_dir', type=str, default='./outputs/eval', help='Path to the output root directory')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of trials to run')
    parser.add_argument('--concurrency', type=int, default=32, help='Number of concurrent samples to process')
    parser.add_argument('--max_judge_ans_context_len', type=int, default=2000, help='Maximum answer context length (in chars) to send to the judge model; over-long responses will be middle-truncated')
    parser.add_argument('--separate_trial_requests', action='store_true', help='Separate API requests for each trial; useful for APIs that do not accept n > 1')
    args = parser.parse_args()

    if 'n' in args.chat_completion_kwargs:
        raise ValueError("'n' cannot be set in chat_completion_kwargs; it is controlled by --num_trials")

    if args.backend == 'qwen_agent':
        if args.num_trials > 1 and not args.separate_trial_requests:
            print("NOTE: Forcing --separate_trial_requests for qwen_agent backend (agent does not support n > 1)")
            args.separate_trial_requests = True
        os.environ['QWEN_AGENT_MAX_LLM_CALL_PER_RUN'] = str(args.qwen_agent_max_calls)
        if args.qwen_agent_max_short_side_length is not None:
            os.environ['QWEN_VL_OAI_MAX_SHORT_SIDE_LENGTH'] = str(args.qwen_agent_max_short_side_length)

    print(f"Running evaluation: {args.eval_name}")
    print("Evaluation arguments:")
    pprint(vars(args))

    trials_done = set()
    for trial_id in range(args.num_trials):
        trial_done_file = get_eval_trial_dir(trial_id, args) / 'done'
        if trial_done_file.exists():
            trials_done.add(trial_id)

    trials_to_run = sorted(set(range(args.num_trials)) - trials_done)
    for trial_id in trials_to_run:
        trial_dir = get_eval_trial_dir(trial_id, args)
        trial_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = get_eval_dir(args)
    print(f"Evaluation output directory: {eval_dir}")
    print(f"# of planned trials: {args.num_trials}")
    print(f"Trials done: {sorted(trials_done) if trials_done else 'none'}")
    print(f"Trials to run: {trials_to_run if trials_to_run else 'none'}")

    if len(trials_to_run) == 0:
        print(f"No trials to run, exiting")
        exit(0)

    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / 'args.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    main_model_token_usage = asyncio.run(run_trials(args, trials_to_run))

    summary_metrics = summarize_over_trials(args)
    summary_metrics['main_model_token_usage'] = main_model_token_usage
    print_summary_metrics(summary_metrics)

    with open(eval_dir / 'summary_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, ensure_ascii=False, indent=4)

    print(f"All trial(s) completed, summary metrics saved to {eval_dir / 'summary_metrics.json'}")
