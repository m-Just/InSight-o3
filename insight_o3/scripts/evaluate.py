import argparse
import asyncio
from dataclasses import dataclass, asdict
from pprint import pprint
from pathlib import Path
from collections import defaultdict
import re
import json

from openai import AsyncOpenAI
import pandas as pd
import numpy as np
from tqdm import tqdm

import insight_o3.prompts as prompts
from insight_o3.inference import query_api_vqa, InferenceResult
from insight_o3.utils.api import query_api


@dataclass
class EvalResult:
    sample_index: int
    success: bool
    fail_reason: str | None = None
    inference_result: InferenceResult | None = None
    extracted_answer: str | None = None
    is_correct: bool = False


def get_eval_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir).resolve() / args.eval_name


def get_eval_trial_dir(trial_id: int, args: argparse.Namespace) -> Path:
    return get_eval_dir(args) / f"trial_{trial_id}"


async def process_sample(
    sample: dict,
    args: argparse.Namespace,
    trial_ids: list[int],
    client_main: AsyncOpenAI,
    client_judge: AsyncOpenAI,
) -> list[EvalResult]:
    index = sample['sample_index']
    image_path = sample['image'] if 'image' in sample else sample['file_name']
    image_path = str(Path(args.img_dir) / image_path)
    question = sample['question']
    options = sample.get('options', None)
    user_prompt = f"{question}\n{options}" if options else question
    ground_truth = sample['answer']

    if args.sys_prompt == 'model_default':
        system_prompt = None
    elif args.sys_prompt == 'think':
        system_prompt = prompts.SIMPLE_SYSTEM_PROMPT_THINKING
    else:
        raise ValueError(f"Invalid system prompt: {args.sys_prompt}")

    chat_completion_kwargs = {
        'n': len(trial_ids),
        **args.chat_completion_kwargs,
    }

    inference_results = await query_api_vqa(
        image_path=image_path,
        user_prompt=user_prompt,
        model=args.model,
        client=client_main,
        image_max_pixels=args.img_max_pixels or None,
        image_url_extra_settings=args.image_url_extra_settings,
        system_prompt=system_prompt,
        **chat_completion_kwargs,
    )

    # Extract answers from the inference results and check if they are correct
    eval_results = []
    for result in inference_results:
        if not result.success:
            eval_results.append(EvalResult(
                sample_index=index,
                success=False,
                fail_reason='inference_failed',
                inference_result=result,
            ))
            continue

        if result.finish_reason != 'stop':
            eval_results.append(EvalResult(
                sample_index=index,
                success=False,
                fail_reason=f'finish_reason:{result.finish_reason}',
                inference_result=result,
            ))
            continue

        # Try rule-based answer extraction first
        matches = re.findall(r'<answer>(.*?)</answer>', result.last_message_content, re.DOTALL)
        raw_answer_str = matches[-1].strip() if matches else result.last_message_content
        matches = re.findall(r'\\boxed\{(.*?)\}', raw_answer_str, re.DOTALL)
        raw_answer_str = matches[-1].strip() if matches else raw_answer_str

        # Correct if the extracted answer matches the ground truth verbatim
        if raw_answer_str == ground_truth:
            eval_results.append(EvalResult(
                sample_index=index,
                success=True,
                inference_result=result,
                extracted_answer=raw_answer_str,
                is_correct=True,
            ))
            continue
        
        # Extract answer using judge model
        if len(raw_answer_str) > args.max_answer_length:
            raw_answer_str = f'[cropped] ... {raw_answer_str[-args.max_answer_length:]}'
            
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
            _, response = await query_api(
                query=answer_extraction_prompt,
                model=args.judge_model,
                client=client_judge,
                max_completion_tokens=2048,
            )
            extracted_answer = response.choices[0].message.content
        except Exception as e:
            print(f"WARNING: Failed to extract answer: {e}")
            eval_results.append(EvalResult(
                sample_index=index,
                success=False,
                fail_reason='answer_extraction_failed',
                inference_result=result,
            ))
            continue

        # If the question is an MCQ, check if the extracted answer matches the ground truth verbatim
        if options:
            eval_results.append(EvalResult(
                sample_index=index,
                success=True,
                inference_result=result,
                extracted_answer=extracted_answer.strip(),
                is_correct=(extracted_answer.strip() == ground_truth),
            ))
            continue

        # Otherwise (the question is not an MCQ), check with the judge model
        judge_prompt = prompts.GPT_JUDGE_ANSWER_PROMPT.format(
            question=question,
            gt_answer=ground_truth,
            model_answer=extracted_answer,
        )

        try:
            _, judge_response = await query_api(
                query=judge_prompt,
                model=args.judge_model,
                client=client_judge,
                max_completion_tokens=2048,
            )
            judge_content = judge_response.choices[0].message.content
        except Exception as e:
            print(f"WARNING: Failed to judge answer: {e}")
            eval_results.append(EvalResult(
                sample_index=index,
                success=False,
                fail_reason='judge_failed',
                inference_result=result,
            ))
            continue

        eval_results.append(EvalResult(
            sample_index=index,
            success=True,
            inference_result=result,
            extracted_answer=extracted_answer,
            is_correct=("correct" in judge_content.lower()),
        ))

    return eval_results


async def main(args: argparse.Namespace, trial_ids: list[int], client_main: AsyncOpenAI, client_judge: AsyncOpenAI):
    # Load samples
    if not Path(args.ann_file).exists():
        raise FileNotFoundError(f"Annotation file not found: {args.ann_file}")
    samples = pd.read_json(args.ann_file, lines=True).to_dict(orient='records')
    for idx, sample in enumerate(samples):
        sample['sample_index'] = idx

    # Evaluate samples concurrently
    trials = defaultdict(list)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def handle_sample(sample: dict) -> tuple[int, list[EvalResult]]:
        async with semaphore:
            results = await process_sample(sample, args, trial_ids, client_main, client_judge)
        return sample['sample_index'], results

    tasks = [asyncio.create_task(handle_sample(sample)) for sample in samples]

    for i, task in enumerate(tqdm(asyncio.as_completed(tasks), total=len(samples), desc="Evaluating")):
        sample_index, eval_results = await task
        sample = samples[sample_index]

        for trial_id, eval_result in zip(trial_ids, eval_results, strict=True):
            trials[trial_id].append({**sample, 'eval_result': asdict(eval_result)})

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
        }

        print(f"Evaluation results (trial {trial_id}):")
        pprint(accuracy_results, sort_dicts=False)

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

    def compute_stats(values: list[float]) -> dict:
        return {'mean': float(np.mean(values)), 'std': float(np.std(values))}

    summary: dict = {'num_trials': args.num_trials}
    summary['overall_success_rate'] = compute_stats(overall_success_rates)
    summary['overall_completion_rate'] = compute_stats(overall_completion_rates)
    summary['overall_accuracy'] = compute_stats(overall_accuracies)
    summary['category_success_rate'] = {c: compute_stats(values) for c, values in category_success_rates.items()}
    summary['category_completion_rate'] = {c: compute_stats(values) for c, values in category_completion_rates.items()}
    summary['category_accuracy'] = {c: compute_stats(values) for c, values in category_accuracies.items()}

    return summary


async def run_trials(args: argparse.Namespace, trials_to_run: list[int]):
    client_main = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base_url,
        timeout=args.client_timeout,
    )
    client_judge = AsyncOpenAI(
        api_key=args.judge_api_key,
        base_url=args.judge_api_base_url,
        timeout=args.client_timeout,
    )

    try:
        if args.separate_trial_requests:
            for trial_id in trials_to_run:
                print(f"Running trial {trial_id}")
                await main(args, [trial_id], client_main, client_judge)
        else:
            print(f"Running trials {trials_to_run} together")
            await main(args, trials_to_run, client_main, client_judge)
    finally:
        await asyncio.gather(client_main.close(), client_judge.close())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, required=True, help='Name of the evaluation')
    parser.add_argument('--ann_file', type=str, required=True, help='Path to the annotation file (JSONL)')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')

    parser.add_argument('--model', type=str, required=True, help='Model to use')
    parser.add_argument('--api_base_url', type=str, required=True, help='API base URL')
    parser.add_argument('--api_key', type=str, required=True, help='API key')
    parser.add_argument('--img_max_pixels', type=int, default=3500 * 3500, help='Resize images to be at most this many pixels before sending to the API; set to 0 to remove the limit')
    parser.add_argument('--sys_prompt', type=str, default='model_default', choices=['model_default', 'think'], help='System prompt to use')
    parser.add_argument('--chat_completion_kwargs', type=json.loads, default={"max_completion_tokens": 16384}, help='Additional kwargs for chat completion API (e.g. \'{"max_completion_tokens": 16384, "temperature": 0.7}\')')
    parser.add_argument('--client_timeout', type=int, default=600, help='OpenAI client timeout in seconds')
    parser.add_argument('--image_url_extra_settings', type=json.loads, default={"detail": "high"}, help='Extra settings for image_url (e.g. \'{"detail": "high"}\')')

    parser.add_argument('--judge_model', type=str, default='gpt-5-nano', help='Judge model to use')
    parser.add_argument('--judge_api_base_url', type=str, required=True, help='Judge model API base URL')
    parser.add_argument('--judge_api_key', type=str, required=True, help='Judge model API key')

    parser.add_argument('--output_dir', type=str, default='./outputs/eval', help='Path to the output root directory')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of trials to run')
    parser.add_argument('--concurrency', type=int, default=128, help='Number of concurrent samples to process')
    parser.add_argument('--max_answer_length', type=int, default=1000, help='Maximum response span (in chars) to consider for answer extraction using the judge model; over-long responses will be left-cropped')
    parser.add_argument('--separate_trial_requests', action='store_true', help='Separate API requests for each trial; useful for APIs that do not accept n > 1')
    args = parser.parse_args()

    if 'n' in args.chat_completion_kwargs:
        raise ValueError("'n' cannot be set in chat_completion_kwargs; it is controlled by --num_trials")

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

    asyncio.run(run_trials(args, trials_to_run))

    summary_metrics = summarize_over_trials(args)
    print(f"Summary metrics over {args.num_trials} trial(s):")
    pprint(summary_metrics, sort_dicts=False)

    with open(eval_dir / 'summary_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, ensure_ascii=False, indent=4)

    print(f"All trial(s) completed, summary metrics saved to {eval_dir / 'summary_metrics.json'}")