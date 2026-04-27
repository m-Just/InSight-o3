import argparse
from pathlib import Path
import json
from collections import Counter

import pandas as pd


def resolve_fail_reason(eval_result: dict) -> str:
    inference_result = eval_result.get('inference_result') or {}
    eval_fail_reason = eval_result.get('fail_reason')
    inference_fail_reason = inference_result.get('fail_reason')

    if eval_fail_reason == 'inference_failed' and inference_fail_reason:
        return inference_fail_reason
    return eval_fail_reason or inference_fail_reason or 'unknown'


def resolve_fail_detail(eval_result: dict) -> str:
    inference_result = eval_result.get('inference_result') or {}
    return inference_result.get('fail_detail') or eval_result.get('fail_detail') or ''


def compact_fail_detail(detail: str, max_chars: int = 180) -> str:
    detail = ' '.join(detail.split())
    if len(detail) > max_chars:
        return detail[: max_chars - 3] + '...'
    return detail


def summarize_fail_reasons(eval_dir: Path) -> tuple[Counter, int, dict[str, Counter]]:
    fail_reasons = Counter()
    fail_details_by_reason = {}
    total_non_success = 0

    for trial_dir in sorted(eval_dir.glob('trial_*')):
        eval_records_path = trial_dir / 'eval_records.jsonl'
        if not eval_records_path.exists():
            continue

        with open(eval_records_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                eval_result = record.get('eval_result', {})
                if eval_result.get('success'):
                    continue

                fail_reason = resolve_fail_reason(eval_result)
                fail_reasons[fail_reason] += 1
                fail_detail = resolve_fail_detail(eval_result)
                if fail_detail:
                    if fail_reason not in fail_details_by_reason:
                        fail_details_by_reason[fail_reason] = Counter()
                    fail_details_by_reason[fail_reason][compact_fail_detail(fail_detail)] += 1
                total_non_success += 1

    return fail_reasons, total_non_success, fail_details_by_reason


def format_fail_reason_summary(fail_reasons: Counter, total_non_success: int) -> str:
    if total_non_success == 0:
        return 'none'

    return ', '.join(
        f'{reason}={count} ({count / total_non_success * 100:.1f}%)'
        for reason, count in fail_reasons.most_common()
    )


def format_fail_detail_summary(
    fail_details_by_reason: dict[str, Counter],
    max_reasons: int = 3,
    max_details_per_reason: int = 3,
) -> str:
    if not fail_details_by_reason:
        return 'none'

    parts = []
    ranked_reasons = sorted(
        fail_details_by_reason.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    )
    for reason, detail_counts in ranked_reasons[:max_reasons]:
        details = ', '.join(
            f'{detail} [{count}]'
            for detail, count in detail_counts.most_common(max_details_per_reason)
        )
        parts.append(f'{reason}: {details}')
    return '; '.join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Gather evaluation results for given settings and datasets from <output_dir>/<setting>/<dataset>/summary_metrics.json.')
    parser.add_argument('--settings', nargs='+', required=True, help='Evaluation settings for which to gather results.')
    parser.add_argument('--datasets', nargs='+', required=True, help='Datasets for which to gather results.')
    parser.add_argument('--output_dir', type=str, default='./outputs/eval', help='Path to the eval outputs directory.')
    args = parser.parse_args()

    rows = []
    for setting in args.settings:
        row = {}
        for dataset in args.datasets:
            row[dataset] = 'N/A'
            eval_dir = Path(args.output_dir) / setting / dataset
            summary_metrics_path = eval_dir / 'summary_metrics.json'
            if not summary_metrics_path.exists():
                print(f"WARNING: Summary metrics not found for {setting}/{dataset}")
                continue
            with open(summary_metrics_path, 'r') as f:
                summary_metrics = json.load(f)
            fail_reasons, total_non_success, fail_details_by_reason = summarize_fail_reasons(eval_dir)
            printed_fail_reason_summary = False
            printed_fail_detail_summary = False
            if summary_metrics['overall_success_rate']['mean'] < 1.0:
                mean = str(round(summary_metrics['overall_success_rate']['mean'] * 100, 1))
                std = str(round(summary_metrics['overall_success_rate']['std'] * 100, 1))
                print(f"WARNING: Overall success rate ({mean}±{std}) is less than 100.0 for {setting}/{dataset}")
                print(f"WARNING: Fail reasons for {setting}/{dataset}: {format_fail_reason_summary(fail_reasons, total_non_success)}")
                print(f"WARNING: Fail details for {setting}/{dataset}: {format_fail_detail_summary(fail_details_by_reason)}")
                printed_fail_reason_summary = True
                printed_fail_detail_summary = True
            if summary_metrics['overall_completion_rate']['mean'] < 1.0:
                mean = str(round(summary_metrics['overall_completion_rate']['mean'] * 100, 1))
                std = str(round(summary_metrics['overall_completion_rate']['std'] * 100, 1))
                print(f"WARNING: Overall completion rate ({mean}±{std}) is less than 100.0 for {setting}/{dataset}")
                if not printed_fail_reason_summary:
                    print(f"WARNING: Fail reasons for {setting}/{dataset}: {format_fail_reason_summary(fail_reasons, total_non_success)}")
                if not printed_fail_detail_summary:
                    print(f"WARNING: Fail details for {setting}/{dataset}: {format_fail_detail_summary(fail_details_by_reason)}")
            mean = str(round(summary_metrics['overall_accuracy']['mean'] * 100, 1))
            std = str(round(summary_metrics['overall_accuracy']['std'] * 100, 1))
            row[dataset] = f"{mean}±{std} @{summary_metrics.get('num_trials', 1)}"
            if len(args.datasets) == 1:
                for category in summary_metrics['category_accuracy']:
                    mean = str(round(summary_metrics['category_accuracy'][category]['mean'] * 100, 1))
                    std = str(round(summary_metrics['category_accuracy'][category]['std'] * 100, 1))
                    row[f"{dataset}-{category}"] = f"{mean}±{std} @{summary_metrics.get('num_trials', 1)}"
        rows.append(row)
    
    df = pd.DataFrame(rows, index=args.settings)
    print(df)
