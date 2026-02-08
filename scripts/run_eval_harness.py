#!/usr/bin/env python3
"""Evaluation Harness — run LLM judges against labeled dataset and report metrics.

Runs each per-criterion evaluator independently against labeled samples,
computes precision, recall, F1, and Cohen's kappa per criterion, and
outputs a single-row summary suitable for tracking over time.

Usage:
  # Full evaluation (requires live LLM)
  python scripts/run_eval_harness.py

  # Assessment criteria only
  python scripts/run_eval_harness.py --mode assess

  # Chat criteria only
  python scripts/run_eval_harness.py --mode chat

  # Output CSV for experiment tracking
  python scripts/run_eval_harness.py --csv results.csv

  # Use a specific split (dev or test)
  python scripts/run_eval_harness.py --split test

Based on the scaling methodology:
  - One evaluator per criterion (never a "God Evaluator")
  - Binary PASS/FAIL labels
  - 75/25 dev/test split
  - Metrics: precision, recall, F1, Cohen's kappa per criterion
  - Single-row output for experiment tracking
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, ".")

from app.llm_judge import (
    _evaluate_single_criterion,
    _format_context,
    _format_patient,
    _FAITHFULNESS_PROMPT,
    _CORRECTNESS_PROMPT,
    _CITATION_ACCURACY_PROMPT,
    _COMPLETENESS_PROMPT,
    _SAFETY_PROMPT,
    _RELEVANCE_PROMPT,
    compute_classification_metrics,
)
from tests.evaluation_dataset import ASSESS_SAMPLES, CHAT_SAMPLES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def split_dataset(samples: list, split: str, ratio: float = 0.75) -> list:
    """Split dataset into dev (first 75%) or test (last 25%)."""
    cutoff = int(len(samples) * ratio)
    if split == "dev":
        return samples[:cutoff]
    elif split == "test":
        return samples[cutoff:]
    return samples  # "all"


def evaluate_assess_criterion(sample: dict, criterion: str) -> dict:
    """Run a single assessment criterion evaluator on one sample."""
    context = _format_context(sample["context_excerpts"])
    patient_data = _format_patient(sample["patient"])
    output = sample["output"]
    symptoms = ", ".join(sample["patient"].get("symptoms", []))
    citations_text = json.dumps(output.get("citations", []), indent=2)[:2000]
    recommendations_text = json.dumps(output.get("recommendations", []), indent=2)[:1000]
    reasoning = output.get("reasoning", "")

    prompt_map = {
        "faithfulness": (
            _FAITHFULNESS_PROMPT,
            {"context": context, "response_text": reasoning},
        ),
        "correctness": (
            _CORRECTNESS_PROMPT,
            {"context": context, "patient_data": patient_data, "risk_level": output.get("risk_level", "")},
        ),
        "citation_accuracy": (
            _CITATION_ACCURACY_PROMPT,
            {"context": context, "response_text": reasoning, "citations": citations_text},
        ),
        "completeness": (
            _COMPLETENESS_PROMPT,
            {"symptoms": symptoms, "reasoning": reasoning},
        ),
        "safety": (
            _SAFETY_PROMPT,
            {"patient_data": patient_data, "risk_level": output.get("risk_level", ""), "recommendations": recommendations_text, "context": context},
        ),
    }

    prompt_template, kwargs = prompt_map[criterion]
    return _evaluate_single_criterion(prompt_template, kwargs, criterion)


def evaluate_chat_criterion(sample: dict, criterion: str) -> dict:
    """Run a single chat criterion evaluator on one sample."""
    context = _format_context(sample["context_excerpts"])
    citations_text = json.dumps(
        [{"page": c.get("page", "?"), "excerpt": c.get("text", "")[:200]} for c in sample["context_excerpts"]],
        indent=2,
    )[:2000]

    prompt_map = {
        "faithfulness": (
            _FAITHFULNESS_PROMPT,
            {"context": context, "response_text": sample["answer"]},
        ),
        "relevance": (
            _RELEVANCE_PROMPT,
            {"question": sample["question"], "answer": sample["answer"]},
        ),
        "citation_accuracy": (
            _CITATION_ACCURACY_PROMPT,
            {"context": context, "response_text": sample["answer"], "citations": citations_text},
        ),
    }

    prompt_template, kwargs = prompt_map[criterion]
    return _evaluate_single_criterion(prompt_template, kwargs, criterion)


def run_assess_eval(samples: list) -> dict:
    """Run all 5 assessment criteria across samples and return metrics."""
    criteria = ["faithfulness", "correctness", "citation_accuracy", "completeness", "safety"]
    results = {}

    for criterion in criteria:
        logger.info("Evaluating assessment criterion: %s (%d samples)", criterion, len(samples))
        verdicts = []
        gold_labels = []
        total_tokens_in = 0
        total_tokens_out = 0
        total_latency_ms = 0.0

        for sample in samples:
            result = evaluate_assess_criterion(sample, criterion)
            verdict_dict = {"criteria": {criterion: result}}
            verdicts.append(verdict_dict)
            gold_labels.append(sample["labels"][criterion])

            total_tokens_in += result.get("tokens_in", 0)
            total_tokens_out += result.get("tokens_out", 0)
            total_latency_ms += result.get("latency_ms", 0.0)

            predicted = result["verdict"]
            expected = sample["labels"][criterion]
            match = "OK" if predicted == expected else "MISMATCH"
            logger.info(
                "  %s [%s]: predicted=%s, expected=%s %s (tokens: %d/%d, %.0fms)",
                sample["id"], criterion, predicted, expected, match,
                result.get("tokens_in", 0), result.get("tokens_out", 0),
                result.get("latency_ms", 0.0),
            )

        metrics = compute_classification_metrics(verdicts, criterion, gold_labels)
        metrics["total_tokens_in"] = total_tokens_in
        metrics["total_tokens_out"] = total_tokens_out
        metrics["avg_latency_ms"] = round(total_latency_ms / max(len(samples), 1), 1)
        results[criterion] = metrics

    return results


def run_chat_eval(samples: list) -> dict:
    """Run all 3 chat criteria across samples and return metrics."""
    criteria = ["faithfulness", "relevance", "citation_accuracy"]
    results = {}

    for criterion in criteria:
        logger.info("Evaluating chat criterion: %s (%d samples)", criterion, len(samples))
        verdicts = []
        gold_labels = []
        total_tokens_in = 0
        total_tokens_out = 0
        total_latency_ms = 0.0

        for sample in samples:
            result = evaluate_chat_criterion(sample, criterion)
            verdict_dict = {"criteria": {criterion: result}}
            verdicts.append(verdict_dict)
            gold_labels.append(sample["labels"][criterion])

            total_tokens_in += result.get("tokens_in", 0)
            total_tokens_out += result.get("tokens_out", 0)
            total_latency_ms += result.get("latency_ms", 0.0)

            predicted = result["verdict"]
            expected = sample["labels"][criterion]
            match = "OK" if predicted == expected else "MISMATCH"
            logger.info(
                "  %s [%s]: predicted=%s, expected=%s %s (tokens: %d/%d, %.0fms)",
                sample["id"], criterion, predicted, expected, match,
                result.get("tokens_in", 0), result.get("tokens_out", 0),
                result.get("latency_ms", 0.0),
            )

        metrics = compute_classification_metrics(verdicts, criterion, gold_labels)
        metrics["total_tokens_in"] = total_tokens_in
        metrics["total_tokens_out"] = total_tokens_out
        metrics["avg_latency_ms"] = round(total_latency_ms / max(len(samples), 1), 1)
        results[criterion] = metrics

    return results


def print_summary(assess_results: dict, chat_results: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    all_results = {}
    grand_tokens_in = 0
    grand_tokens_out = 0

    if assess_results:
        print("\nAssessment Criteria:")
        print(f"  {'Criterion':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Kappa':>10} {'Tok In':>8} {'Tok Out':>8} {'Avg ms':>8}")
        print("  " + "-" * 88)
        for criterion, metrics in assess_results.items():
            print(
                f"  {criterion:<20} {metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} "
                f"{metrics['cohens_kappa']:>10.4f}"
                f" {metrics.get('total_tokens_in', 0):>8}"
                f" {metrics.get('total_tokens_out', 0):>8}"
                f" {metrics.get('avg_latency_ms', 0):>8.0f}"
            )
            grand_tokens_in += metrics.get("total_tokens_in", 0)
            grand_tokens_out += metrics.get("total_tokens_out", 0)
            all_results[f"assess_{criterion}"] = metrics

    if chat_results:
        print("\nChat Criteria:")
        print(f"  {'Criterion':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Kappa':>10} {'Tok In':>8} {'Tok Out':>8} {'Avg ms':>8}")
        print("  " + "-" * 88)
        for criterion, metrics in chat_results.items():
            print(
                f"  {criterion:<20} {metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} "
                f"{metrics['cohens_kappa']:>10.4f}"
                f" {metrics.get('total_tokens_in', 0):>8}"
                f" {metrics.get('total_tokens_out', 0):>8}"
                f" {metrics.get('avg_latency_ms', 0):>8.0f}"
            )
            grand_tokens_in += metrics.get("total_tokens_in", 0)
            grand_tokens_out += metrics.get("total_tokens_out", 0)
            all_results[f"chat_{criterion}"] = metrics

    print(f"\n  Total tokens: {grand_tokens_in:,} in / {grand_tokens_out:,} out")
    print("=" * 80)
    return all_results


def write_csv(all_results: dict, csv_path: str, model: str):
    """Append a single-row summary to a CSV file for experiment tracking."""
    # Build the row
    row = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
    }
    for key, metrics in all_results.items():
        row[f"{key}_precision"] = metrics["precision"]
        row[f"{key}_recall"] = metrics["recall"]
        row[f"{key}_f1"] = metrics["f1"]
        row[f"{key}_kappa"] = metrics["cohens_kappa"]
        row[f"{key}_tokens_in"] = metrics.get("total_tokens_in", 0)
        row[f"{key}_tokens_out"] = metrics.get("total_tokens_out", 0)
        row[f"{key}_avg_latency_ms"] = metrics.get("avg_latency_ms", 0)

    # Write (append mode, create header if file is new)
    file_exists = False
    try:
        with open(csv_path, "r") as f:
            file_exists = bool(f.readline())
    except FileNotFoundError:
        pass

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info("Results appended to %s", csv_path)


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge evaluation harness")
    parser.add_argument(
        "--mode", choices=["all", "assess", "chat"], default="all",
        help="Which evaluators to run (default: all)",
    )
    parser.add_argument(
        "--split", choices=["all", "dev", "test"], default="all",
        help="Dataset split to evaluate (default: all)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV file for experiment tracking (appends one row)",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-pro",
        help="Model name for CSV tracking (default: gemini-2.5-pro)",
    )
    args = parser.parse_args()

    start_time = time.time()

    assess_results = {}
    chat_results = {}

    if args.mode in ("all", "assess"):
        samples = split_dataset(ASSESS_SAMPLES, args.split)
        logger.info("Running assessment eval: %d samples (split=%s)", len(samples), args.split)
        assess_results = run_assess_eval(samples)

    if args.mode in ("all", "chat"):
        samples = split_dataset(CHAT_SAMPLES, args.split)
        logger.info("Running chat eval: %d samples (split=%s)", len(samples), args.split)
        chat_results = run_chat_eval(samples)

    all_results = print_summary(assess_results, chat_results)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    if args.csv:
        write_csv(all_results, args.csv, args.model)


if __name__ == "__main__":
    main()
