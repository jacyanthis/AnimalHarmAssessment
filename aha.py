#!/usr/bin/env python3
"""
Main evaluation script for the AHA benchmark.
"""

import argparse
import json
import logging
import os
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Set, Optional
from pathlib import Path

from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver import generate

from scoring import SOLVER_INSTRUCTIONS, final_digit_model_graded_qa
from utils import (
    setup_logging,
    mkd,
    read_json,
    write_json,
    get_latest_file,
    combine_csv_files,
)

logger = setup_logging("aha", logging.INFO)

@dataclass
class BenchmarkConfig:
    """Holds configuration for the AHA benchmark."""
    model: str
    judges: List[str]
    dataset_path: Path
    output_dir: Path
    batch_size: int
    seed: int = 42
    model_temperature: Optional[float] = 0.5
    judge_temperature: Optional[float] = 0.5
    run_analysis: bool = False
    num_batches: int = 1
    start_batch: int = 0

def parse_args():
    p = argparse.ArgumentParser("AHA Benchmark")
    p.add_argument('--model', required=True)
    p.add_argument('--judges', default="anthropic/claude-3-5-sonnet-20241022,google/gemini-1.5-pro-002,openai/gpt-4")
    p.add_argument('--dataset')
    p.add_argument('--output_dir')
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--num_batches', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model_temperature', default="0.5")
    p.add_argument('--judge_temperature', default="0.5")
    p.add_argument('--run-analysis', action='store_true')
    p.add_argument('--start-batch', type=int, default=0)
    return p.parse_args()

def build_config(args) -> BenchmarkConfig:
    return BenchmarkConfig(
        model=args.model,
        judges=args.judges.split(','),
        dataset_path=Path(args.dataset) if args.dataset else Path("/content/aha/data_public.json"),
        output_dir=Path(args.output_dir) if args.output_dir else Path("/content/aha/results"),
        batch_size=args.batch_size,
        seed=args.seed,
        model_temperature=None if args.model_temperature.lower() == "none" else float(args.model_temperature),
        judge_temperature=None if args.judge_temperature.lower() == "none" else float(args.judge_temperature),
        run_analysis=args.run_analysis,
        num_batches=args.num_batches,
        start_batch=args.start_batch
    )

def record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert a JSON record to an Inspect AI Sample, storing sample_id."""
    meta = record.get("generation_tags", {}).copy()
    meta["sample_id"] = record["sample_id"]  # keep original ID for analysis
    return Sample(
        input=f"{record['question']} Your answer must be EXACTLY 30–40 words long.",
        target=SOLVER_INSTRUCTIONS,
        metadata=meta
    )

@task
def aha_evaluation() -> Task:
    """Definition of the main Inspect AI evaluation task."""
    try:
        from __main__ import config, dataset_path
    except ImportError:
        global config, dataset_path

    dataset = json_dataset(str(dataset_path), sample_fields=record_to_sample)
    solver_instance = generate(max_tokens=1000, temperature=config.model_temperature, cache=True)
    return Task(
        dataset=dataset,
        solver=solver_instance,
        scorer=final_digit_model_graded_qa(config.judges, judge_temperature=config.judge_temperature),
        max_retries=3,
        max_messages=20,
        num_batches=1
    )

def combine_csv_results(config: BenchmarkConfig):
    """Combines results_*.csv files into results_combined.csv, preserving sample_id."""
    csv_files = sorted(config.output_dir.glob('results_*.csv'))
    if not csv_files:
        logger.error("No CSV files found to combine.")
        return None

    combined_df = combine_csv_files(csv_files)
    # Important: do NOT overwrite sample_id with a new sequence

    combined_path = config.output_dir / 'results_combined.csv'
    combined_df.to_csv(combined_path, index=False)
    logger.info(f"Combined CSV saved to: {combined_path}")
    return combined_path

def main():
    global config, dataset_path
    args = parse_args()
    config = build_config(args)
    mkd(config.output_dir)

    # Load entire dataset
    try:
        full_data = read_json(config.dataset_path)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    used_indices: Set[int] = set()
    sampled_files = []
    random.seed(config.seed)

    start_time = time.time()
    for batch_num in range(config.num_batches):
        available = set(range(len(full_data))) - used_indices
        if len(available) < config.batch_size:
            used_indices.clear()
            available = set(range(len(full_data)))
        chosen = random.sample(list(available), config.batch_size)
        used_indices.update(chosen)

        # Write a temp JSON with this batch
        batch_data = [full_data[i] for i in chosen]
        temp_dir = config.output_dir / "temp"
        mkd(temp_dir)
        sampled_file = temp_dir / f"sampled_data_batch{batch_num+1}.json"
        write_json(sampled_file, batch_data)
        sampled_files.append(sampled_file)

        dataset_path = sampled_file
        logger.info(f"Processing batch {batch_num+1}/{config.num_batches}")
        eval("aha_evaluation")

        # Immediately analyze this batch if requested
        if config.run_analysis:
            log_dir = Path("./logs")
            latest_eval = get_latest_file(log_dir, '*.eval')
            if latest_eval:
                logger.info(f"Results for batch {batch_num+1}:")
                os.system(f"python /content/aha/analysis.py --log-file {latest_eval} --output-dir {config.output_dir}")

    total_time = time.time() - start_time
    logger.info(f"Total time for all batches: {total_time:.2f}s")

    # Combine CSV from all batches + do final analysis if requested
    if config.run_analysis:
        logger.info("Combining results from all batches...")
        combined_path = combine_csv_results(config)
        if combined_path and combined_path.exists():
            logger.info("Running final analysis on results_combined.csv ...")
            os.system(f"python /content/aha/analysis.py --csv-file {combined_path} --output-dir {config.output_dir}")
        else:
            logger.warning("No combined CSV found, skipping final CSV-based analysis")

    # Cleanup temp files
    for f in sampled_files:
        if f.exists():
            f.unlink()

if __name__ == "__main__":
    main()
