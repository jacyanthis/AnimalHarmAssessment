#!/usr/bin/env python3
"""
Evaluation script for the AHA benchmark.
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

import pandas as pd
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
    generate_timestamp,
)

# Set up logging and adjust specific logger levels to hide TRACE messages.
setup_logging(logging.INFO)
logging.getLogger("inspect_ai.model").setLevel(logging.INFO)


class Config:
    def __init__(self, args):
        self.model = args.model
        self.scorer_models = args.scorer_models.split(',')
        self.dataset_path = Path(args.dataset) if args.dataset else Path("/content/aha/data_public.json")
        # Output directory (e.g. "aha/results")
        self.output_dir = Path(args.output_dir) if args.output_dir else Path("/content/aha/results")
        self.batch_size = args.batch_size
        self.seed = args.seed
        # If the temperature string equals "None" (case-insensitive), set to None; otherwise convert to float.
        if args.temperature.lower() == "none":
            self.temperature = None
        else:
            self.temperature = float(args.temperature)
        self.run_analysis = args.run_analysis
        self.num_batches = args.num_batches
        self.current_batch = 1


def setup_environment(config: Config) -> None:
    """Set up directories, seed, and environment variables."""
    # Ensure the output directory exists.
    mkd(config.output_dir)
    random.seed(config.seed)
    os.environ["INSPECT_EVAL_MODEL"] = config.model
    logging.info(
        f"AHA benchmark ({config.batch_size * config.num_batches} total samples, "
        f"{config.num_batches} batches of {config.batch_size}): {config.model}"
    )
    logging.info(f"max_tokens: 1000, temperature: {config.temperature}, seed: {config.seed}")


def load_and_sample_data(
    config: Config, full_data: List[Dict[str, Any]], used_indices: Set[int]
) -> Tuple[Path, List[Dict[str, Any]]]:
    """Sample a batch of data from the full dataset, avoiding duplicates."""
    available = set(range(len(full_data))) - used_indices
    if len(available) < config.batch_size:
        used_indices.clear()
        available = set(range(len(full_data)))
    batch = set(random.sample(list(available), config.batch_size))
    used_indices.update(batch)
    sampled = [full_data[i] for i in batch]
    # Create a temp subdirectory inside the output directory for temporary sampled files.
    temp_dir = config.output_dir / "temp"
    mkd(temp_dir)
    sampled_file = temp_dir / f"sampled_data_batch{config.current_batch}.json"
    write_json(sampled_file, sampled)
    return sampled_file, sampled


def record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert a record from the dataset to the Sample format required by inspect_ai."""
    return Sample(
        input=f"{record['question']} Your answer must be EXACTLY 30–40 words long.",
        target=SOLVER_INSTRUCTIONS,
        metadata=record.get("generation_tags", {})
    )


# Globals required by the inspect_ai framework.
global config, dataset_path

@task
def aha_evaluation() -> Task:
    """The main evaluation task for the AHA benchmark."""
    global config, dataset_path
    dataset = json_dataset(str(dataset_path), sample_fields=record_to_sample)
    if config.temperature is not None:
        solver_instance = generate(max_tokens=1000, temperature=config.temperature, cache=True)
    else:
        solver_instance = generate(max_tokens=1000, cache=True)
    return Task(
        dataset=dataset,
        solver=solver_instance,
        scorer=final_digit_model_graded_qa(config.scorer_models),
        max_retries=3,
        max_messages=20,
        num_batches=1  # Batch handling is performed at a higher level.
    )


def combine_csv_results(config: Config, full_data: List[Dict[str, Any]], start_batch: int) -> None:
    """Combine CSV files from multiple batches, adding tag columns."""
    csv_files = sorted(config.output_dir.glob('results_*.csv'))
    if not csv_files:
        logging.error("No CSV files found to combine")
        return
    combined_df = combine_csv_files(csv_files)
    start_idx = start_batch * config.batch_size
    tags_list = [
        data.get('tags', [])
        for data in full_data[start_idx:start_idx + (config.num_batches * config.batch_size)]
    ]
    max_tags = max((len(tags) for tags in tags_list), default=0)
    combined_df['sample_id'] = range(1, len(combined_df) + 1)
    for i in range(max_tags):
        col = f'tag{i+1}'
        combined_df[col] = ''
        for idx, tags in enumerate(tags_list):
            if idx < len(combined_df) and i < len(tags):
                combined_df.at[idx, col] = tags[i]
    combined_path = config.output_dir / 'results_combined.csv'
    combined_df.to_csv(combined_path, index=False)
    logging.info(f"Combined CSV saved to: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the AHA benchmark evaluation")
    parser.add_argument('--model', required=True, help='Model to evaluate')
    parser.add_argument(
        '--scorer_models',
        default="anthropic/claude-3-5-sonnet-20241022,google/gemini-1.5-pro-002,openai/gpt-4o-2024-08-06",
        help='Comma-separated list of scoring models'
    )
    parser.add_argument('--dataset', help='Path to dataset JSON')
    parser.add_argument('--output_dir', help='Directory for output files')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--num_batches', type=int, default=1, help='Number of batches to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--temperature',
        type=str,
        default="0.5",
        help='Temperature for generation (or "None" to disable)'
    )
    parser.add_argument('--run-analysis', action='store_true', help='Run analysis.py after evaluation')
    parser.add_argument('--start-batch', type=int, default=0, help='Starting batch number for combining results')
    args = parser.parse_args()

    global config, dataset_path
    config = Config(args)
    setup_environment(config)
    try:
        full_data = read_json(config.dataset_path)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    start_time = time.time()
    sampled_files: List[Path] = []
    used_indices: Set[int] = set()
    try:
        for batch in range(config.num_batches):
            config.current_batch = batch + 1
            sampled_file, _ = load_and_sample_data(config, full_data, used_indices)
            sampled_files.append(sampled_file)
            dataset_path = sampled_file  # Used by the inspect_ai task.
            logging.info(f"Processing batch {config.current_batch}/{config.num_batches}")
            eval("aha_evaluation")
            if config.run_analysis:
                log_dir = Path("./logs")
                latest = get_latest_file(log_dir, '*.eval')
                if latest:
                    logging.info(f"Results for batch {config.current_batch}:")
                    os.system(f"python /content/aha/analysis.py --log-file {latest} --output-dir {config.output_dir}")
        total_time = time.time() - start_time
        logging.info(f"Total time for all batches: {total_time:.2f}s")
        if config.run_analysis:
            logging.info("Combining results from all batches...")
            combine_csv_results(config, full_data, args.start_batch)
            combined_path = config.output_dir / 'results_combined.csv'
            if combined_path.exists():
                logging.info("Running final analysis on results_combined.csv ...")
                os.system(f"python /content/aha/analysis.py --csv-file {combined_path} --output-dir {config.output_dir}")
            else:
                logging.warning("results_combined.csv not found, skipping final CSV-based analysis")
    finally:
        # Cleanup: remove temporary sampled files.
        for f in sampled_files:
            if f.exists():
                f.unlink()


if __name__ == "__main__":
    main()
