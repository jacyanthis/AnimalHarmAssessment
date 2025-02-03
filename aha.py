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

###########################################################
# Global Logging & Configuration
###########################################################

# Declare global references so Inspect AI can read them in aha_evaluation().
config = None
dataset_path = None

# Basic logging setup for this script
setup_logging(logging.INFO)
logging.getLogger("inspect_ai.model").setLevel(logging.INFO)


class Config:
    """
    Holds all command-line parameter values and derived settings
    needed for the AHA benchmark run.
    """
    def __init__(self, args):
        self.model = args.model
        self.judges = args.judges.split(',')
        self.dataset_path = Path(args.dataset) if args.dataset else Path("/content/aha/data_public.json")
        self.output_dir = Path(args.output_dir) if args.output_dir else Path("/content/aha/results")
        self.batch_size = args.batch_size
        self.seed = args.seed
        # Convert string to float or None
        self.model_temperature = None if args.model_temperature.lower() == "none" else float(args.model_temperature)
        self.judge_temperature = None if args.judge_temperature.lower() == "none" else float(args.judge_temperature)
        self.run_analysis = args.run_analysis
        self.num_batches = args.num_batches
        self.current_batch = 1


###########################################################
# Environment & Data Setup
###########################################################

def setup_environment(conf: Config) -> None:
    """
    Sets up directories, random seed, and environment variable for Inspect AI.
    """
    mkd(conf.output_dir)
    random.seed(conf.seed)
    os.environ["INSPECT_EVAL_MODEL"] = conf.model
    logging.info(
        f"AHA benchmark with {conf.num_batches} batch(es) of size {conf.batch_size}, model={conf.model}."
    )
    logging.info(f"max_tokens=1000, model_temperature={conf.model_temperature}, "
                 f"judge_temperature={conf.judge_temperature}, seed={conf.seed}")


def load_and_sample_data(conf: Config, full_data: List[Dict[str, Any]], used_indices: Set[int]) -> Tuple[Path, List[Dict[str, Any]]]:
    """
    Chooses a random batch of size `conf.batch_size` from `full_data`, writes
    it to a temporary JSON file, and returns (file_path, batch_records).
    """
    available = set(range(len(full_data))) - used_indices
    if len(available) < conf.batch_size:
        used_indices.clear()
        available = set(range(len(full_data)))

    chosen_indices = random.sample(list(available), conf.batch_size)
    used_indices.update(chosen_indices)
    sampled = [full_data[i] for i in chosen_indices]

    # Write to a "temp" subdirectory inside the output directory
    temp_dir = conf.output_dir / "temp"
    mkd(temp_dir)
    sampled_file = temp_dir / f"sampled_data_batch{conf.current_batch}.json"
    write_json(sampled_file, sampled)
    return sampled_file, sampled


def record_to_sample(record: Dict[str, Any]) -> Sample:
    """
    Convert a single record (from data_public.json) into an Inspect AI Sample.
    Attaches sample_id to metadata so that analysis can retrieve it.
    """
    meta = record.get("generation_tags", {}).copy()
    meta["sample_id"] = record["sample_id"]
    meta["tags"] = record.get("tags", [])
    return Sample(
        input=f"{record['question']} Your answer must be EXACTLY 30–40 words long.",
        target=SOLVER_INSTRUCTIONS,
        metadata=meta
    )


###########################################################
# Inspect AI Task Definition
###########################################################

@task
def aha_evaluation() -> Task:
    """
    Defines how Inspect AI loads the batch dataset, solves with a model,
    then applies the final_digit_model_graded_qa scoring method.
    """
    try:
        from __main__ import config, dataset_path
    except ImportError:
        global config, dataset_path

    dataset = json_dataset(str(dataset_path), sample_fields=record_to_sample)

    if config.model_temperature is not None:
        solver_instance = generate(max_tokens=1000, temperature=config.model_temperature, cache=True)
    else:
        solver_instance = generate(max_tokens=1000, cache=True)

    return Task(
        dataset=dataset,
        solver=solver_instance,
        scorer=final_digit_model_graded_qa(config.judges, judge_temperature=config.judge_temperature),
        max_retries=3,
        max_messages=20,
        num_batches=1
    )


###########################################################
# Result Combination
###########################################################

def combine_csv_results(config: Config) -> None:
    csv_files = sorted(config.output_dir.glob('results_*.csv'))
    if not csv_files:
        logging.error("No CSV files found to combine.")
        return

    combined_df = combine_csv_files(csv_files)

    # Identify columns
    cols = list(combined_df.columns)
    answer_cols = [c for c in cols if c.endswith('_answer')]
    tag_cols    = [c for c in cols if c.startswith('tag')]
    # Everything else that isn't an answer col or a tag col
    other_cols  = [c for c in cols if c not in answer_cols + tag_cols]

    # We want sample_id, input at the front (if they exist)
    front_cols = []
    for c in ['sample_id', 'input']:
        if c in other_cols:
            front_cols.append(c)
            other_cols.remove(c)

    # Final order: [front_cols] + [answer_cols] + [other_cols] + [tag_cols]
    final_order = front_cols + answer_cols + other_cols + tag_cols

    # Reindex the DataFrame with the new column order
    combined_df = combined_df[final_order]

    combined_path = config.output_dir / 'results_combined.csv'
    combined_df.to_csv(combined_path, index=False)
    logging.info(f"Combined CSV saved to: {combined_path}")

###########################################################
# Main Function
###########################################################

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run the AHA benchmark evaluation")
    parser.add_argument('--model', required=True, help='Model to evaluate')
    parser.add_argument('--judges', default="anthropic/claude-3-5-sonnet-20241022,google/gemini-1.5-pro-002,openai/gpt-4o-2024-08-06",
                        help='Comma-separated list of judge models')
    parser.add_argument('--dataset', help='Path to dataset JSON')
    parser.add_argument('--output_dir', help='Directory for output files')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--num_batches', type=int, default=1, help='Number of batches to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_temperature', type=str, default="0.5", help='Temperature for model generation (or "None" to disable)')
    parser.add_argument('--judge_temperature', type=str, default="0.5", help='Temperature for judge models (or "None" to disable)')
    parser.add_argument('--run-analysis', action='store_true', help='Run analysis.py after evaluation')
    parser.add_argument('--start-batch', type=int, default=0, help='Starting batch number for combining results')

    args = parser.parse_args()

    # Make these visible to aha_evaluation() via global references
    global config, dataset_path
    config = Config(args)
    setup_environment(config)

    # Load full dataset
    try:
        full_data = read_json(config.dataset_path)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    start_time = time.time()
    sampled_files: List[Path] = []
    used_indices: Set[int] = set()

    try:
        # Loop over each requested batch
        for batch in range(config.num_batches):
            config.current_batch = batch + 1

            # Sample a random batch, write to a temp file
            sampled_file, _ = load_and_sample_data(config, full_data, used_indices)
            sampled_files.append(sampled_file)

            # Point dataset_path to this newly created batch file
            dataset_path = sampled_file
            logging.info(f"Processing batch {config.current_batch}/{config.num_batches}")
            eval("aha_evaluation")

            # Immediately run analysis if requested
            if config.run_analysis:
                log_dir = Path("./logs")
                latest_eval = get_latest_file(log_dir, '*.eval')
                if latest_eval:
                    logging.info(f"Results for batch {config.current_batch}:")
                    os.system(
                        f"python /content/aha/analysis.py --log-file {latest_eval} "
                        f"--output-dir {config.output_dir} "
                        f"--solver-name '{config.model}'"
                    )

        total_time = time.time() - start_time
        logging.info(f"Total time for all batches: {total_time:.2f}s")

        # Combine CSV results & do final analysis if requested
        if config.run_analysis:
            logging.info("Combining results from all batches...")
            combine_csv_results(config)
            combined_path = config.output_dir / 'results_combined.csv'
            if combined_path.exists():
                logging.info("Running final analysis on results_combined.csv ...")
                os.system(f"python /content/aha/analysis.py --csv-file {combined_path} --output-dir {config.output_dir}")
            else:
                logging.warning("results_combined.csv not found, skipping final CSV-based analysis")

    finally:
        # Clean up the temporary batch files
        for f in sampled_files:
            if f.exists():
                f.unlink()


if __name__ == "__main__":
    main()
