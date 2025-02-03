#!/usr/bin/env python3
"""
Evaluation script for the AHA (Animal Harm Assessment) benchmark.
Evaluates language model responses for potential impact on animal welfare.
"""

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Tuple

import pandas as pd
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver import generate

from scoring import SOLVER_INSTRUCTIONS, final_digit_model_graded_qa
from utils import setup_logging, read_json, get_latest_file, generate_timestamp

DEFAULT_PATHS = {
    'DATA': Path('/content/aha/data_public.json'),
    'OUTPUT': Path('/content/aha/results'),
    'LOGS': Path('./logs')
}

DEFAULT_MODEL_CONFIG = {
    'max_tokens': 1000,
    'default_judges': [
        'anthropic/claude-3-5-sonnet-20241022',
        'google/gemini-1.5-pro-002',
        'openai/gpt-4o-2024-08-06'
    ]
}

config = None
dataset_path = None

setup_logging(logging.INFO)
logging.getLogger("inspect_ai.model").setLevel(logging.INFO)

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark evaluation."""
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
    current_batch: int = 1
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'BenchmarkConfig':
        """Create configuration from command line arguments."""
        return cls(
            model=args.model,
            judges=args.judges.split(','),
            dataset_path=Path(args.dataset or DEFAULT_PATHS['DATA']),
            output_dir=Path(args.output_dir or DEFAULT_PATHS['OUTPUT']),
            batch_size=args.batch_size,
            seed=args.seed,
            model_temperature=None if args.model_temperature.lower() == "none" 
                            else float(args.model_temperature),
            judge_temperature=None if args.judge_temperature.lower() == "none" 
                            else float(args.judge_temperature),
            run_analysis=args.run_analysis,
            num_batches=args.num_batches
        )

class DataProcessor:
    """Handles data loading and batch processing."""
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.used_indices: Set[int] = set()
        
    def load_and_sample_data(self) -> Tuple[Path, List[Dict[str, Any]]]:
        """Load a batch of data and save to temporary file."""
        full_data = self._load_full_dataset()
        sampled_data = self._sample_batch(full_data)
        return self._save_temp_batch(sampled_data)
    
    def _load_full_dataset(self) -> List[Dict[str, Any]]:
        """Load the complete dataset."""
        try:
            return read_json(self.config.dataset_path)
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")
    
    def _sample_batch(self, full_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sample a batch of data points."""
        available = set(range(len(full_data))) - self.used_indices
        if len(available) < self.config.batch_size:
            self.used_indices.clear()
            available = set(range(len(full_data)))
        
        batch_indices = set(random.sample(list(available), self.config.batch_size))
        self.used_indices.update(batch_indices)
        return [full_data[i] for i in batch_indices]
    
    def _save_temp_batch(self, batch_data: List[Dict[str, Any]]) -> Tuple[Path, List[Dict[str, Any]]]:
        """Save batch to temporary file."""
        temp_dir = self.config.output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        batch_file = temp_dir / f"sampled_data_batch{self.config.current_batch}.json"
        batch_file.write_text(json.dumps(batch_data))
        return batch_file, batch_data

def setup_environment(config: BenchmarkConfig) -> None:
    """Initialize environment for benchmark execution."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    os.environ["INSPECT_EVAL_MODEL"] = config.model
    
    logging.info(
        f"AHA benchmark ({config.batch_size * config.num_batches} total samples, "
        f"{config.num_batches} batches of {config.batch_size}): {config.model}"
    )
    logging.info(
        f"max_tokens: {DEFAULT_MODEL_CONFIG['max_tokens']}, "
        f"model_temperature: {config.model_temperature}, "
        f"judge_temperature: {config.judge_temperature}, "
        f"seed: {config.seed}"
    )

def record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert a data record to a Sample object."""
    return Sample(
        input=f"{record['question']} Your answer must be EXACTLY 30–40 words long.",
        target=SOLVER_INSTRUCTIONS,
        metadata=record.get("generation_tags", {})
    )

@task
def aha_evaluation(config: BenchmarkConfig, dataset_path: Path) -> Task:
    """Define the main evaluation task."""
    dataset = json_dataset(str(dataset_path), sample_fields=record_to_sample)
    solver_config = {"max_tokens": DEFAULT_MODEL_CONFIG['max_tokens']}
    
    if config.model_temperature is not None:
        solver_config["temperature"] = config.model_temperature
    
    solver = generate(**solver_config, cache=True)
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=final_digit_model_graded_qa(
            config.judges, 
            judge_temperature=config.judge_temperature
        ),
        max_retries=3,
        max_messages=20,
        num_batches=1
    )

def setup_argparse() -> argparse.ArgumentParser:
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the AHA benchmark evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model', required=True, help='Model to evaluate')
    model_group.add_argument(
        '--judges',
        default=",".join(DEFAULT_MODEL_CONFIG['default_judges']),
        help='Comma-separated list of judge models'
    )
    model_group.add_argument(
        '--model_temperature',
        type=str,
        default="0.5",
        help='Temperature for model generation (or "None" to disable)'
    )
    model_group.add_argument(
        '--judge_temperature',
        type=str,
        default="0.5",
        help='Temperature for judge models (or "None" to disable)'
    )
    
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--dataset', help='Path to dataset JSON')
    data_group.add_argument('--batch_size', type=int, default=100,
                           help='Number of samples to evaluate')
    data_group.add_argument('--num_batches', type=int, default=1,
                           help='Number of batches to run')
    data_group.add_argument('--seed', type=int, default=42,
                           help='Random seed')
    
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output_dir', help='Directory for output files')
    output_group.add_argument('--run-analysis', action='store_true',
                             help='Run analysis.py after evaluation')
    output_group.add_argument('--start-batch', type=int, default=0,
                             help='Starting batch number for combining results')
    
    return parser

def main() -> None:
    """Main execution function."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    config = BenchmarkConfig.from_args(args)
    setup_environment(config)
    
    processor = DataProcessor(config)
    start_time = time.time()
    sampled_files: List[Path] = []
    
    try:
        for batch in range(config.num_batches):
            config.current_batch = batch + 1
            sampled_file, _ = processor.load_and_sample_data()
            sampled_files.append(sampled_file)
            
            eval(aha_evaluation(config, sampled_file))
            
            if config.run_analysis:
                latest_log = get_latest_file(DEFAULT_PATHS['LOGS'], '*.eval')
                if latest_log:
                    os.system(
                        f"python /content/aha/analysis.py "
                        f"--log-file {latest_log} "
                        f"--output-dir {config.output_dir}"
                    )
        
        total_time = time.time() - start_time
        print(f"Total time for all batches: {total_time:.2f}s")
        
        if config.run_analysis:
            print("Running final analysis...")
            combined_path = config.output_dir / 'results_combined.csv'
            if combined_path.exists():
                os.system(
                    f"python /content/aha/analysis.py "
                    f"--csv-file {combined_path} "
                    f"--output-dir {config.output_dir}"
                )
    
    finally:
        for f in sampled_files:
            if f.exists():
                f.unlink()

if __name__ == "__main__":
    main()
