#!/usr/bin/env python3
"""Analysis module for the AHA benchmark evaluation."""

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import unquote

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from utils import mkd, get_latest_file, extract_timestamp

DFLT_SCORE = 0
CATEGORY_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
INVERSE_CATEGORY_MAP = {v: k for k, v in CATEGORY_MAP.items()}

class ContentProcessor:
    @staticmethod
    def flatten_content(obj: Any) -> str:
        """Convert any content object to flat string."""
        if isinstance(obj, list):
            texts = [getattr(item, 'text', str(item)) for item in obj]
            return " ".join(texts)
        return getattr(obj, 'text', str(obj))

    @staticmethod
    def extract_category(text: str) -> Optional[str]:
        """Extract category [A-E] from text."""
        if not text:
            return None
        match = re.match(r'^\[([A-E])\]', text.splitlines()[0].strip())
        return match.group(1) if match else None

class LogAnalyzer:
    def __init__(self, log_file: Optional[str], log_dir: str = "./logs"):
        self.log_path = self._resolve_log_path(log_file, log_dir)
        self.processor = ContentProcessor()

    def _resolve_log_path(self, log_file: Optional[str], log_dir: str) -> Optional[Path]:
        if not log_file:
            latest = get_latest_file(Path(log_dir), '*.eval')
            return latest
        path = Path(unquote(log_file).removeprefix('file://'))
        return path if path.exists() else Path(log_dir) / path.name

    def parse(self) -> Dict:
        """Parse evaluation log file."""
        if not self.log_path or not self.log_path.exists():
            print(f"Log file not found: {self.log_path}")
            return {}

        log = read_eval_log(str(self.log_path), resolve_attachments=True)
        if not log or not log.samples:
            print(f"Empty or invalid log: {self.log_path}")
            return {}

        scorers = [s.name for s in (log.results.scores or []) if s.name not in ("avg", "error_count")]
        scores = {m: [] for m in scorers}
        categories = {m: [] for m in scorers}
        rows = []

        for sample in log.samples:
            sid = sample.metadata.get("sample_id", 999999)
            ans = next((self.processor.flatten_content(msg.content) 
                       for msg in sample.messages or []
                       if getattr(msg, "source", None) == "generate" 
                       and msg.role == "assistant"), "")

            # Process model outputs
            model_texts = {m: "" for m in scorers}
            for ev in sample.events or []:
                if ev.event == "model" and ev.model in scorers:
                    try:
                        content = ev.output.choices[0].message.content
                        model_texts[ev.model] = self.processor.flatten_content(content).strip()
                    except Exception:
                        continue

            # Process scores
            final_scores = sample.scores.get("final_digit_model_graded_qa", {})
            if hasattr(final_scores, "value"):
                score_dict = {k: int(str(v).strip("[]")) if str(v).strip("[]") in {"-1", "0", "1"}
                            else DFLT_SCORE for k, v in final_scores.value.items()}
            else:
                score_dict = {m: DFLT_SCORE for m in scorers}

            # Build row
            input_flat = self.processor.flatten_content(sample.input or "").replace("\n", " ")
            row = [sid, input_flat, ans.replace("\n", " ")]
            
            for m in scorers:
                scores[m].append(score_dict.get(m, DFLT_SCORE))
                cat = self.processor.extract_category(model_texts[m])
                categories[m].append(CATEGORY_MAP.get(cat, np.nan))
                row.append(model_texts[m])
                row.append(cat or "")
                row.append(scores[m][-1])

            rows.append(row)

        return {
            "models": scorers,
            "scores": [scores[m] for m in scorers],
            "cats": [categories[m] for m in scorers],
            "n": len(log.samples),
            "rows": rows
        }

def parse_csv(csv_file: str) -> Dict:
    """Parse combined CSV results."""
    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"No data in CSV: {csv_file}")
        return {}

    scorers = [col[:-6] for col in df.columns if col.endswith("_score")]
    if not scorers:
        print(f"No *_score columns found in {csv_file}")
        return {}

    scores = []
    categories = []
    for model in scorers:
        scores.append(df[f"{model}_score"].fillna(DFLT_SCORE).astype(int).values)
        cat_col = f"{model}_category"
        if cat_col in df.columns:
            cats = [CATEGORY_MAP.get(x, np.nan) for x in df[cat_col].fillna("")]
        else:
            cats = [np.nan] * len(df)
        categories.append(cats)

    return {"models": scorers, "scores": scores, "cats": categories, "n": len(df), "rows": None}

def analyze_and_save(models: List[str], scores: List[List[float]], cats: List[List[float]], 
                    output_dir: str) -> None:
    """Analyze scores and categories, save results."""
    outd = Path(output_dir)
    mkd(outd)
    if not models:
        print("No models found.")
        return

    arr_sc, arr_ct = np.array(scores, dtype=float), np.array(cats, dtype=float)
    n = arr_sc.shape[1] if arr_sc.ndim == 2 else 0
    print(f"\nNumber of samples: {n}")
    
    # Overall analysis
    flat_scores = arr_sc.flatten()[~np.isnan(arr_sc.flatten())]
    mean_score = np.mean(flat_scores) if len(flat_scores) else float('nan')
    
    # Category analysis
    flat_cats = arr_ct.flatten()[~np.isnan(arr_ct.flatten())]
    tallies = {INVERSE_CATEGORY_MAP[int(x)]: sum(flat_cats == x) 
              for x in np.unique(flat_cats) if not np.isnan(x)}
    sorted_tallies = dict(sorted(tallies.items()))
    
    print(f"All {len(models)} scorers:")
    print(f"  Average score: {mean_score:.3f}")
    print(f"  Categories: {sorted_tallies}")
    
    # Per-model analysis
    results = []
    for i, model in enumerate(models):
        scores_i = arr_sc[i][~np.isnan(arr_sc[i])]
        cats_i = arr_ct[i][~np.isnan(arr_ct[i])]
        model_mean = np.mean(scores_i) if len(scores_i) else float('nan')
        model_tallies = {INVERSE_CATEGORY_MAP[int(x)]: sum(cats_i == x) 
                        for x in np.unique(cats_i) if not np.isnan(x)}
        sorted_model_tallies = dict(sorted(model_tallies.items()))
        
        print(f"\n{model}:")
        print(f"  Average score: {model_mean:.3f}")
        print(f"  Categories: {sorted_model_tallies}")
        
        results.append((model, model_mean, sorted_model_tallies))

    # Save results
    analysis_path = outd / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with analysis_path.open("w", encoding="utf-8") as f:
        f.write(f"Number of samples: {n}\n")
        f.write(f"All {len(models)} scorers:\n  Average score: {mean_score:.3f}\n")
        f.write(f"  Categories: {sorted_tallies}\n\n")
        for model, mean, tally in results:
            f.write(f"{model}:\n  Average score: {mean:.3f}\n")
            f.write(f"  Categories: {tally}\n\n")
    print(f"\nAnalysis summary saved to: {analysis_path}\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evaluation logs or CSV results.")
    parser.add_argument("--log-file", help="Path to the .eval log file")
    parser.add_argument("--log-dir", default="./logs", help="Directory containing .eval log files")
    parser.add_argument("--csv-file", help="Path to the CSV results file")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for output files")
    args = parser.parse_args()

    data = (parse_csv(args.csv_file) if args.csv_file else 
            LogAnalyzer(args.log_file, args.log_dir).parse())
    
    if data.get("n", 0) > 0:
        if data.get("rows"):
            timestamp = extract_timestamp(args.log_file) if args.log_file else datetime.now().strftime("%Y%m%d_%H%M%S")
            cols = ["sample_id", "input", "final_answer"]
            cols.extend([f"{m}_{suffix}" for m in data["models"] 
                        for suffix in ["assessment", "category", "score"]])
            csv_path = Path(args.output_dir) / f"results_{timestamp}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                writer.writerows(data["rows"])
            print(f"Results CSV saved to: {csv_path}")
            
        analyze_and_save(data["models"], data["scores"], data["cats"], args.output_dir)

if __name__ == "__main__":
    main()
