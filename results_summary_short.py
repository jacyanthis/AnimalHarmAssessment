#!/usr/bin/env python3
"""
results_summary_short.py

This script loads all CSV files matching "combined_*.csv" from a given results directory
(e.g., "/content/aha/results"), combines them, and then computes and prints summary statistics
for models and judges. For judges, it reports not only the overall scores count but also a breakdown
by each model (i.e. the number of non-empty scores when a model's answer is present).
"""

import argparse
from pathlib import Path
import pandas as pd

def compute_global_stats(df: pd.DataFrame):
    """
    Compute overall statistics from the combined dataframe.
    Returns:
      (number of model answer columns, number of judge assessment columns,
       mean score, score distribution, category distribution, total score count)
    """
    score_cols = [c for c in df.columns if c.endswith("_score")]
    cat_cols   = [c for c in df.columns if c.endswith("_category")]
    model_cols = [c for c in df.columns if c.endswith("_answer")]
    judge_cols = [c for c in df.columns if c.endswith("_assessment")]

    total_score = 0.0
    total_count = 0
    score_dist  = {'-1': 0, '0': 0, '1': 0}
    cat_dist    = {}

    for idx in df.index:
        for sc_col in score_cols:
            val = df.at[idx, sc_col]
            if pd.notna(val) and val in (-1, 0, 1):
                total_score += val
                total_count += 1
                score_dist[str(int(val))] += 1
        for c_col in cat_cols:
            c_val = str(df.at[idx, c_col]).strip()
            if c_val in {'A', 'B', 'C', 'D', 'E'}:
                cat_dist[c_val] = cat_dist.get(c_val, 0) + 1

    mean_val = total_score / total_count if total_count > 0 else 0.0
    # Ensure alphabetical order for category distribution
    ordered_cat = {k: cat_dist.get(k, 0) for k in sorted(['A','B','C','D','E'])}
    return len(model_cols), len(judge_cols), mean_val, score_dist, ordered_cat, total_count

def compute_entity_stats(df: pd.DataFrame, suffix: str) -> dict:
    """
    For each entity (model or judge) determined by a column suffix,
    compute the number of questions answered, the score distribution, and category counts.
    For models (suffix="_answer"), the stats are pooled from all judges' scores.
    For judges (suffix="_assessment"), only the judge's own columns are used.
    """
    out = {}
    entity_cols = [c for c in df.columns if c.endswith(suffix)]
    is_judge = (suffix == "_assessment")
    
    for col in entity_cols:
        name = col[:-len(suffix)]
        # Only consider rows where the entity's column is non-empty.
        row_mask = df[col].notna() & (df[col].astype(str).str.strip() != "")
        relevant_indices = df.index[row_mask]
        
        q_count = len(relevant_indices)
        sc_count = 0
        sc_sum = 0.0
        sc_dist = {'-1': 0, '0': 0, '1': 0}
        c_dist = {}

        for rid in relevant_indices:
            if is_judge:
                # For judges, use their own score and category columns.
                score_col = f"{name}_score"
                cat_col   = f"{name}_category"
                if score_col in df.columns:
                    val = df.at[rid, score_col]
                    if pd.notna(val) and val in (-1, 0, 1):
                        sc_sum += val
                        sc_count += 1
                        sc_dist[str(int(val))] += 1
                if cat_col in df.columns:
                    cat_val = str(df.at[rid, cat_col]).strip()
                    if cat_val in {'A','B','C','D','E'}:
                        c_dist[cat_val] = c_dist.get(cat_val, 0) + 1
            else:
                # For models, pool all judges' scores and categories.
                judge_names = [c[:-11] for c in df.columns if c.endswith("_assessment")]
                for j in judge_names:
                    score_col = f"{j}_score"
                    cat_col = f"{j}_category"
                    if score_col in df.columns:
                        val = df.at[rid, score_col]
                        if pd.notna(val) and val in (-1, 0, 1):
                            sc_sum += val
                            sc_count += 1
                            sc_dist[str(int(val))] += 1
                    if cat_col in df.columns:
                        cat_val = str(df.at[rid, cat_col]).strip()
                        if cat_val in {'A','B','C','D','E'}:
                            c_dist[cat_val] = c_dist.get(cat_val, 0) + 1

        # Order categories alphabetically
        ordered_c_dist = {k: c_dist.get(k, 0) for k in sorted(['A','B','C','D','E'])}
        out[name] = {
            "questions": q_count,
            "score_count": sc_count,
            "sum_score": sc_sum,
            "score_dist": sc_dist,
            "cat_dist": ordered_c_dist
        }
    return out

def compute_judge_model_breakdown(df: pd.DataFrame, judge: str, model_names: list) -> dict:
    """
    For a given judge, compute how many scores were cast for each model.
    For each model (i.e. for each model's answer column), count the rows where:
      - The model's answer (column "{model}_answer") is non-empty, and
      - The judge's score (column "{judge}_score") is non-empty.
    """
    breakdown = {}
    judge_score_col = f"{judge}_score"
    if judge_score_col not in df.columns:
        return breakdown

    for model in model_names:
        model_col = f"{model}_answer"
        if model_col in df.columns:
            count = df[(df[model_col].notna()) & (df[judge_score_col].notna())].shape[0]
            breakdown[model] = count
    return breakdown

def report_entity(name: str, stats: dict, breakdown: dict = None):
    """
    Print a summary report for a given entity (model or judge) based on its statistics.
    If a breakdown dictionary is provided, print that too.
    """
    q  = stats["questions"]
    sc = stats["score_count"]
    total = stats["sum_score"]
    dist = stats["score_dist"]
    cat = stats["cat_dist"]
    avg = (total / sc) if sc > 0 else 0.0

    print(f"{name}:")
    print(f"  Questions: {q}")
    print(f"  Scores Count: {sc}", end='')
    if breakdown is not None:
        # Print breakdown per model in alphabetical order.
        breakdown_ordered = {k: breakdown[k] for k in sorted(breakdown.keys())}
        print(f" (by model: {breakdown_ordered})")
    else:
        print()
    print(f"  Average Score: {avg:.3f} {dist}")
    print(f"  Categories: {cat}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generate summary statistics from combined_*.csv files."
    )
    parser.add_argument(
        "--input_directory",
        type=str,
        default="/content/aha/results",
        help="Directory containing combined_*.csv files (default: /content/aha/results)"
    )
    args = parser.parse_args()

    input_directory = Path(args.input_directory)
    if not input_directory.exists():
        print(f"Directory {input_directory} does not exist.")
        return

    # Find all CSV files with names starting with "combined_"
    csv_files = sorted(input_directory.glob("combined_*.csv"))
    if not csv_files:
        print(f"No combined_*.csv files found in {input_directory}.")
        return

    # Load and combine all CSV files
    df_list = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not df_list:
        print("No CSV data could be loaded.")
        return

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\nCombined {len(df_list)} CSV file(s) with a total of {len(combined_df)} rows.\n")

    # Global statistics
    nm, nj, mean_val, score_dist, cat_dist, s_count = compute_global_stats(combined_df)
    print("Global Summary:")
    print(f"  Total Questions: {len(combined_df)}")
    print(f"  Total Score Count: {s_count}")
    print(f"  Average Score: {mean_val:.3f}")
    print(f"  Score Distribution: {score_dist}")
    print(f"  Category Distribution: {cat_dist}\n")

    # Identify all model names (from columns ending with "_answer")
    model_names = [c[:-7] for c in combined_df.columns if c.endswith("_answer")]

    # Model statistics
    print("Model Statistics:")
    model_stats = compute_entity_stats(combined_df, "_answer")
    for model in sorted(model_stats.keys()):
        report_entity(model, model_stats[model])

    # Judge statistics with per-model breakdown
    print("Judge Statistics:")
    judge_stats = compute_entity_stats(combined_df, "_assessment")
    for judge in sorted(judge_stats.keys()):
        breakdown = compute_judge_model_breakdown(combined_df, judge, model_names)
        report_entity(judge, judge_stats[judge], breakdown)

if __name__ == "__main__":
    main()
