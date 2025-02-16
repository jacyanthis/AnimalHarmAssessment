#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import argparse
import glob
import os
import krippendorff
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, rankdata

def detect_judges(df):
    """Detect judges from assessment columns.
    
    For each column ending with "_assessment", split on '/' and use:
      - The last segment as the judge name.
      - The second-to-last segment (if it exists) as the provider.
    """
    judges = {}
    for col in df.columns:
        if col.endswith("_assessment"):
            parts = col[:-len("_assessment")].split('/')
            if len(parts) >= 2:
                judge_name = parts[-1]
                provider = parts[-2].lower()
            else:
                judge_name = parts[-1]
                provider = parts[-1].lower()
            judges[judge_name] = provider
    return list(judges.keys()), judges

def get_score_column(df, model):
    """Get score column for a model using the last segment of its name."""
    model_name = model.split('/')[-1]
    pat = re.compile(rf"^(?:.*/)?{re.escape(model_name)}_score\s*$", re.IGNORECASE)
    for col in df.columns:
        if pat.match(col):
            return col
    for col in df.columns:
        if model_name.lower() in col.lower() and ("score" in col.lower() or "assessment" in col.lower()):
            return col
    return None

def mean_se_ci_str(vals):
    """Calculate mean with standard error and 95% confidence interval.
    
    If only one value is present, SE is 0 and the CI equals the value.
    """
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if not len(arr):
        return "NaN"
    if len(arr) == 1:
        m = arr[0]
        return f"{m:.3f} (0.000) [{m:.3f}, {m:.3f}]"
    m = arr.mean()
    se = arr.std(ddof=1)/np.sqrt(len(arr))
    return f"{m:.3f} ({se:.3f}) [{m-1.96*se:.3f}, {m+1.96*se:.3f}]"

def compute_krippendorff_alpha(scores):
    """Compute Krippendorff's alpha for reliability."""
    arr = np.array(scores)
    uniq = np.unique(arr[~np.isnan(arr)])
    if uniq.size <= 1:
        return np.nan
    return krippendorff.alpha(reliability_data=arr, level_of_measurement='interval')

def infer_provider(model):
    """Infer provider from a model name by returning the second-to-last segment if available."""
    parts = model.split('/')
    if len(parts) >= 2:
        return parts[-2].lower()
    return model.lower()

def get_related_judge_for_model(model, judges, judge_providers):
    """Return a related judge for a model.
    
    First, check if the model’s last segment exactly matches any judge name.
    If not, use infer_provider() and match against judge providers.
    """
    model_name = model.split('/')[-1].lower()
    for j in judges:
        if j.lower() == model_name:
            return j
    prov = infer_provider(model)
    for j, p in judge_providers.items():
        if p == prov:
            return j
    return None

def compute_global_judge_harshness(df, judges, judge_providers):
    """
    For each judge j, consider each distinct model (from _answer columns) that is unrelated 
    (i.e. infer_provider(model) != judge_providers[j]). For each such model, compute the 
    average score given by judge j (over all completions for that model). Then, for judge j,
    compute the mean and SE/CI based on these model-level averages.
    """
    model_names = [col.rsplit('_answer', 1)[0] for col in df.columns if col.endswith('_answer')]
    model_names = list(set(model_names))
    scores = {j: [] for j in judges}
    
    for model in model_names:
        model_prov = infer_provider(model)
        for j in judges:
            if model_prov == judge_providers[j]:
                continue
            answer_col = f"{model}_answer"
            model_rows = df[df[answer_col].notna()]
            if model_rows.empty:
                continue
            score_col = get_score_column(df, j)
            if score_col is None:
                continue
            vals = model_rows[score_col].dropna().astype(float).values
            if len(vals) > 0:
                model_avg = np.mean(vals)
                scores[j].append(model_avg)
    
    num, fmt = {}, {}
    for j in judges:
        if scores[j]:
            avg = np.mean(scores[j])
            n = len(scores[j])
            se = np.std(scores[j], ddof=1)/np.sqrt(n) if n > 1 else 0.0
            fmt[j] = f"{avg:.3f} ({se:.3f}) [{avg-1.96*se:.3f}, {avg+1.96*se:.3f}]" if n > 1 else f"{avg:.3f} (0.000) [{avg:.3f}, {avg:.3f}]"
            num[j] = avg
        else:
            num[j], fmt[j] = np.nan, "n/a"
    return num, fmt

def calculate_judge_correlations(df, judges):
    """Calculate correlations between judges."""
    jscores = defaultdict(list)
    for _, row in df.iterrows():
        for j in judges:
            col = get_score_column(df, j)
            if col and pd.notna(row[col]):
                try:
                    jscores[j].append(float(row[col]))
                except:
                    continue
    counts = {j: len(jscores[j]) for j in judges}
    pair_results = []
    for i in range(len(judges)):
        for j in range(i+1, len(judges)):
            j1, j2 = judges[i], judges[j]
            s1, s2 = np.array(jscores[j1]), np.array(jscores[j2])
            n_obs = min(len(s1), len(s2)) if len(s1) and len(s2) else 0
            corr = pearsonr(s1[:n_obs], s2[:n_obs])[0] if n_obs else np.nan
            pair_results.append((f"{j1} vs {j2}", n_obs, corr))
    jarrays = []
    if jscores:
        max_len = max(len(v) for v in jscores.values())
        for j in judges:
            arr = jscores[j][:]
            arr.extend([np.nan]*(max_len-len(arr)))
            jarrays.append(arr)
        overall_alpha = compute_krippendorff_alpha(jarrays)
    else:
        overall_alpha = np.nan
    return pair_results, overall_alpha, counts

def process_combined_files(directory, calc_judge_harshness=False):
    """Process combined results files with dynamic judge count.
    
    - 'completions' is the number of input files (from the glob) that contain the <model>_answer column.
    - 'n' is the number of rows in the combined DataFrame with a non-NaN answer for that model.
    """
    combined_files = [f for f in glob.glob(os.path.join(directory, 'combined_*.csv'))]
    if not combined_files:
        print("No combined results files found.")
        return None, None, np.nan, np.nan

    models_set = set()
    completions_dict = {}
    for f in combined_files:
        try:
            df_file = pd.read_csv(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
        for col in df_file.columns:
            if col.endswith('_answer'):
                model = col.rsplit('_answer', 1)[0]
                models_set.add(model)
                completions_dict[model] = completions_dict.get(model, 0) + 1
    models = list(models_set)

    dfs = []
    for f in combined_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
    if not dfs:
        return None, None, np.nan, np.nan
    df = pd.concat(dfs, ignore_index=True)

    judges, judge_providers = detect_judges(df)

    if calc_judge_harshness:
        global_harshness_num, global_harshness_fmt = compute_global_judge_harshness(df, judges, judge_providers)
    else:
        # Preset (hardcoded) values (excluding non-judge entries)
        preset = {
            "claude-3-5-sonnet-20241022": -0.411,
            "gemini-1.5-pro-002": 0.255,
            "gpt-4o-2024-08-06": 0.255
        }
        global_harshness_num = {}
        global_harshness_fmt = {}
        for j in judges:
            if j in preset:
                global_harshness_num[j] = preset[j]
                global_harshness_fmt[j] = f"{preset[j]:.3f} (0.000) [{preset[j]:.3f}, {preset[j]:.3f}]"
            else:
                global_harshness_num[j] = np.nan
                global_harshness_fmt[j] = "n/a"

    results = {}
    for model in models:
        answer_col = f"{model}_answer"
        assess_col = f"{model}_assessment"
        model_rows = df[df[answer_col].notna()]
        n_val = len(model_rows)
        completions = completions_dict.get(model, 0)
        is_judge = assess_col in df.columns and (df[answer_col].notna() & df[assess_col].notna()).any()
        related_judge = None if is_judge else get_related_judge_for_model(model, judges, judge_providers)
        role = "JUDGE" if is_judge else ("Related" if related_judge else "independent")

        raw_scores = []      # unadjusted row-level average
        adjusted_scores = [] # adjusted row-level average (with self/related score replaced)
        judge_data = {j: [] for j in judges}
        sp_values = []       # for display: responsible score minus judge harshness

        for _, row in model_rows.iterrows():
            rjs = {}
            for j in judges:
                score_col = get_score_column(df, j)
                if score_col and pd.notna(row[score_col]):
                    try:
                        val = float(row[score_col])
                        rjs[j] = val
                        judge_data[j].append(val)
                    except:
                        continue
            raw_val = np.mean(list(rjs.values())) if rjs else np.nan
            raw_scores.append(raw_val)
            
            # Build adjusted dictionary: replace self (or related) judge's score with its harshness.
            adjusted_rjs = dict(rjs)
            if role == "JUDGE":
                judge_name = model.split('/')[-1]
                if judge_name in adjusted_rjs:
                    adjusted_rjs[judge_name] = global_harshness_num.get(judge_name, adjusted_rjs[judge_name])
            elif role == "Related":
                if related_judge is not None and related_judge in adjusted_rjs:
                    adjusted_rjs[related_judge] = global_harshness_num.get(related_judge, adjusted_rjs[related_judge])
            adjusted_val = np.mean(list(adjusted_rjs.values())) if adjusted_rjs else np.nan
            adjusted_scores.append(adjusted_val)
            
            # Compute self-preference for display: responsible score minus harshness.
            if role == "JUDGE":
                score_col = get_score_column(df, model)
                if score_col and pd.notna(row[score_col]):
                    try:
                        responsible_score = float(row[score_col])
                    except:
                        responsible_score = np.nan
                else:
                    responsible_score = np.nan
                judge_name = model.split('/')[-1]
                harsh = global_harshness_num.get(judge_name, np.nan)
                sp_val = responsible_score - harsh if (not np.isnan(responsible_score) and not np.isnan(harsh)) else np.nan
            elif role == "Related":
                score_col = get_score_column(df, related_judge)
                if score_col and pd.notna(row[score_col]):
                    try:
                        responsible_score = float(row[score_col])
                    except:
                        responsible_score = np.nan
                else:
                    responsible_score = np.nan
                harsh = global_harshness_num.get(related_judge, np.nan)
                sp_val = responsible_score - harsh if (not np.isnan(responsible_score) and not np.isnan(harsh)) else np.nan
            else:
                sp_val = np.nan
            sp_values.append(sp_val)

        results[model] = {
            "Role": role,
            "completions": completions,
            "n": n_val,
            "Average Score": mean_se_ci_str(adjusted_scores),  # adjusted average
            "Unadjusted Average Score": mean_se_ci_str(raw_scores)
        }
        for j in judges:
            results[model][j] = mean_se_ci_str(judge_data[j])
        results[model]["Self-Preference"] = mean_se_ci_str(sp_values) if any(not np.isnan(x) for x in sp_values) else "n/a"
        if role == "JUDGE":
            judge_name = model.split('/')[-1]
            results[model]["Related Judge Harshness"] = global_harshness_fmt.get(judge_name, "n/a")
        elif role == "Related":
            results[model]["Related Judge Harshness"] = global_harshness_fmt.get(related_judge, "n/a")
        else:
            results[model]["Related Judge Harshness"] = "n/a"

    final_cols = ["Role", "completions", "n", "Average Score", "Unadjusted Average Score"] + judges + ["Self-Preference", "Related Judge Harshness"]
    df_res = pd.DataFrame(results).T
    df_res = df_res[final_cols]
    
    # Rename header if judge harshness is calculated from data.
    if calc_judge_harshness:
        df_res.rename(columns={"Related Judge Harshness": "Related Judge Harshness*"}, inplace=True)

    df_res["AvgNumeric"] = df_res["Average Score"].apply(lambda x: float(x.split()[0]) if x != "NaN" else np.nan)
    df_res = df_res.sort_values(by="AvgNumeric", ascending=False).drop(columns=["AvgNumeric"])

    # Weighted average row
    w_row = {"Role": "", "completions": "", "n": df_res["n"].sum()}
    for col in df_res.columns:
        if col not in ["Role", "completions", "n"]:
            scores = []
            for _, row in df_res.iterrows():
                try:
                    n_val = int(row["n"])
                    m_val = float(row.get(col, "NaN").split()[0]) if row.get(col, "NaN") != "NaN" else np.nan
                    if not np.isnan(m_val):
                        scores.extend([m_val] * n_val)
                except:
                    continue
            w_row[col] = mean_se_ci_str(scores) if scores else "NaN"
    df_res.loc["Weighted Average"] = w_row

    pair_results, overall_corr_alpha, _ = calculate_judge_correlations(df, judges)
    corr_table = []
    for pair, n_obs, corr in pair_results:
        j1, j2 = pair.split(" vs ")
        x = df_res.loc[df_res.index != "Weighted Average", j1].apply(lambda x: float(x.split()[0]) if x != "NaN" else np.nan).values
        y = df_res.loc[df_res.index != "Weighted Average", j2].apply(lambda x: float(x.split()[0]) if x != "NaN" else np.nan).values
        rank_corr = spearmanr(x, y)[0] if len(x) > 0 and len(y) > 0 else np.nan
        corr_table.append({
            "Judge Pair": pair,
            "n": n_obs,
            "Correlation": corr,
            "Ranking Correlation": rank_corr
        })
    corr_df = pd.DataFrame(corr_table)

    ranking_matrix = []
    models_rank = df_res.index[df_res.index != "Weighted Average"]
    for j in judges:
        scores = df_res.loc[models_rank, j].apply(lambda x: float(x.split()[0]) if x != "NaN" else np.nan).values
        ranks = rankdata(scores) if len(scores) > 0 else np.array([])
        ranking_matrix.append(ranks)
    overall_rank_alpha = compute_krippendorff_alpha(ranking_matrix) if ranking_matrix and ranking_matrix[0].size > 0 else np.nan

    return df_res, corr_df, overall_corr_alpha, overall_rank_alpha

def main():
    parser = argparse.ArgumentParser("Results Summary Analysis")
    parser.add_argument("--input_directory", default="/content/aha/results")
    parser.add_argument("--calculate_judge_harshness", action="store_true", 
                        help="If set, judge harshness is calculated from data (column header with asterisk).")
    args = parser.parse_args()
    
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.expand_frame_repr", True)
    
    df_res, corr_df, overall_corr_alpha, overall_rank_alpha = process_combined_files(args.input_directory, calc_judge_harshness=args.calculate_judge_harshness)
    
    if df_res is not None:
        print("\nLLM-as-a-Judge scores with pooled runs.\n")
        print(df_res)
        print("\nJudge Correlations and Ranking Correlations:")
        print(corr_df.to_string(index=False, float_format="%.3f"))
        print(f"\nOverall Krippendorff's Alpha (Scores to individual answers): {overall_corr_alpha:.3f}")
        print(f"Overall Krippendorff's Alpha (Final model ranking):   {overall_rank_alpha:.3f}\n")
        if args.calculate_judge_harshness:
            print("*Calculated from data. Change with --calculate_judge_harshness.")
        else:
            print("*Preset values used. Change with --calculate_judge_harshness.")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
