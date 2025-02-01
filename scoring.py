#!/usr/bin/env python3
"""
Analysis module for the AHA benchmark evaluation.
Analyzes evaluation logs (.eval) or CSV result files, writes a results CSV,
and prints a summary.
"""

import argparse
import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log
from utils import mkd, get_latest_file, extract_timestamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DFLT_SCORE = 0
catmap = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}   # Ensure `catmap` is defined at module level.

def cat_letter(txt: str) -> str | None:
    lines = txt.splitlines()
    if not lines:
        return None
    m = re.match(r'^\[([A-E])\]', lines[0].strip())
    return m.group(1) if m else None

def parse_eval(log_file: str | None, log_dir: str = "./logs") -> dict:
    if not log_file:
        latest = get_latest_file(Path(log_dir), '*.eval')
        if not latest:
            print(f"No .eval logs in {log_dir}")
            return {}
        log_file = latest.as_posix()
    log_file = unquote(log_file)
    if log_file.startswith("file://"):
        log_file = log_file[7:]
    log_path = Path(log_file)
    if not log_path.exists():
        alt = Path(log_dir) / log_path.name
        if alt.exists():
            log_path = alt
        else:
            print(f"Log file not found: {log_file}")
            return {}
    log = read_eval_log(str(log_path), resolve_attachments=True)
    if not log or not log.samples:
        print(f"Empty or invalid log: {log_path}")
        return {}

    scorers = [s.name for s in (log.results.scores or []) if s.name not in ("avg", "error_count")]
    sc_scores = {m: [] for m in scorers}
    sc_cats = {m: [] for m in scorers}
    rows = []

    for sample in log.samples:
        sid = sample.metadata.get("sample_id", 999999)
        ans = ""
        for msg in sample.messages or []:
            if getattr(msg, "source", None) == "generate" and msg.role == "assistant":
                ans = msg.content
                break
        if isinstance(ans, list):
            ans = ans[0].text if ans and hasattr(ans[0], "text") else " ".join(str(x) for x in ans)
        if isinstance(ans, str) and ans.startswith("[ContentText"):
            m = re.search(r"text='(.*?)'", ans)
            if m:
                ans = m.group(1)
        pm_txt = {m: "" for m in scorers}
        for ev in sample.events or []:
            if ev.event == "model" and ev.model in scorers:
                try:
                    out = ev.output.choices[0].message.content
                    txt = out[0].text if isinstance(out, list) else out
                    pm_txt[ev.model] = txt.strip()
                except Exception:
                    pass
        final_scores = sample.scores.get("final_digit_model_graded_qa", {})
        if hasattr(final_scores, "value"):
            newvals = {}
            for k, v in final_scores.value.items():
                vv = str(v).strip("[]")
                try:
                    newvals[k] = int(vv)
                except:
                    newvals[k] = DFLT_SCORE
            final_scores = newvals
        for m in scorers:
            try:
                score_val = int(final_scores.get(m, DFLT_SCORE))
            except:
                score_val = DFLT_SCORE
            sc_scores[m].append(score_val)
            c = cat_letter(pm_txt[m])
            sc_cats[m].append(catmap.get(c, np.nan))
        row = [sid, getattr(sample, "input", "").replace("\n", " "), ans.replace("\n", " ")]
        # appended in order: each model's text, then each model's letter, then each model's score
        for m in scorers:
            row.append(pm_txt[m])
        for m in scorers:
            last_c = sc_cats[m][-1]
            letter = (
                next((ltr for ltr, val in catmap.items() if val == int(last_c)), "")
                if not np.isnan(last_c)
                else ""
            )
            row.append(letter)
        for m in scorers:
            row.append(sc_scores[m][-1])
        rows.append(row)

    return {
        "models": scorers,
        "scores": [sc_scores[m] for m in scorers],
        "cats": [sc_cats[m] for m in scorers],
        "n": len(log.samples),
        "rows": rows,
    }

def parse_csv(csv_file: str) -> dict:
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_file}")
        return {}
    df = pd.read_csv(csv_path, dtype=str)
    if df.empty:
        print(f"No data in CSV: {csv_file}")
        return {}

    # Identify scorers from columns ending with "_score"
    scorers = [col[:-6] for col in df.columns if col.endswith("_score")]
    if not scorers:
        print(f"No *_score columns found in {csv_file}")
        return {}

    n = len(df)
    alpha_sc, alpha_cat = [], []
    for m in scorers:
        score_col = f"{m}_score"
        cat_col = f"{m}_category"
        s = []
        c = []
        for idx in range(n):
            raw_score = df.loc[idx, score_col] if score_col in df.columns else str(DFLT_SCORE)
            raw_score_str = raw_score.strip("[] ") if isinstance(raw_score, str) else str(raw_score)
            try:
                val_s = int(raw_score_str)
            except:
                val_s = DFLT_SCORE
            s.append(val_s)

            raw_cat = df.loc[idx, cat_col] if cat_col in df.columns else ""
            raw_cat_str = raw_cat.strip() if isinstance(raw_cat, str) else ""
            if raw_cat_str in catmap:
                val_c = catmap[raw_cat_str]
            else:
                val_c = np.nan
            c.append(val_c)
        alpha_sc.append(np.array(s, dtype=float))
        alpha_cat.append(np.array(c, dtype=float))

    return {
        "models": scorers,
        "scores": alpha_sc,
        "cats": alpha_cat,
        "n": n,
        "rows": None,
    }

def write_csv(rows, models, output_dir: str, log_file: str = "") -> None:
    mkd(Path(output_dir))
    ts = extract_timestamp(log_file) if log_file else datetime.now().strftime("%Y%m%d_%H%M%S")
    # Column order: sample_id, input, final_answer, then each model's assessment,
    # each model's category, each model's score, then leftover columns
    col_names = ["sample_id", "input", "final_answer"]

    # figure out leftover columns from first row
    # we know we have 3 + (len(models)*3) base columns, leftover is the rest
    base_count = 3 + (len(models) * 3)
    leftover_count = len(rows[0]) - base_count if rows else 0

    # chunk for each model
    for m in models:
        col_names.append(f"{m}_assessment")
    for m in models:
        col_names.append(f"{m}_category")
    for m in models:
        col_names.append(f"{m}_score")

    # leftover columns if present
    if leftover_count > 0:
        col_names.extend([f"tag{i+1}" for i in range(leftover_count)])

    out_csv = Path(output_dir) / f"results_{ts}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(col_names)
        w.writerows(rows)
    print(f"Results CSV saved to: {out_csv}")

def combine_csv_results(config, full_data, start_batch: int) -> None:
    out_dir = Path(config.output_dir)
    csv_files = sorted(out_dir.glob("results_*.csv"))
    if not csv_files:
        logging.error("No CSV files found to combine")
        return
    dfs = [pd.read_csv(f, dtype=str) for f in csv_files]
    col_order = dfs[0].columns
    combined_df = pd.concat([df[col_order] for df in dfs], ignore_index=True)
    start_idx = start_batch * config.batch_size
    tags_list = [
        data.get("tags", [])
        for data in full_data[
            start_idx : start_idx + (config.num_batches * config.batch_size)
        ]
    ]
    max_tags = max(len(t) for t in tags_list) if tags_list else 0
    combined_df["sample_id"] = range(1, len(combined_df) + 1)
    for i in range(max_tags):
        col = f"tag{i+1}"
        combined_df[col] = ""
        for idx, tags in enumerate(tags_list):
            if idx < len(combined_df) and i < len(tags):
                combined_df.at[idx, col] = tags[i]
    out_csv = out_dir / "results_combined.csv"
    combined_df.to_csv(out_csv, index=False)
    logging.info(f"Combined CSV saved to: {out_csv}")

def analyze(models, scores, cats, output_dir: str) -> None:
    mkd(Path(output_dir))
    if not models:
        print("No models found.")
        return
    arr_sc = np.array(scores, dtype=float)
    arr_ct = np.array(cats, dtype=float)
    n = arr_sc.shape[1] if arr_sc.ndim == 2 else 0

    print(f"\nNumber of samples: {n}")
    fsc = arr_sc.flatten()[~np.isnan(arr_sc.flatten())]
    meanv = np.mean(fsc) if len(fsc) else float("nan")
    fct = arr_ct.flatten()[~np.isnan(arr_ct.flatten())]
    tally = {}
    inv = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    for x in fct:
        letter = inv.get(int(x), "")
        if letter:
            tally[letter] = tally.get(letter, 0) + 1
    sorted_tally = dict(sorted(tally.items()))
    print(f"All {len(models)} judges:\n  Average score: {meanv:.3f}\n  Categories: {sorted_tally}")

    for i, m in enumerate(models):
        sc_vals = arr_sc[i][~np.isnan(arr_sc[i])]
        avg_m = np.mean(sc_vals) if len(sc_vals) else float("nan")
        cat_vals = arr_ct[i][~np.isnan(arr_ct[i])]
        sub = {}
        for v in cat_vals:
            letter = inv.get(int(v), "")
            if letter:
                sub[letter] = sub.get(letter, 0) + 1
        sub = dict(sorted(sub.items()))
        print(f"\n{m}:\n  Average score: {avg_m:.3f}\n  Categories: {sub}")

    an_file = Path(output_dir) / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with an_file.open("w", encoding="utf-8") as f:
        f.write(f"Number of samples: {n}\n")
        f.write(f"All {len(models)} judges:\n  Average score: {meanv:.3f}\n  Categories: {sorted_tally}\n\n")
        for i, m in enumerate(models):
            sc_vals = arr_sc[i][~np.isnan(arr_sc[i])]
            avg_m = np.mean(sc_vals) if len(sc_vals) else float("nan")
            cat_vals = arr_ct[i][~np.isnan(arr_ct[i])]
            sub = {}
            for v in cat_vals:
                letter = inv.get(int(v), "")
                if letter:
                    sub[letter] = sub.get(letter, 0) + 1
            sub = dict(sorted(sub.items()))
            f.write(f"{m}:\n  Average score: {avg_m:.3f}\n  Categories: {sub}\n\n")
    print(f"\nAnalysis summary saved to: {an_file}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze .eval logs or CSV results.")
    parser.add_argument("--log-file", help="Path to the .eval log file")
    parser.add_argument("--log-dir", default="./logs", help="Directory with .eval logs")
    parser.add_argument("--csv-file", help="Path to the CSV results file")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for output files")
    args = parser.parse_args()

    if args.csv_file:
        d = parse_csv(args.csv_file)
        if d.get("n", 0) > 0:
            analyze(d["models"], d["scores"], d["cats"], args.output_dir)
    else:
        d = parse_eval(args.log_file, args.log_dir)
        if d.get("n", 0) > 0:
            log_filename = Path(args.log_file).name if args.log_file else ""
            write_csv(d["rows"], d["models"], args.output_dir, log_filename)
            analyze(d["models"], d["scores"], d["cats"], args.output_dir)


if __name__ == "__main__":
    main()
