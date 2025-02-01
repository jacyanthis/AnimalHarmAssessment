#!/usr/bin/env python3
"""
Analysis module for the AHA benchmark evaluation.
Analyzes evaluation logs (.eval) or CSV result files, writes a results CSV,
and prints a summary.
"""

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log
from utils import mkd, get_latest_file, extract_timestamp

DFLT_SCORE = 0

def cat_letter(txt: str) -> str | None:
    m = re.match(r'^\[([A-E])\]', txt.splitlines()[0].strip())
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
    catmap = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
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
                newvals[k] = int(vv) if vv in {"-1", "0", "1"} else DFLT_SCORE
            final_scores = newvals
        for m in scorers:
            sc_scores[m].append(final_scores.get(m, DFLT_SCORE))
            c = cat_letter(pm_txt[m])
            sc_cats[m].append(catmap.get(c, np.nan))
        row = [sid, (sample.input or "").replace("\n", " "), ans.replace("\n", " ")]
        for m in scorers:
            row.append(pm_txt[m])
        for m in scorers:
            last_c = sc_cats[m][-1]
            letter = next((ltr for ltr, val in catmap.items() if val == int(last_c)), "") if not np.isnan(last_c) else ""
            row.append(letter)
        for m in scorers:
            row.append(sc_scores[m][-1])
        rows.append(row)
    return {"models": scorers, "scores": [sc_scores[m] for m in scorers],
            "cats": [sc_cats[m] for m in scorers], "n": len(log.samples), "rows": rows}

def parse_csv(csv_file: str) -> dict:
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_file}")
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"No data in CSV: {csv_file}")
        return {}
    scorers = [col[:-6] for col in df.columns if col.endswith("_score")]
    if not scorers:
        print(f"No *_score columns found in {csv_file}")
        return {}
    catmap = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    n = len(df)
    alpha_sc = []
    alpha_cat = []
    for m in scorers:
        s = df[f"{m}_score"].fillna(DFLT_SCORE).astype(int).values
        catcol = f"{m}_category"
        if catcol in df.columns:
            c_list = [catmap[x] if x in catmap else np.nan for x in df[catcol].fillna("").values]
        else:
            c_list = [np.nan] * n
        alpha_sc.append(s)
        alpha_cat.append(c_list)
    return {"models": scorers, "scores": alpha_sc, "cats": alpha_cat, "n": n, "rows": None}

def write_csv(rows, models, output_dir: str, log_file: str = "") -> None:
    outd = Path(output_dir)
    mkd(outd)
    timestamp = extract_timestamp(log_file) if log_file else datetime.now().strftime("%Y%m%d_%H%M%S")
    cols = ["sample_id", "input", "final_answer"]
    for m in models:
        cols.append(f"{m}_assessment")
    for m in models:
        cols.append(f"{m}_category")
    for m in models:
        cols.append(f"{m}_score")
    csv_path = outd / f"results_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)
    print(f"Results CSV saved to: {csv_path}")

def analyze(models, scores, cats, output_dir: str) -> None:
    outd = Path(output_dir)
    mkd(outd)
    if not models:
        print("No models found.")
        return
    arr_sc = np.array(scores, dtype=float)
    arr_ct = np.array(cats, dtype=float)
    n = arr_sc.shape[1] if arr_sc.ndim == 2 else 0
    print(f"\nNumber of samples: {n}")
    fsc = arr_sc.flatten()
    fsc = fsc[~np.isnan(fsc)]
    meanv = np.mean(fsc) if len(fsc) else float('nan')
    fct = arr_ct.flatten()
    fct = fct[~np.isnan(fct)]
    inv = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E'}
    tally = {}
    for letter in [inv[int(x)] for x in fct]:
        tally[letter] = tally.get(letter, 0) + 1
    sorted_tally = dict(sorted(tally.items()))
    print(f"All {len(models)} scorers:\n  Average score: {meanv:.3f}\n  Categories: {sorted_tally}")
    for i, m in enumerate(models):
        sc_vals = arr_sc[i][~np.isnan(arr_sc[i])]
        avg_m = np.mean(sc_vals) if len(sc_vals) else float('nan')
        cat_vals = arr_ct[i][~np.isnan(arr_ct[i])]
        sub = {}
        for val in cat_vals:
            sub[inv[int(val)]] = sub.get(inv[int(val)], 0) + 1
        sub = dict(sorted(sub.items()))
        print(f"\n{m}:\n  Average score: {avg_m:.3f}\n  Categories: {sub}")
    analysis_path = outd / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with analysis_path.open("w", encoding="utf-8") as f:
        f.write(f"Number of samples: {n}\n")
        f.write(f"All {len(models)} scorers:\n  Average score: {meanv:.3f}\n  Categories: {sorted_tally}\n\n")
        for i, m in enumerate(models):
            sc_vals = arr_sc[i][~np.isnan(arr_sc[i])]
            avg_m = np.mean(sc_vals) if len(sc_vals) else float('nan')
            cat_vals = arr_ct[i][~np.isnan(arr_ct[i])]
            sub = {}
            for val in cat_vals:
                sub[inv[int(val)]] = sub.get(inv[int(val)], 0) + 1
            sub = dict(sorted(sub.items()))
            f.write(f"{m}:\n  Average score: {avg_m:.3f}\n  Categories: {sub}\n\n")
    print(f"\nAnalysis summary saved to: {analysis_path}\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evaluation logs or CSV results.")
    parser.add_argument("--log-file", help="Path to the .eval log file")
    parser.add_argument("--log-dir", default="./logs", help="Directory containing .eval log files")
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
