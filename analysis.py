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
from inspect_ai.log import read_eval_log
from utils import mkd, get_latest_file, extract_timestamp

DFLT_SCORE = 0

def flatten_claude_content(obj) -> str:
    """
    Convert either a string, or a list-of-likely-ContentText objects,
    into a flat string for further processing.
    """
    if isinstance(obj, list):
        texts = []
        for item in obj:
            if hasattr(item, "text"):
                texts.append(item.text)
            else:
                texts.append(str(item))
        return " ".join(texts)
    return str(obj)

def cat_letter(txt: str):
    """
    Extract bracketed category [A-E] from the first line if present.
    Returns None if not found.
    """
    lines = txt.splitlines()
    if not lines:
        return None
    m = re.match(r'^\[([A-E])\]', lines[0].strip())
    return m.group(1) if m else None

def parse_eval(log_file: str | None, log_dir: str = "./logs") -> dict:
    """
    Parse an .eval log from Inspect AI to produce data needed for CSV output & analysis.
    Now also extracts 'tags' from sample.metadata if present.
    """
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
    max_tag_count = 0  # track how many tags we need columns for

    for sample in log.samples:
        sid = sample.metadata.get("sample_id", 999999)
        # Retrieve the record's tags from metadata if present
        tags = sample.metadata.get("tags", [])
        max_tag_count = max(max_tag_count, len(tags))

        # Flatten the final assistant answer
        ans = ""
        for msg in sample.messages or []:
            if getattr(msg, "source", None) == "generate" and msg.role == "assistant":
                ans = flatten_claude_content(msg.content)
                break

        # Gather text from each judge
        pm_txt = {m: "" for m in scorers}
        for ev in sample.events or []:
            if ev.event == "model" and ev.model in scorers:
                try:
                    out = ev.output.choices[0].message.content
                    pm_txt[ev.model] = flatten_claude_content(out).strip()
                except Exception:
                    pass

        # Parse final numeric scores
        final_scores = sample.scores.get("final_digit_model_graded_qa", {})
        if hasattr(final_scores, "value"):
            newvals = {}
            for k, v in final_scores.value.items():
                vv = str(v).strip("[]")
                newvals[k] = int(vv) if vv in {"-1", "0", "1"} else DFLT_SCORE
            final_scores = newvals

        # Fill out per-model scores & categories
        for m in scorers:
            sc_val = final_scores.get(m, DFLT_SCORE)
            sc_scores[m].append(sc_val)

            c = cat_letter(pm_txt[m])
            sc_cats[m].append(catmap.get(c, np.nan))

        # Build a row: [sample_id, input, final_answer]
        inp = flatten_claude_content(sample.input or "").replace("\n", " ")
        ans_flat = ans.replace("\n", " ")
        row = [sid, inp, ans_flat]

        # Each model's raw assessment
        for m in scorers:
            row.append(pm_txt[m])
        # Each model's category letter
        for m in scorers:
            last_c = sc_cats[m][-1]
            letter = ""
            if not np.isnan(last_c):
                letter = next((ltr for ltr, val in catmap.items() if val == int(last_c)), "")
            row.append(letter)
        # Each model's numeric score
        for m in scorers:
            row.append(sc_scores[m][-1])

        # Finally, store the row itself + tags for later
        # We'll append tags at the end, or in write_csv
        # For now, keep them in row; we'll handle them below in write_csv
        row.append(tags)
        rows.append(row)

    return {
        "models": scorers,
        "scores": [sc_scores[m] for m in scorers],
        "cats": [sc_cats[m] for m in scorers],
        "n": len(log.samples),
        "rows": rows,
        "max_tag_count": max_tag_count
    }

def parse_csv(csv_file: str) -> dict:
    """
    Parse a CSV results file, returning model arrays and category arrays
    so we can re-run `analyze`.
    """
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

    # Rebuild numeric scores & categories
    alpha_sc = []
    alpha_cat = []
    for m in scorers:
        s = df[f"{m}_score"].fillna(DFLT_SCORE).astype(int).values
        catcol = f"{m}_category"
        if catcol in df.columns:
            c_list = [
                catmap[x] if x in catmap else np.nan
                for x in df[catcol].fillna("").values
            ]
        else:
            c_list = [np.nan] * n
        alpha_sc.append(s)
        alpha_cat.append(c_list)

    return {
        "models": scorers,
        "scores": alpha_sc,
        "cats": alpha_cat,
        "n": n,
        "rows": None,
    }

def write_csv(
    rows, 
    models, 
    output_dir: str, 
    log_file: str = "", 
    max_tag_count: int = 0,
    solver_name: str = ""
) -> None:
    """
    Write CSV with columns:
      sample_id, input, <solver_name>_answer,
      each model's assessment, each model's category, each model's score, tag columns...
    """
    outd = Path(output_dir)
    mkd(outd)
    timestamp = extract_timestamp(log_file) or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build standard columns
    # The third column used to be "final_answer"
    # Now we rename it to <solver_name>_answer (if solver_name is present)
    answer_col = "final_answer" if not solver_name else f"{solver_name}_answer"

    cols = ["sample_id", "input", answer_col]
    for m in models:
        cols.append(f"{m}_assessment")
    for m in models:
        cols.append(f"{m}_category")
    for m in models:
        cols.append(f"{m}_score")

    # Add tagN columns
    for i in range(max_tag_count):
        cols.append(f"tag{i+1}")

    csv_path = outd / f"results_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)

        for row in rows:
            *core_cols, tags_list = row
            # (core_cols includes sample_id, input, final_answer, ... model columns, scores)
            # expand tags into columns
            row_tags = []
            if isinstance(tags_list, list):
                row_tags = tags_list[:]
            row_tags += [""] * (max_tag_count - len(row_tags))
            final_line = list(core_cols) + row_tags
            writer.writerow(final_line)

    print(f"Results CSV saved to: {csv_path}")

def analyze(models, scores, cats, output_dir: str) -> None:
    """
    Summarize results: average score, category tallies, etc.
    """
    outd = Path(output_dir)
    mkd(outd)
    if not models:
        print("No models found.")
        return

    arr_sc = np.array(scores, dtype=float)  # shape (num_models, n)
    arr_ct = np.array(cats,   dtype=float)  # shape (num_models, n)
    n = arr_sc.shape[1] if arr_sc.ndim == 2 else 0
    print(f"\nNumber of samples: {n}")

    fsc = arr_sc.flatten()
    fsc = fsc[~np.isnan(fsc)]
    meanv = np.mean(fsc) if len(fsc) else float('nan')

    fct = arr_ct.flatten()
    fct = fct[~np.isnan(fct)]
    inv = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E'}

    tally = {}
    for x in fct:
        letter = inv.get(int(x), "")
        if letter:
            tally[letter] = tally.get(letter, 0) + 1
    sorted_tally = dict(sorted(tally.items()))

    print(f"All {len(models)} scorers:\n  Average score: {meanv:.3f}\n  Categories: {sorted_tally}")

    for i, m in enumerate(models):
        sc_vals = arr_sc[i][~np.isnan(arr_sc[i])]
        avg_m = np.mean(sc_vals) if len(sc_vals) else float('nan')

        cat_vals = arr_ct[i][~np.isnan(arr_ct[i])]
        sub_tally = {}
        for cval in cat_vals:
            letter = inv.get(int(cval), "")
            if letter:
                sub_tally[letter] = sub_tally.get(letter, 0) + 1
        sub_tally = dict(sorted(sub_tally.items()))
        print(f"\n{m}:\n  Average score: {avg_m:.3f}\n  Categories: {sub_tally}")

    analysis_path = outd / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with analysis_path.open("w", encoding="utf-8") as f:
        f.write(f"Number of samples: {n}\n")
        f.write(f"All {len(models)} scorers:\n  Average score: {meanv:.3f}\n  Categories: {sorted_tally}\n\n")

        for i, m in enumerate(models):
            sc_vals = arr_sc[i][~np.isnan(arr_sc[i])]
            avg_m = np.mean(sc_vals) if len(sc_vals) else float('nan')

            cat_vals = arr_ct[i][~np.isnan(arr_ct[i])]
            s_tally = {}
            for cval in cat_vals:
                letter = inv.get(int(cval), "")
                if letter:
                    s_tally[letter] = s_tally.get(letter, 0) + 1
            s_tally = dict(sorted(s_tally.items()))
            f.write(f"{m}:\n  Average score: {avg_m:.3f}\n  Categories: {s_tally}\n\n")

    print(f"\nAnalysis summary saved to: {analysis_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation logs or CSV results.")
    parser.add_argument("--log-file")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--csv-file")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--solver-name", default="", help="Solver model name for naming the final_answer column.")
    args = parser.parse_args()

    if args.csv_file:
        d = parse_csv(args.csv_file)
        if d.get("n", 0) > 0:
            analyze(d["models"], d["scores"], d["cats"], args.output_dir)
    else:
        d = parse_eval(args.log_file, args.log_dir)
        if d.get("n", 0) > 0:
            log_filename = Path(args.log_file).name if args.log_file else ""
            write_csv(
                d["rows"], 
                d["models"], 
                args.output_dir, 
                log_file=log_filename,
                max_tag_count=d.get("max_tag_count", 0),
                solver_name=args.solver_name  # pass the solver name here
            )
            analyze(d["models"], d["scores"], d["cats"], args.output_dir)

if __name__ == "__main__":
    main()
