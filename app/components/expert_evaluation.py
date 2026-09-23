"""
LLM Doctor - Expert-Based Clinical Evaluation (Reviewer #2 / #3)

Usage:
  1. Run `python -m app.components.expert_evaluation --template`
     This creates expert_review_template.csv with 50 randomly-selected
     Model-4 answers and three empty rating columns.

  2. A qualified physician (MBBS/MD) rates each answer on a 1-5 Likert scale:
       factual_accuracy
       clinical_safety
       completeness

  3. Run `python -m app.components.expert_evaluation --aggregate`
     This reads expert_review_filled.csv and produces
     expert_evaluation_summary.csv + expert_evaluation_table.tex.
"""

import sys
import random
import pandas as pd


def make_template(n=50, seed=42, input_csv="evaluation_details.csv",
                  output_csv="expert_review_template.csv"):
    random.seed(seed)
    df = pd.read_csv(input_csv)
    if len(df) < n:
        print(f"[WARN] Only {len(df)} samples available; using all of them.")
        n = len(df)
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    sample = sample[["question", "answer", "ground_truth"]].copy()
    sample["factual_accuracy"] = ""   # 1-5
    sample["clinical_safety"] = ""    # 1-5
    sample["completeness"] = ""       # 1-5
    sample["reviewer_notes"] = ""
    sample.to_csv(output_csv, index=False)
    print(f"Template written to {output_csv}")
    print("Give this file to a physician reviewer. They fill the rating columns.")
    print("Save as 'expert_review_filled.csv' when done.")


def aggregate(input_csv="expert_review_filled.csv",
              output_csv="expert_evaluation_summary.csv"):
    df = pd.read_csv(input_csv)
    cols = ["factual_accuracy", "clinical_safety", "completeness"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    means = df[cols].mean()
    summary = pd.DataFrame({
        "Dimension": ["Factual Accuracy", "Clinical Safety", "Completeness"],
        "Mean Score (1-5)": [f"{means[c]:.2f}" for c in cols],
    })
    summary.to_csv(output_csv, index=False)
    print("=" * 60)
    print("EXPERT-BASED CLINICAL EVALUATION (n={})".format(len(df)))
    print("=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{|l|c|}",
        r"\hline",
        r"\textbf{Dimension} & \textbf{Mean Score (1--5)} \\ \hline",
    ]
    for _, r in summary.iterrows():
        lines.append(f"{r['Dimension']} & {r['Mean Score (1-5)']} \\\\ \\hline")
    lines += [
        r"\end{tabular}",
        r"\caption{Expert-based clinical evaluation of 50 Model 4 answers (1 = poor, 5 = excellent).}",
        r"\label{tab:expert}",
        r"\end{table}",
    ]
    with open("expert_evaluation_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nLaTeX snippet saved: expert_evaluation_table.tex")


if __name__ == "__main__":
    if "--template" in sys.argv:
        make_template()
    elif "--aggregate" in sys.argv:
        aggregate()
    else:
        print("Usage:")
        print("  python -m app.components.expert_evaluation --template")
        print("  python -m app.components.expert_evaluation --aggregate")