"""
LLM Doctor - Hallucination & Failure-Case Analysis (Reviewer #2)

Reads evaluation_details.csv (Model 4 results) and classifies each answer:

  grounded_correct      : answer fully supported by context and matches reference
  grounded_incomplete   : answer supported by context but omits a key detail
  ungrounded_hallucinated: answer contains a claim not in the context
  refusal               : system correctly declined to answer

Uses a lightweight rule-based heuristic. For a definitive analysis,
inspect the CSV and manually override the "category" column.
"""

import pandas as pd
import re

REFUSAL_MARKERS = [
    "i couldn't find",
    "i could not find",
    "does not contain",
    "not provided",
    "not enough information",
]

HALLUCINATION_MARKERS = [
    "studies show",
    "research indicates",
    "according to recent",
    "in my experience",
    "it is well known",
]


def classify(row):
    answer = str(row.get("answer", "")).lower()
    context = " ".join(row.get("contexts", [])) if isinstance(row.get("contexts"), list) else str(row.get("contexts", ""))
    context_lower = context.lower()

    # Refusal
    if any(m in answer for m in REFUSAL_MARKERS):
        return "refusal"

    # Very low lexical overlap -> likely incomplete or hallucinated
    f1 = row.get("f1", 0.0)
    bert = row.get("bert_f1", 0.0)

    if bert < 0.70:
        # Low semantic overlap with reference
        if any(m in answer for m in HALLUCINATION_MARKERS):
            return "ungrounded_hallucinated"
        return "ungrounded_hallucinated"

    if f1 < 0.15 and bert < 0.80:
        return "grounded_incomplete"

    if f1 < 0.25:
        return "grounded_incomplete"

    return "grounded_correct"


def main():
    df = pd.read_csv("evaluation_details.csv")
    df["category"] = df.apply(classify, axis=1)

    counts = df["category"].value_counts().reindex(
        ["grounded_correct", "grounded_incomplete",
         "ungrounded_hallucinated", "refusal"],
        fill_value=0,
    )
    total = len(df)
    summary = pd.DataFrame({
        "Category": counts.index,
        "Count": counts.values,
        "Percentage": [f"{100 * c / total:.1f}%" for c in counts.values],
    })
    summary.to_csv("hallucination_analysis.csv", index=False)

    print("=" * 60)
    print("HALLUCINATION & FAILURE-CASE ANALYSIS (Model 4)")
    print("=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)

    # Write LaTeX table
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{|l|c|c|}",
        r"\hline",
        r"\textbf{Category} & \textbf{Count (n=%d)} & \textbf{Percentage} \\ \hline" % total,
    ]
    for _, r in summary.iterrows():
        cat = r["Category"].replace("_", "-").title()
        lines.append(f"{cat} & {r['Count']} & {r['Percentage']} \\\\ \\hline")
    lines += [
        r"\end{tabular}",
        r"\caption{Failure-case distribution for Model 4 (LLM Doctor Main).}",
        r"\label{tab:hallucination}",
        r"\end{table}",
    ]
    with open("hallucination_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nLaTeX snippet saved: hallucination_table.tex")


if __name__ == "__main__":
    main()