"""
LLM Doctor - Paper Table Extractor

Reads all result CSVs and prints (or writes) the LaTeX tables you need to
paste into the paper.
"""

import pandas as pd
import os


def safe_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    print("=" * 70)
    print("LLM DOCTOR - PAPER TABLE EXTRACTOR")
    print("=" * 70)

    # --- Table 3 (main comparison) ---
    if os.path.exists("multi_model_comparison_summary.csv"):
        df = pd.read_csv("multi_model_comparison_summary.csv")
        print("\nTABLE 3 - Overall Comparison of RAG Configurations\n")
        for _, row in df.iterrows():
            print(f"  {row['Config']:25} "
                  f"F1={safe_float(row.get('Avg_F1')):.4f}  "
                  f"BERT={safe_float(row.get('Avg_BERTScore')):.4f}  "
                  f"Faith={safe_float(row.get('faithfulness')):.4f}  "
                  f"Rel={safe_float(row.get('answer_relevancy')):.4f}")
        print("\n  LaTeX snippet: paper_tables.tex")

    # --- Table 4 (hallucination) ---
    if os.path.exists("hallucination_analysis.csv"):
        df = pd.read_csv("hallucination_analysis.csv")
        print("\nTABLE 4 - Hallucination & Failure-Case Distribution\n")
        print(df.to_string(index=False))
        print("\n  LaTeX snippet: hallucination_table.tex")

    # --- Table 5 (expert evaluation) ---
    if os.path.exists("expert_evaluation_summary.csv"):
        df = pd.read_csv("expert_evaluation_summary.csv")
        print("\nTABLE 5 - Expert-Based Clinical Evaluation\n")
        print(df.to_string(index=False))
        print("\n  LaTeX snippet: expert_evaluation_table.tex")
    else:
        print("\nTABLE 5 - Expert-Based Clinical Evaluation: SKIPPED (no expert_evaluation_summary.csv found)")

    # --- Section 5.7 (real-world) ---
    if os.path.exists("real_world_evaluation_summary.csv"):
        df = pd.read_csv("real_world_evaluation_summary.csv")
        print("\nSECTION 5.7 - Real-World Clinical Query Evaluation\n")
        print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Done. Copy the numbers / LaTeX snippets into the paper.")
    print("=" * 70)


if __name__ == "__main__":
    main()