"""
LLM Doctor - Main Evaluator

Runs the full evaluation pipeline and produces:
  - evaluation_details.csv                (per-sample results for Model 4)
  - evaluation_summary.txt                (human-readable summary)
  - full_evaluation_plots.png             (metric distributions for Model 4)
  - multi_model_comparison_details.csv    (per-sample results for all configs)
  - multi_model_comparison_summary.csv    (summary table for all configs)
  - model_comparison_chart.png            (bar chart for all configs)
  - hallucination_analysis.csv            (Model 4 hallucination breakdown)
  - paper_tables.tex                      (LaTeX snippets for the paper)
"""

import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datasets import Dataset
from bert_score import score as bert_score_func
import transformers
import warnings

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from app.components.retriever import create_qa_chain, create_non_rag_chain
from app.components.llm import load_llm
from app.components.embeddings import get_embedding_model
from app.config.config import MODEL_COMBINATIONS
from test_questions import PRIMARY_SET


# =============================================================================
# UTILITIES
# =============================================================================

def calculate_token_f1(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens) if pred_tokens else 0.0
    rec = len(common) / len(gt_tokens) if gt_tokens else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


def get_safe_ragas_scores(rag_results):
    try:
        if hasattr(rag_results, "scores") and isinstance(rag_results.scores, dict):
            return rag_results.scores
        elif hasattr(rag_results, "to_pandas"):
            return rag_results.to_pandas().mean(numeric_only=True).to_dict()
    except Exception:
        pass
    return getattr(rag_results, "scores", {})


# =============================================================================
# SINGLE-CONFIG EVALUATION
# =============================================================================

def run_evaluation_for_config(config_name, question_set):
    print(f"\n>>> EVALUATING CONFIGURATION: {config_name} <<<")
    config = MODEL_COMBINATIONS[config_name]

    if config_name == "Non_RAG_Baseline":
        qa_chain = create_non_rag_chain(config_name)
    else:
        qa_chain = create_qa_chain(config_name)

    if qa_chain is None:
        raise RuntimeError(f"QA Chain for '{config_name}' could not be initialized.")

    results_data = []
    for item in question_set:
        query = item["question"]
        try:
            response = qa_chain.invoke({"question": query, "chat_history": []})
        except Exception as e:
            print(f"  [WARN] {config_name} failed on '{query[:40]}...': {e}")
            response = {"answer": "ERROR", "source_documents": []}

        answer = response.get("answer", "")
        source_docs = response.get("source_documents", [])
        context = [doc.page_content for doc in source_docs] if source_docs else ["No context"]

        results_data.append({
            "config": config_name,
            "question": query,
            "answer": answer,
            "contexts": context,
            "ground_truth": item["ground_truth"],
            "f1": calculate_token_f1(answer, item["ground_truth"]),
        })

    # BERTScore
    all_answers = [r["answer"] for r in results_data]
    all_gt = [r["ground_truth"] for r in results_data]
    try:
        _, _, F1 = bert_score_func(all_answers, all_gt,
                                    model_type="distilbert-base-uncased",
                                    lang="en", verbose=False)
        for i, score in enumerate(F1):
            results_data[i]["bert_f1"] = score.item()
    except Exception as e:
        print(f"  [WARN] BERTScore failed: {e}")
        for r in results_data:
            r["bert_f1"] = 0.0

    # RAGAS
    try:
        llm_obj = load_llm(
            model_name=config["llm"],
            temperature=config.get("temperature", 0.2),
            max_tokens=config.get("max_tokens", 500),
        )
        embed_model = get_embedding_model(
            model_name=config["embeddings"] or "sentence-transformers/all-mpnet-base-v2"
        )
        dataset = Dataset.from_dict({
            "question": [r["question"] for r in results_data],
            "answer": [r["answer"] for r in results_data],
            "contexts": [r["contexts"] for r in results_data],
            "ground_truth": [r["ground_truth"] for r in results_data],
        })
        ragas_res = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=LangchainLLMWrapper(llm_obj),
            embeddings=LangchainEmbeddingsWrapper(embed_model),
        )
        scores = get_safe_ragas_scores(ragas_res)
    except Exception as e:
        print(f"  [WARN] RAGAS failed for {config_name}: {e}")
        scores = {"faithfulness": None, "answer_relevancy": None}
        ragas_res = None

    return pd.DataFrame(results_data), scores, ragas_res


# =============================================================================
# COMPARATIVE EVALUATION
# =============================================================================

def run_comparative_evaluation(question_set=None, configs=None):
    question_set = question_set or PRIMARY_SET
    configs = configs or list(MODEL_COMBINATIONS.keys())

    all_detailed_results = []
    summary_data = []

    for config_name in configs:
        df_details, ragas_scores_dict, ragas_full_obj = run_evaluation_for_config(
            config_name, question_set
        )
        all_detailed_results.append((config_name, df_details, ragas_full_obj))

        summary_row = {
            "Config": config_name,
            "Avg_F1": df_details["f1"].mean(),
            "Avg_BERTScore": df_details["bert_f1"].mean(),
            **ragas_scores_dict,
        }
        summary_data.append(summary_row)

    comparison_df = pd.concat([item[1] for item in all_detailed_results])
    comparison_df.to_csv("multi_model_comparison_details.csv", index=False)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("multi_model_comparison_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("      MULTI-MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)
    return summary_df, all_detailed_results


# =============================================================================
# REPORTS & PLOTS
# =============================================================================

def generate_reports(df, rag_results):
    csv_path = "evaluation_details.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[1/3] Detailed CSV report saved: {csv_path}")

    report_path = "evaluation_summary.txt"
    avg_f1 = df["f1"].mean()
    avg_bert = df["bert_f1"].mean()
    avg_ragas = get_safe_ragas_scores(rag_results) if rag_results else {}

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("===========================================\n")
        f.write("      LLM DOCTOR RAG EVALUATION REPORT      \n")
        f.write("===========================================\n\n")
        f.write(f"Total Samples Evaluated: {len(df)}\n")
        f.write("-" * 43 + "\n")
        f.write(f"Average Token F1 Score:  {avg_f1:.4f}\n")
        f.write(f"Average BERTScore:       {avg_bert:.4f}\n")
        f.write("\n--- Ragas Metrics (High Level) ---\n")
        for metric, val in avg_ragas.items():
            try:
                f.write(f"{metric:20}: {float(val):.4f}\n")
            except (TypeError, ValueError):
                f.write(f"{metric:20}: N/A\n")

        f.write("\n" + "=" * 43 + "\n")
        f.write("   PERFORMANCE HIGHLIGHTS (BERTScore)\n")
        f.write("=" * 43 + "\n")
        for i, (_, row) in enumerate(df.nlargest(3, "bert_f1").iterrows(), 1):
            f.write(f"{i}. Question: {row['question']}\n   Score: {row['bert_f1']:.4f}\n\n")

    print(f"[2/3] Comprehensive summary report saved: {report_path}")


def plot_evaluations(df, rag_results):
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        plt.style.use('ggplot')

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    avg_ragas = get_safe_ragas_scores(rag_results) if rag_results else {}

    sns.kdeplot(df["f1"], ax=axes[0], fill=True, label="Token F1", color="royalblue")
    sns.kdeplot(df["bert_f1"], ax=axes[0], fill=True, label="BERTScore", color="seagreen")
    axes[0].set_title("Performance Distribution")
    axes[0].legend()

    df_melted = df.melt(value_vars=["f1", "bert_f1"], var_name="Metric", value_name="Score")
    sns.boxplot(data=df_melted, x="Metric", y="Score", ax=axes[1], palette="pastel")
    axes[1].set_title("Score Statistics")
    axes[1].set_ylim(0, 1.1)

    m_names = [n.replace('_', ' ').title() for n in avg_ragas.keys()]
    m_scores = []
    for v in avg_ragas.values():
        try:
            m_scores.append(float(v))
        except (TypeError, ValueError):
            m_scores.append(0.0)
    sns.barplot(x=m_names, y=m_scores, ax=axes[2], palette="viridis")
    axes[2].set_title("Overall RAG Quality")
    axes[2].set_ylim(0, 1.1)
    for i, v in enumerate(m_scores):
        axes[2].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("full_evaluation_plots.png", dpi=300)
    print("[3/3] High-resolution plots saved: full_evaluation_plots.png")


# =============================================================================
# LATEX TABLE EXPORT (auto-fills the paper)
# =============================================================================

def export_paper_tables(summary_df):
    """
    Produce a LaTeX snippet with the main comparison table.
    Paste this into the paper to replace Table 3.
    """
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{|l|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Configuration} & \textbf{Avg F1} & \textbf{Avg BERTScore} & \textbf{Faithfulness} & \textbf{Answer Relevancy} \\ \hline")
    for _, row in summary_df.iterrows():
        f1 = row.get("Avg_F1", 0.0)
        bs = row.get("Avg_BERTScore", 0.0)
        fa = row.get("faithfulness", None)
        ar = row.get("answer_relevancy", None)
        fa_str = f"{fa:.4f}" if fa is not None and not pd.isna(fa) else "--"
        ar_str = f"{ar:.4f}" if ar is not None and not pd.isna(ar) else "--"
        lines.append(f"{row['Config']} & {f1:.4f} & {bs:.4f} & {fa_str} & {ar_str} \\\\ \\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Comparison of baselines and four RAG configurations (n=200 samples).}")
    lines.append(r"\label{tab:comparison_configs}")
    lines.append(r"\end{table}")

    with open("paper_tables.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nLaTeX table snippet saved: paper_tables.tex")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        # Step 1: run the full comparative evaluation
        summary_results_df, all_detailed_results = run_comparative_evaluation(
            question_set=PRIMARY_SET,
            configs=list(MODEL_COMBINATIONS.keys()),
        )

        # Step 2: extract Model 4 detailed results for its own reports
        model4_config_name = "Setup_Model4"
        model4_detailed_df = None
        model4_ragas_obj = None
        for config_name, df, ragas_obj in all_detailed_results:
            if config_name == model4_config_name:
                model4_detailed_df = df
                model4_ragas_obj = ragas_obj
                break

        if model4_detailed_df is not None:
            print(f"\n--- Generating main reports for '{model4_config_name}' ---")
            generate_reports(model4_detailed_df, model4_ragas_obj)
            plot_evaluations(model4_detailed_df, model4_ragas_obj)

        # Step 3: comparative bar chart
        plt.figure(figsize=(14, 7))
        metrics_to_plot = ["Avg_F1", "Avg_BERTScore", "faithfulness", "answer_relevancy"]
        plot_df = summary_results_df.melt(id_vars="Config", value_vars=metrics_to_plot,
                                          var_name="Metric", value_name="Score")
        sns.barplot(data=plot_df, x="Metric", y="Score", hue="Config")
        plt.title("Comparative Performance Across Configurations")
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("model_comparison_chart.png", dpi=300)
        print("\nComparative chart saved: model_comparison_chart.png")

        # Step 4: export LaTeX table
        export_paper_tables(summary_results_df)

        print("\nPIPELINE FINISHED SUCCESSFULLY")

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()