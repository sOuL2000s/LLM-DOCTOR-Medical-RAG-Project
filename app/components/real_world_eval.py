"""
LLM Doctor - Secondary Evaluation on Real-World Clinical Queries (Reviewer #3)

Evaluates the main Model 4 system on a small set of clinical-style queries
that are structurally different from encyclopedia entries.
"""

import pandas as pd
from datasets import Dataset
from bert_score import score as bert_score_func
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from app.components.retriever import create_qa_chain
from app.components.llm import load_llm
from app.components.embeddings import get_embedding_model
from app.config.config import MODEL_COMBINATIONS
from test_questions import REAL_WORLD_SET


def token_f1(pred, gt):
    p = pred.lower().split()
    g = gt.lower().split()
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def main():
    config_name = "Setup_Model4"
    config = MODEL_COMBINATIONS[config_name]
    qa_chain = create_qa_chain(config_name)

    rows = []
    for item in REAL_WORLD_SET:
        response = qa_chain.invoke({"question": item["question"], "chat_history": []})
        answer = response.get("answer", "")
        docs = response.get("source_documents", [])
        rows.append({
            "question": item["question"],
            "answer": answer,
            "contexts": [d.page_content for d in docs],
            "ground_truth": item["ground_truth"],
            "f1": token_f1(answer, item["ground_truth"]),
        })

    df = pd.DataFrame(rows)
    df.to_csv("real_world_evaluation_details.csv", index=False)

    _, _, F1 = bert_score_func(
        df["answer"].tolist(), df["ground_truth"].tolist(),
        model_type="distilbert-base-uncased", lang="en", verbose=False,
    )
    df["bert_f1"] = [f.item() for f in F1]

    llm_obj = load_llm(model_name=config["llm"],
                       temperature=config.get("temperature", 0.2),
                       max_tokens=config.get("max_tokens", 500))
    embed_model = get_embedding_model(config["embeddings"])

    ds = Dataset.from_dict({
        "question": df["question"].tolist(),
        "answer": df["answer"].tolist(),
        "contexts": df["contexts"].tolist(),
        "ground_truth": df["ground_truth"].tolist(),
    })
    ragas_res = evaluate(ds, metrics=[faithfulness, answer_relevancy],
                         llm=LangchainLLMWrapper(llm_obj),
                         embeddings=LangchainEmbeddingsWrapper(embed_model))

    try:
        scores = ragas_res.to_pandas().mean(numeric_only=True).to_dict()
    except Exception:
        scores = getattr(ragas_res, "scores", {})

    summary = {
        "n": len(df),
        "Avg_F1": df["f1"].mean(),
        "Avg_BERTScore": df["bert_f1"].mean(),
        **scores,
    }
    pd.DataFrame([summary]).to_csv("real_world_evaluation_summary.csv", index=False)

    print("=" * 60)
    print("SECONDARY EVALUATION ON REAL-WORLD CLINICAL QUERIES")
    print("=" * 60)
    for k, v in summary.items():
        try:
            print(f"{k:22}: {float(v):.4f}")
        except (TypeError, ValueError):
            print(f"{k:22}: {v}")
    print("=" * 60)
    print("\nPaste these numbers into Section 5.7 of the paper.")


if __name__ == "__main__":
    main()