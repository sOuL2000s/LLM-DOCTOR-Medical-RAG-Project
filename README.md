# 🩺 LLM Doctor: A Retrieval-Augmented Generation (RAG) Medical Chatbot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-blueviolet.svg)](https://www.langchain.com/)

This project is a complete, reproducible research pipeline for a **Retrieval-Augmented Generation (RAG) system** designed to answer medical questions using a specific local corpus of medical PDF documentation. It is built as a research artifact accompanying a paper on evaluating RAG architectures for medical question-answering.

The system enables reproducible experimentation by comparing multiple RAG configurations (different LLMs, embeddings, rerankers) against a non-RAG baseline. It includes a full evaluation suite using **Token F1**, **BERTScore**, and **Ragas** metrics (faithfulness, answer relevancy), along with automated report and LaTeX table generation for direct inclusion in a paper.

---

## ✨ Key Features & Research Contributions

- **Multiple RAG Configurations:** Compare 4+ distinct RAG setups against a **Non-RAG Baseline** and a **Naive RAG Baseline**. Configurations vary by LLM (Groq-hosted Llama, Qwen, GPT-OSS), embedding model (Nomic, Sentence-Transformers), and reranking (BAAI/bge-reranker-base).
- **Full Evaluation Pipeline:** A single script (`evaluator.py`) runs all configurations, computes metrics, and generates:
    - Per-sample results (`evaluation_details.csv`, `multi_model_comparison_details.csv`)
    - Summary reports (`evaluation_summary.txt`, `multi_model_comparison_summary.csv`)
    - Comparative performance charts (`full_evaluation_plots.png`, `model_comparison_chart.png`)
    - **LaTeX table snippets** (`paper_tables.tex`) ready to paste into a research paper.
- **Reproducible Research:** A dedicated `evaluator_config.yaml` documents all hyperparameters (hardware, software versions, retrieval settings, generation params, and evaluation metrics). A `test_questions.py` file contains the **exact 200-question primary evaluation set** and a **30-question real-world clinical set** used in the paper.
- **Interactive Web Application:** A Flask-based chat UI (`app/application.py`) lets users query the system, select different model configurations, and even choose between "concise" and "detailed" answer styles at runtime.
- **Expert Evaluation Support:** The pipeline includes scripts (`app/components/expert_evaluation.py`) to generate a template for a human physician to rate answers on factual accuracy, clinical safety, and completeness, and then aggregate the results.
- **Hallucination Analysis:** A dedicated module (`app/components/hallucination_analysis.py`) automatically classifies answers into categories like `grounded_correct`, `ungrounded_hallucinated`, and `refusal` to analyze failure modes.

---

## 🚀 Reproducibility: Getting Started

This section provides step-by-step instructions for reviewers and examiners to set up the project and reproduce the results.

### 1. Prerequisites

- **Python 3.9 or higher**.
- **A Groq API Key:** The project uses Groq for fast LLM inference. Get a free key at [console.groq.com](https://console.groq.com/).
- **~10 GB of free disk space** (for dependencies like PyTorch and HuggingFace models).
- **Hardware:** While the system can run on CPU, a CUDA-enabled GPU is highly recommended for faster embedding generation and evaluation.

### 2. Installation

Clone this repository and navigate into the project directory:

```bash
git clone https://github.com/your-username/llm-doctor.git
cd llm-doctor
```

Create and activate a virtual environment:

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install the required dependencies. **Note:** Use `requirements-old.txt` as it contains the tested, stable versions for the research environment.

```bash
pip install --upgrade pip
pip install -r requirements-old.txt
```

### 3. Configuration (Environment Variables)

Create a `.env` file in the project root directory and add your Groq API key:

```env
GROQ_API_KEY="your_actual_groq_api_key_here"
```

### 4. Prepare Your Data

The system requires the source PDF documents to be in the `data/` directory.

1.  If the `data/` folder does not exist, create it.
2.  Place the medical PDF files (e.g., `Current Essentials of Medicine.pdf`, `The_GALE_ENCYCLOPEDIA_of_MEDICINE.pdf`) inside the `data/` folder.

*Note: The evaluation results reported in the paper were generated using the two PDFs listed in the project structure. You must use the same corpus to reproduce the exact numbers.*

### 5. Initialize the Vector Database (One-Time Setup)

Before running the app or the evaluation, you must process the PDFs and create the FAISS vector indexes for all model configurations. This is a critical step.

```bash
python -m app.components.data_loader --rebuild-all
```

This command will:
- Load all PDFs from `data/`.
- Split them into chunks.
- For each unique embedding model in `MODEL_COMBINATIONS`, generate embeddings and save a FAISS index in the `vectorstore/` directory.

This may take several minutes depending on your hardware and the size of the PDFs.

### 6. Running the Project

#### A. Start the Web Application (Interactive Demo)

To interact with the chatbot UI:

```bash
python -m app.application
```

Open your browser and navigate to `http://127.0.0.1:5000`. You can now select a model configuration and ask medical questions based on the PDFs.

#### B. Run the Full Evaluation Pipeline (Reproduce Paper Results)

To reproduce the results, charts, and tables from the paper:

```bash
python evaluator.py
```

This script will:
- Run the `PRIMARY_SET` of 200 questions against all configured models (including baselines).
- Calculate Token F1, BERTScore, and Ragas metrics.
- Generate all CSV reports, summary text files, and plots in the project root.
- **Most importantly, it creates `paper_tables.tex`,** which contains the LaTeX code for the main comparison table.

#### C. Run Secondary Evaluations

**Real-World Clinical Queries:**
To evaluate the main Model 4 on the 30-question `REAL_WORLD_SET`:

```bash
python -m app.components.real_world_eval
```

**Hallucination & Failure-Case Analysis:**
To analyze the failure modes of the main Model 4:

```bash
python -m app.components.hallucination_analysis
```

**Expert Evaluation (Template & Aggregation):**
To generate a template for a human expert to review 50 answers from Model 4:

```bash
python -m app.components.expert_evaluation --template
```
(After a physician fills out `expert_review_filled.csv`, run the following to aggregate results):
```bash
python -m app.components.expert_evaluation --aggregate
```

---

## 📂 Project Structure (Detailed)

| File/Folder | Purpose |
| :--- | :--- |
| `app/application.py` | The main Flask entry point for the web UI. |
| `app/components/` | Core logic for the RAG pipeline. |
| `├── data_loader.py` | CLI script to load PDFs and build all vector stores. |
| `├── embeddings.py` | Loads HuggingFace embedding models. |
| `├── retriever.py` | Constructs the QA chains, including prompts and the optional reranker. |
| `├── llm.py` | Loads the Groq-hosted LLM. |
| `├── reranker.py` | Implements the CrossEncoder reranking logic. |
| `├── vector_store.py` | Handles loading and saving FAISS vector stores. |
| `├── evaluator.py` | **(Main Script)** Runs the full comparative evaluation pipeline. |
| `├── real_world_eval.py` | Runs the secondary evaluation on clinical-style queries. |
| `├── hallucination_analysis.py` | Analyzes and categorizes failure cases. |
| `└── expert_evaluation.py` | Generates and aggregates templates for human expert review. |
| `app/config/config.py` | Defines `MODEL_COMBINATIONS` (the RAG configurations) and global settings. |
| `data/` | **(User Created)** Folder for source PDF files. |
| `vectorstore/` | **(Auto-Generated)** Local FAISS database files. |
| `evaluator.py` | The primary evaluation script. |
| `test_questions.py` | Contains the **exact question sets** (`PRIMARY_SET`, `REAL_WORLD_SET`) used for evaluation. |
| `evaluator_config.yaml` | Documents all hyperparameters and software versions for reproducibility. |
| `extract_paper_tables.py`| A utility to print the contents of the generated result CSVs for easy copying. |

---

## ⚙️ Experimental Setup (Section 5.1 of the Paper)

The key hyperparameters and configurations used for the research are defined in two places:

1.  **`app/config/config.py`**: This file contains the `MODEL_COMBINATIONS` dictionary, which specifies the `llm`, `embeddings`, `rerank` boolean, `initial_k`, and other settings for each model.
2.  **`evaluator_config.yaml`**: This YAML file documents the hardware, software versions, and global evaluation parameters (e.g., `chunk_size`, `temperature`, `ragas_judge`).

**To reproduce the paper's results, do not modify these files.**

### Model Configurations Summary

| Config Name | LLM | Embeddings | Reranker | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `Setup_Default` | Llama-4-Scout-17B | nomic-embed-text-v1.5 | No | Model 1: High Speed & Efficiency |
| `Setup_Enhanced` | GPT-OSS-20B | all-distilroberta-v1 | No | Model 2: Advanced Reasoning |
| `Setup_Medical` | Qwen3-32B | paraphrase-multilingual-MiniLM | **Yes** (bge-reranker) | Model 3: High Precision (Reranked) |
| `Setup_Model4` | Llama-3.1-8B-Instant | all-mpnet-base-v2 | No | **Model 4: LLM Doctor Main System** |
| `Non_RAG_Baseline` | Llama-3.1-8B-Instant | N/A | N/A | Baseline: No Retrieval |
| `Naive_RAG_Baseline` | Llama-3.1-8B-Instant | all-mpnet-base-v2 | No | Baseline: RAG without Reranker |

---

## 📊 Evaluation Metrics Explained

The pipeline reports the following metrics, which are detailed in the paper:

- **Token F1:** A lexical overlap metric that measures the word-level similarity between the AI's answer and a reference (ground truth) answer. It is strict and penalizes paraphrasing.
- **BERTScore:** A semantic similarity metric that uses contextual embeddings to compare the AI's answer and the reference. It is more robust to paraphrasing than Token F1.
- **Faithfulness (Ragas):** Measures whether the generated answer is factually consistent with the retrieved context. A high score means the model is not "hallucinating" information not present in the source documents.
- **Answer Relevancy (Ragas):** Measures how directly the answer addresses the user's question. A high score means the answer is on-topic and complete.

---

## ⚠️ Important Notes

-   **Safety:** This is an AI research project for **educational purposes only**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.
-   **API Limits:** Groq has rate limits. If you run the evaluator or make many requests in a short period, you may encounter `429 Too Many Requests` errors. The evaluation script does not include automatic backoff, so you may need to wait a minute and re-run.
-   **FAISS Deserialization:** The project loads FAISS indexes with `allow_dangerous_deserialization=True`. Only load `.faiss` files that you have generated yourself or that come from a trusted source.
-   **Data Privacy:** The `data/` folder and `vectorstore/` folder are in `.gitignore`. Do not commit sensitive documents or generated vector stores to a public repository.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or issues regarding this research artifact, please open an issue on the GitHub repository.

**Disclaimer:** This project is an independent research effort and is not affiliated with Groq, HuggingFace, LangChain, or any of the other tool providers mentioned.