from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import Any

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.components.reranker import get_reranker_model, rerank_documents
from app.config.config import MODEL_COMBINATIONS

logger = get_logger(__name__)

# =============================================================================
# PROMPT TEMPLATES (reported verbatim in Section 5.1 of the paper)
# =============================================================================

CONCISE_PROMPT_TEMPLATE = """You are an expert medical assistant. Your goal is to answer medical questions factually and concisely, using ONLY the provided context.
If the context does not contain enough information to answer the question, state "I couldn't find a definitive answer in the provided medical documents for this question. Please provide more context or rephrase your query." Do not try to make up an answer.
Keep your answer to a maximum of 4-5 sentences.

Context:
{context}

Question:
{question}

Answer:
"""

DETAILED_PROMPT_TEMPLATE = """You are a highly knowledgeable medical researcher. Your goal is to provide a comprehensive and factual answer to the medical question, drawing ONLY from the provided context.
Elaborate on the key aspects and details found in the context to give a thorough explanation.
If the context does not provide sufficient information, clearly state that you cannot fully answer based on the given documents. Avoid external knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

NON_RAG_PROMPT_TEMPLATE = """You are an expert medical assistant. Answer the following medical question factually and concisely based on your own knowledge. Do not say "based on the context" - there is no context. Keep your answer to 4-5 sentences.

Question:
{question}

Answer:
"""


def set_custom_prompt(style: str = "concise"):
    """Return a PromptTemplate based on the requested answer style."""
    if style == "detailed":
        template = DETAILED_PROMPT_TEMPLATE
    else:
        template = CONCISE_PROMPT_TEMPLATE
    return PromptTemplate(template=template, input_variables=["context", "question"])


# =============================================================================
# CUSTOM RERANKING RETRIEVER
# =============================================================================

class RerankingRetriever(BaseRetriever):
    """
    Retrieve documents from a base retriever, then rerank them with a
    CrossEncoder. Attaches the reranker relevance score to each returned
    document's metadata so the application layer can compute a confidence
    score.
    """
    base_retriever: BaseRetriever
    reranker_model: Any
    top_n: int = 2

    def __init__(self, base_retriever: BaseRetriever, reranker_model: Any,
                 top_n: int = 2, **kwargs: Any):
        super().__init__(
            base_retriever=base_retriever,
            reranker_model=reranker_model,
            top_n=top_n,
            **kwargs,
        )

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        retrieved_docs = self.base_retriever.get_relevant_documents(query)
        if self.reranker_model and retrieved_docs:
            return rerank_documents(query, retrieved_docs,
                                    self.reranker_model, self.top_n)
        return retrieved_docs

    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        return self._get_relevant_documents(query, **kwargs)


# =============================================================================
# STANDARD RAG QA CHAIN (Models 1-4)
# =============================================================================

def create_qa_chain(config_name: str = "Setup_Default"):
    try:
        config = MODEL_COMBINATIONS.get(config_name)
        if not config:
            raise CustomException(f"Config {config_name} not found in config.py")

        logger.info(f"Creating QA chain for {config_name}")

        db = load_vector_store(config["vectorstore"], config["embeddings"])
        if db is None:
            raise CustomException(
                f"Vector store not found for {config_name}. Run data_loader first."
            )

        llm = load_llm(
            model_name=config["llm"],
            temperature=config.get("temperature", 0.3),
            max_tokens=config.get("max_tokens", 500),
        )
        if llm is None:
            raise CustomException(f"LLM {config['llm']} failed to load.")

        initial_k = config.get("initial_k", 2)
        base_retriever = db.as_retriever(search_kwargs={"k": initial_k})

        if config.get("rerank"):
            reranker = get_reranker_model(config["reranker_model"])
            retriever_to_use = RerankingRetriever(
                base_retriever=base_retriever,
                reranker_model=reranker,
                top_n=config.get("top_n", 2),
            )
        else:
            retriever_to_use = base_retriever

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever_to_use,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": set_custom_prompt(config.get("answer_style", "concise"))
            },
        )

        logger.info(f"Successfully created QA chain for {config_name}")
        return qa_chain

    except Exception as e:
        error_message = CustomException(f"Failed to create QA chain for {config_name}", e)
        logger.error(str(error_message))
        return None


# =============================================================================
# NON-RAG BASELINE CHAIN (Reviewer #2 request)
# =============================================================================

def create_non_rag_chain(config_name: str = "Non_RAG_Baseline"):
    """
    Return a callable that takes a question and returns an answer generated
    by the LLM WITHOUT any retrieval. Used as the Non-RAG baseline.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    config = MODEL_COMBINATIONS[config_name]
    llm = load_llm(
        model_name=config["llm"],
        temperature=config.get("temperature", 0.2),
        max_tokens=config.get("max_tokens", 500),
    )
    prompt = ChatPromptTemplate.from_template(NON_RAG_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    class _NonRAGChain:
        def invoke(self, inputs, **kwargs):
            question = inputs.get("question", "")
            answer = chain.invoke({"question": question})
            return {"answer": answer, "source_documents": []}

    return _NonRAGChain()