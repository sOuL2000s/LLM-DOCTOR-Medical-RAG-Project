from langchain_groq import ChatGroq
from app.config.config import GROQ_API_KEY
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
logger = get_logger(__name__)


def load_llm(
    model_name: str = "llama-3.1-8b-instant",
    groq_api_key: str = GROQ_API_KEY,
    temperature: float = 0.3,
    max_tokens: int = 500,
):
    """
    Load a Groq-hosted LLM with configurable temperature and max_tokens.
    The defaults match the values reported in Section 5.1 of the paper.
    """
    try:
        logger.info(
            f"Loading LLM from Groq: model={model_name}, "
            f"temperature={temperature}, max_tokens={max_tokens}"
        )

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info("LLM loaded successfully from Groq.")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load LLM from Groq", e)
        logger.error(str(error_message))
        return None