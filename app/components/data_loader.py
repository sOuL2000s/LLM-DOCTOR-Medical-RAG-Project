import os
import sys
from dotenv import load_dotenv
load_dotenv()

from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.components.vector_store import save_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import MODEL_COMBINATIONS

logger = get_logger(__name__)


def process_and_store_pdfs(target_config=None, rebuild_all=False):
    try:
        logger.info("Starting vectorstore generation pipeline...")
        documents = load_pdf_files()
        if not documents:
            logger.error("No documents found to process.")
            return

        text_chunks = create_text_chunks(documents)

        unique_configs = {}
        if target_config and target_config in MODEL_COMBINATIONS:
            cfg = MODEL_COMBINATIONS[target_config]
            if cfg.get("embeddings") and cfg.get("vectorstore"):
                unique_configs[cfg["embeddings"]] = cfg["vectorstore"]
            logger.info(f"Targeting specific configuration: {target_config}")
        elif rebuild_all:
            logger.info("Rebuilding ALL unique embedding models from MODEL_COMBINATIONS...")
            for config_name, config in MODEL_COMBINATIONS.items():
                if config.get("embeddings") and config.get("vectorstore"):
                    unique_configs[config["embeddings"]] = config["vectorstore"]
        else:
            logger.info("Processing all unique embedding models from MODEL_COMBINATIONS...")
            for config_name, config in MODEL_COMBINATIONS.items():
                if config.get("embeddings") and config.get("vectorstore"):
                    unique_configs[config["embeddings"]] = config["vectorstore"]

        for embed_model, store_path in unique_configs.items():
            logger.info(f"Processing embedding model: {embed_model} -> {store_path}")
            save_vector_store(text_chunks, store_path, embed_model)

        logger.info("Vectorstore processing completed successfully.")
    except Exception as e:
        error_message = CustomException("Pipeline failed", e)
        logger.error(str(error_message))


if __name__ == "__main__":
    # Usage:
    #   python -m app.components.data_loader                  # all configs
    #   python -m app.components.data_loader Setup_Model4     # one config
    #   python -m app.components.data_loader --rebuild-all    # explicit
    args = sys.argv[1:]
    if "--rebuild-all" in args:
        process_and_store_pdfs(rebuild_all=True)
    else:
        target = args[0] if args else None
        process_and_store_pdfs(target)