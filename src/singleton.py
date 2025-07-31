import torch
import logging

from sentence_transformers import SentenceTransformer, CrossEncoder
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL

_embedder_model_bag: dict[str, SentenceTransformer] | None = None
_reranker_model_bag: dict[str, CrossEncoder] | None = None

logger = logging.getLogger(__name__)


def init_sentence_transformers_models(load_default_models: bool = True) -> None:
    global _embedder_model_bag
    global _reranker_model_bag

    if _embedder_model_bag is None:
        _embedder_model_bag = {}

    if _reranker_model_bag is None:
        _reranker_model_bag = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if load_default_models:
        logger.info(f"loading default embedder: {DEFAULT_EMBEDDING_MODEL}")
        default_embedder = SentenceTransformer(
            model_name_or_path=DEFAULT_EMBEDDING_MODEL, device=device
        )

        _embedder_model_bag[DEFAULT_EMBEDDING_MODEL] = default_embedder

        logger.info(f"loading default reranker {DEFAULT_RERANKER_MODEL}")
        default_reranker = CrossEncoder(
            model_name_or_path=DEFAULT_RERANKER_MODEL, device=device
        )

        _reranker_model_bag[DEFAULT_RERANKER_MODEL] = default_reranker


def get_embedder_model(model_name: str) -> SentenceTransformer:
    global _embedder_model_bag
    assert _embedder_model_bag is not None, "_embedder_model_bag is not initialized"

    model = _embedder_model_bag.get(model_name, None)

    if model:
        return model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("not found GPU, using CPU device for inference")

    # load a sentence_transformer model and save to the bag here
    try:
        model = SentenceTransformer(model_name_or_path=model_name, device=device)
    except Exception as e:
        raise RuntimeError(
            f"something went wrong when loading embedding model {model_name}, {e}"
        )

    _embedder_model_bag[model_name] = model
    return model
