import torch
import logging

from sentence_transformers import SentenceTransformer, CrossEncoder
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL

# TODO: store a max of N models to prevent OOM errors when you store too many models
_embedder_model_bag: dict[str, SentenceTransformer] | None = None
_reranker_model_bag: dict[str, CrossEncoder] | None = None

logger = logging.getLogger(__name__)


def grab_best_device() -> str:
    best_device = "cpu"
    if torch.backends.mps.is_available():
        best_device = "mps"

    if torch.cuda.is_available():
        best_device = "cuda"

    return best_device


def init_sentence_transformers_models() -> None:
    global _embedder_model_bag
    global _reranker_model_bag

    if _embedder_model_bag is None:
        _embedder_model_bag = {}

    if _reranker_model_bag is None:
        _reranker_model_bag = {}


def get_embedder_model(model_name: str | None) -> SentenceTransformer:
    if model_name is None:
        model_name = DEFAULT_EMBEDDING_MODEL

    global _embedder_model_bag
    assert _embedder_model_bag is not None, "_embedder_model_bag is not initialized"

    model = _embedder_model_bag.get(model_name, None)

    if model:
        return model

    device = grab_best_device()

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


def get_reranking_model(model_name: str | None) -> CrossEncoder:
    """
    Get a reranker model from the in-memory model bag.
    If the bag does not has the requested model, create a CrossEncoder model with the given model_name
    If the model_name does not exists or not compatible, a RuntimeError is raised
    """
    if model_name is None:
        model_name = DEFAULT_RERANKER_MODEL

    global _reranker_model_bag
    assert _reranker_model_bag is not None, "get_reranking_model is not initialized"
    model = _reranker_model_bag.get(model_name, None)

    if model:
        return model
    device = grab_best_device()

    if device == "cpu":
        logger.warning("not found GPU, using CPU device for inference")

    try:
        model = CrossEncoder(model_name_or_path=model_name, device=device)
    except Exception as e:
        raise RuntimeError(
            f"something went wrong when loading embedding model {model_name}, {e}"
        )

    _reranker_model_bag[model_name] = model

    return model
