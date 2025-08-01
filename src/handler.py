import runpod
from typing import Any

from models import embedding_models
from services import embedding_service

from singleton import init_sentence_transformers_models


# --- The RunPod Handler ---
def handler(event: dict[str, Any]):
    validated_request = embedding_models.OpenAITextEmbeddingRequest.model_validate(
        event["input"]
    )
    resp = embedding_service.embed_texts(validated_request)
    return resp.model_dump()


# Start the Serverless function when the script is run
if __name__ == "__main__":
    init_sentence_transformers_models(
        load_default_embedder=True, load_default_reranker=True
    )

    runpod.serverless.start({"handler": handler})
