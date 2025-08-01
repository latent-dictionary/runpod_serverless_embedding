import runpod
from typing import Any
from enum import StrEnum
from models import embedding_models, reranking_models
from services import embedding_service, reranking_service

from singleton import init_sentence_transformers_models


class Tasks(StrEnum):
    Embedding = "embedding"
    Reranking = "reranking"


# --- The RunPod Handler ---
def handler(event: dict[str, Any]):
    task_name = event["input"].get("task_name", Tasks.Embedding)
    if task_name == Tasks.Embedding:
        validated_request = embedding_models.OpenAITextEmbeddingRequest.model_validate(
            event["input"]
        )
        try:
            resp = embedding_service.embed_texts(validated_request)
        except Exception as e:
            resp = embedding_models.OpenAITextEmbeddingResponse(
                data=[], model="", error=f"Exception occured: {e}"
            )
    elif task_name == Tasks.Reranking:
        validated_request = reranking_models.RerankingRequest.model_validate(
            event["input"]
        )
        try:
            resp = reranking_service.rerank_texts(validated_request)
        except Exception as e:
            resp = reranking_models.RerankingResponse(scores=[], error=f"Exception {e}")

    else:
        return {"error": f"unknown task_name {task_name}"}

    return resp.model_dump(exclude_none=True)


# Start the Serverless function when the script is run
if __name__ == "__main__":
    init_sentence_transformers_models()

    runpod.serverless.start({"handler": handler})
