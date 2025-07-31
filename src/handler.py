import runpod
from typing import Any

# from models import embedding_models
from services import embedding_service

from singleton import init_sentence_transformers_models


# --- The RunPod Handler ---
def handler(event: dict[str, Any]):
    request = event["input"]
    resp = embedding_service.embed_texts(request)
    return resp


# Start the Serverless function when the script is run
if __name__ == "__main__":
    init_sentence_transformers_models()

    runpod.serverless.start({"handler": handler})
