from typing import TypeAlias
from pydantic import BaseModel, Field
from enum import StrEnum


EmbeddingVector: TypeAlias = list[float]

class TextEmbeddingModel(StrEnum):
    QwenEmbedderSmall = "Qwen/Qwen3-Embedding-0.6B"

class TextEmbeddingRequest(BaseModel):
    model_name: TextEmbeddingModel = Field(default=TextEmbeddingModel.QwenEmbedderSmall, description="huggingface model to run embed")
    texts: list[str] = Field(description="text to embed")


class TextEmbeddingResponse(BaseModel):
    embeddings: list[EmbeddingVector] = Field(description="list of list of float (2D array) as embedding vectors")
