from typing import TypeAlias
from pydantic import BaseModel, Field
from enum import StrEnum


EmbeddingVector: TypeAlias = list[float]


class TextEmbeddingModel(StrEnum):
    QwenEmbedderSmall = "Qwen/Qwen3-Embedding-0.6B"


class OpenAITextEmbeddingRequest(BaseModel):
    input: list[str] | str = Field(
        default_factory=list, description="input texts to embed"
    )
    model: str = Field(
        default=TextEmbeddingModel.QwenEmbedderSmall, description="model name"
    )

    encoding_format: str = Field(default="float")


class OpenAIEmbeddingObject(BaseModel):
    object: str = Field(default="embedding")
    embeddings: list[float]
    index: int = Field(default=0)


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class OpenAITextEmbeddingResponse(BaseModel):
    embeddings: list[EmbeddingVector] = Field(
        description="list of list of float (2D array) as embedding vectors"
    )

    object: str = Field(default="list")
    data: list[OpenAIEmbeddingObject]
    model: str = Field(description="model name that generated the embeddings")
    usage: EmbeddingUsage
