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
    model_name: str = Field(
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
    data: list[OpenAIEmbeddingObject]

    object: str = Field(default="list")
    model: str = Field(description="model name that generated the embeddings")
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)
    error: str | None = Field(default=None)
