from pydantic import BaseModel, Field
from enum import StrEnum


class RerankingModels(StrEnum):
    QwenRerankerSmall = "Qwen/Qwen3-Reranker-0.6B"


class RerankingRequest(BaseModel):
    query: str = Field(description="the query string")
    documents: list[str] = Field(description="the documents to rank")
    model_name: str = Field(
        default=RerankingModels.QwenRerankerSmall,
        description="huggingface reranking model name to use",
    )


class RerankingResponse(BaseModel):
    scores: list[float] = Field(
        description="logits returned by the model for all (query, document) pairs in the request, ordered the same as the documents list in the request"
    )

    error: str | None = Field(default=None)
