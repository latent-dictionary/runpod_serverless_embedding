from models import embedding_models


async def embed_texts(request: embedding_models.TextEmbeddingRequest) -> embedding_models.TextEmbeddingResponse:
    ...