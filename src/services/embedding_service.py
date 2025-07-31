from models import embedding_models


async def embed_texts(
    request: embedding_models.OpenAITextEmbeddingRequest,
) -> embedding_models.OpenAITextEmbeddingResponse: ...
