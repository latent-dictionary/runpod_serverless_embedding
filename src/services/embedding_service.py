import numpy as np
from models import embedding_models
from src.singleton import get_embedder_model


def embeddings_to_objects(
    embeddings: np.ndarray,
) -> list[embedding_models.OpenAIEmbeddingObject]:
    resp = []
    for row_index in embeddings.shape[0]:
        embedding_vector = embeddings[row_index, :]
        resp.append(
            embedding_models.OpenAIEmbeddingObject(
                embeddings=embedding_vector.tolist(), index=row_index
            )
        )

    return resp


def embed_texts(
    request: embedding_models.OpenAITextEmbeddingRequest,
) -> embedding_models.OpenAITextEmbeddingResponse:
    model = get_embedder_model(request.model)

    embeddings = model.encode(request.input, convert_to_numpy=True)

    return embedding_models.OpenAITextEmbeddingResponse(
        data=embeddings_to_objects(embeddings),
        model=request.model,
        usage=embedding_models.EmbeddingUsage(
            prompt_tokens=0, total_tokens=0
        ),  # just dummy data
    )
