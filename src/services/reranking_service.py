from models.reranking_models import RerankingRequest, RerankingResponse
from singleton import get_reranking_model


def rerank_texts(request: RerankingRequest) -> RerankingResponse:
    model = get_reranking_model(request.model_name)
    pairs = [(request.query, doc) for doc in request.documents]
    scores = model.predict(sentences=pairs, convert_to_numpy=True).tolist()
    # score is one dimensional array
    return RerankingResponse(scores=scores)
