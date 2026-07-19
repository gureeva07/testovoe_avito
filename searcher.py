from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from cleaner import clean_query


def rrf_score(rank: int, k: int = 60) -> float:
    # чем выше позиция, тем больше очков, константа 60 сглаживает разницу между позициями
    return 1.0 / (k + rank)


def search(
    query_text: str,
    client: QdrantClient,
    dense_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    collection_name: str,
    reranker: CrossEncoder = None,
    article_titles: dict = None,
    top_k: int = 10,
    candidates: int = 50,
) -> str:
    query_text = clean_query(query_text)

    # e5-large нужен префикс "query: " для запросов при поиске
    query_with_prefix = "query: " + query_text
    query_dense = dense_model.encode([query_with_prefix], normalize_embeddings=True)[0].tolist()
    query_sparse = list(sparse_model.embed([query_text]))[0]

    # ищу по смыслу
    dense_results = client.query_points(
        collection_name=collection_name,
        query=query_dense,
        using="dense",
        limit=candidates,
        with_payload=True,
    ).points

    # ищу по ключевым словам
    sparse_results = client.query_points(
        collection_name=collection_name,
        query=models.SparseVector(
            indices=query_sparse.indices.tolist(),
            values=query_sparse.values.tolist(),
        ),
        using="sparse",
        limit=candidates,
        with_payload=True,
    ).points

    article_scores: dict[int, float] = {}
    # запоминаю лучший чанк для каждой статьи, нужен если будет reranker
    article_best_text: dict[int, str] = {}

    for rank, hit in enumerate(dense_results):
        art_id = hit.payload["article_id"]
        article_scores[art_id] = article_scores.get(art_id, 0.0) + rrf_score(rank)
        if art_id not in article_best_text:
            article_best_text[art_id] = hit.payload["text"]

    for rank, hit in enumerate(sparse_results):
        art_id = hit.payload["article_id"]
        article_scores[art_id] = article_scores.get(art_id, 0.0) + rrf_score(rank)
        if art_id not in article_best_text:
            article_best_text[art_id] = hit.payload["text"]

    sorted_candidates = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)

    if reranker is not None:
        pairs = []
        for art_id, _ in sorted_candidates:
            title = article_titles.get(art_id, "") if article_titles else ""
            chunk = article_best_text.get(art_id, "")
            # заголовок ставлю первым, он важнее тела
            text = (title + ". " + chunk).strip()
            pairs.append((query_text, text))

        rerank_scores = reranker.predict(pairs)

        reranked = sorted(
            zip([art_id for art_id, _ in sorted_candidates], rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top_articles = [str(art_id) for art_id, _ in reranked[:top_k]]
    else:
        top_articles = [str(art_id) for art_id, _ in sorted_candidates[:top_k]]

    return " ".join(top_articles)
