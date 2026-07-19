import uuid
import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding


def split_into_chunks(text: str, chunk_size: int = 250, overlap: int = 50) -> list[str]:
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # перекрытие 50 слов, чтобы не потерять смысл на границах чанков
        start += chunk_size - overlap

    return chunks


def build_index(
    articles: pd.DataFrame,
    client: QdrantClient,
    dense_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    collection_name: str,
    vector_size: int,
    chunk_size: int = 250,
    overlap: int = 50,
    batch_size: int = 64,
):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            # cosine считает угол между векторами, стандарт для текстового поиска
            "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )

    # сначала собираю все чанки, потом считаю эмбеддинги батчами
    all_points = []

    for _, row in articles.iterrows():
        article_id = int(row["article_id"])
        full_text = row["full_text"]

        chunks = split_into_chunks(full_text, chunk_size=chunk_size, overlap=overlap)

        # если статья пустая, хотя бы заголовок кладу чтобы статья не пропала из индекса
        if not chunks:
            chunks = [row["clean_title"] or f"article {article_id}"]

        for chunk_text in chunks:
            # uuid гарантирует уникальный id для каждой точки
            point_id = str(uuid.uuid4())
            all_points.append({
                "id": point_id,
                "article_id": article_id,
                "text": chunk_text,
            })

    print(f"получилось {len(all_points)} чанков")
    print("считаю эмбеддинги:")

    qdrant_points = []

    for i in range(0, len(all_points), batch_size):
        batch = all_points[i : i + batch_size]
        texts = [p["text"] for p in batch]

        # e5-large требует префикс "passage: " при индексации текстов
        texts_with_prefix = ["passage: " + t for t in texts]
        dense_vecs = dense_model.encode(texts_with_prefix, normalize_embeddings=True)
        sparse_vecs = list(sparse_model.embed(texts))

        for point, dense_vec, sparse_vec in zip(batch, dense_vecs, sparse_vecs):
            qdrant_points.append(
                models.PointStruct(
                    id=point["id"],
                    vector={
                        "dense": dense_vec.tolist(),
                        # sparse хранится как пары индекс-значение, не как полный массив
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "article_id": point["article_id"],
                        "text": point["text"],
                    },
                )
            )

        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch)}/{len(all_points)}")

    for i in range(0, len(qdrant_points), 256):
        client.upsert(
            collection_name=collection_name,
            points=qdrant_points[i : i + 256],
        )

    print(f"залила {len(qdrant_points)} чанков в qdrant")
