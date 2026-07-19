import pandas as pd
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from cleaner import clean_html
from indexer import build_index
from searcher import search
from evaluate import compute_map_at_10

articles = pd.read_feather("candidate_data/articles.f")
calibration = pd.read_feather("candidate_data/calibration.f")
test = pd.read_feather("candidate_data/test.f")
print(f"Статей: {len(articles)}, калибровка: {len(calibration)}, тест: {len(test)}")

print(articles.head(3))
print(calibration.head(3))
print(test.head(3))

articles["clean_body"] = articles["body"].apply(clean_html)
articles["clean_title"] = articles["title"].fillna("").apply(clean_html)

# заголовок дублирую дважды чтобы при поиске он весил больше чем тело
articles["full_text"] = articles["clean_title"] + ". " + articles["clean_title"] + ". " + articles["clean_body"]

print("смотрю что получилось после очистки:")
print(articles.loc[0, "clean_title"])
print(articles.loc[0, "clean_body"][:300])

# нужен чтобы передавать заголовок статьи в reranker
article_titles = dict(zip(articles["article_id"], articles["clean_title"]))

# e5-large хорошо работает на русском и весит ~560MB, на нашем датасете дала лучший результат
dense_model = SentenceTransformer("intfloat/multilingual-e5-large")
sparse_model = SparseTextEmbedding("Qdrant/bm25")

# reranker пока отключен, на cpu 500 запросов считались бы около 1.5 часа
reranker = None

VECTOR_SIZE = dense_model.get_sentence_embedding_dimension()
COLLECTION_NAME = "articles"

client = QdrantClient("qdrant", port=6333)

existing = [c.name for c in client.get_collections().collections]

if COLLECTION_NAME not in existing:
    print("коллекции нет, строю индекс")
    build_index(
        articles=articles,
        client=client,
        dense_model=dense_model,
        sparse_model=sparse_model,
        collection_name=COLLECTION_NAME,
        vector_size=VECTOR_SIZE,
    )
else:
    print("коллекция уже есть, пропускаю")

print("проверяю качество на всех 500 запросах из calibration")

# оборачиваю в функцию чтобы удобно передавать через apply
def search_fn(query_text):
    return search(query_text, client, dense_model, sparse_model, COLLECTION_NAME, reranker, article_titles, candidates=30)

map_score = compute_map_at_10(calibration, search_fn)
print(f"MAP@10 = {map_score:.4f}")

test["answer"] = test["query_text"].apply(search_fn)
test[["query_id", "answer"]].to_csv("answer.csv", index=False)
print("сохранила answer.csv")
print(test[["query_id", "answer"]].head())
