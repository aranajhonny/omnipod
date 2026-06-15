"""Dense vector search via sentence-transformers MPS + Qdrant.

Pure 384d cosine similarity search. No BM25.
"""

from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL,
    HYBRID_TOP_K,
    QDRANT_URL,
)

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, device="mps")
    return _model


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )


def hybrid_search(
    client: QdrantClient,
    query: str,
    top_k: int = HYBRID_TOP_K,
    guest_filter: str | None = None,
) -> list[dict]:
    """Dense vector search with optional guest name filter."""
    model = get_model()
    query_vec = model.encode(query, show_progress_bar=False)

    search_filter = None
    if guest_filter:
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="guest",
                    match=models.MatchValue(value=guest_filter),
                )
            ]
        )

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec.tolist(),
        query_filter=search_filter,
        with_payload=True,
        limit=top_k,
    )

    results = []
    for hit in hits.points:
        results.append(
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "guest": hit.payload.get("guest", ""),
                "title": hit.payload.get("title", ""),
                "type": hit.payload.get("type", ""),
                "source_file": hit.payload.get("source_file", ""),
            }
        )
    return results
