"""The retrieval rungs. Every rung returns chunk indexes into common.chunks(), best first.

R1 BM25 (bm25s, English stopwords, Snowball stemmer) · R2 dense (bge-small, Qdrant local mode on disk)
R3 reciprocal rank fusion of R1 and R2 · R4 R3's top 20 re-scored by a cross-encoder."""
from functools import cache

from .common import DATA, EMB_MODEL, EMB_REV, RERANK_MODEL, RERANK_REV, chunks

INDEX = DATA / "index"
RRF_K, FUSE_DEPTH, RERANK_DEPTH = 60, 50, 20
BGE_QUERY = "Represent this sentence for searching relevant passages: "  # bge v1.5 query instruction


def _tok(texts):
    import Stemmer
    import bm25s
    return bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)


@cache
def _bm25():
    import bm25s
    r = bm25s.BM25()
    r.index(_tok([c["text"] for c in chunks()]), show_progress=False)
    return r


@cache
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL, revision=EMB_REV, device="cpu")


@cache
def _qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(path=str(INDEX / "qdrant"))


def build_indexes() -> None:
    from qdrant_client import models
    q = _qdrant()
    n = len(chunks())
    if q.collection_exists("chunks") and q.count("chunks").count == n:
        print(f"qdrant: {n} chunks already indexed")
    else:
        vecs = _embedder().encode([c["text"] for c in chunks()], batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=True)
        if q.collection_exists("chunks"):
            q.delete_collection("chunks")
        q.create_collection("chunks", vectors_config=models.VectorParams(size=vecs.shape[1], distance=models.Distance.COSINE))
        q.upload_collection("chunks", vectors=vecs, ids=list(range(n)), wait=True)
        print(f"qdrant: indexed {n} chunks")
    _bm25()
    print(f"bm25: indexed {n} chunks")


def bm25(query: str, k: int) -> list[int]:
    ids, _ = _bm25().retrieve(_tok([query]), k=k, show_progress=False)
    return [int(i) for i in ids[0]]


def dense(query: str, k: int) -> list[int]:
    v = _embedder().encode(BGE_QUERY + query, normalize_embeddings=True)
    return [int(p.id) for p in _qdrant().query_points("chunks", query=v.tolist(), limit=k).points]


def rrf(*rankings: list[int]) -> list[int]:
    score = {}
    for r in rankings:
        for rank, i in enumerate(r, 1):
            score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(score, key=lambda i: (-score[i], i))


def hybrid(query: str, k: int) -> list[int]:
    return rrf(bm25(query, FUSE_DEPTH), dense(query, FUSE_DEPTH))[:k]


@cache
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, revision=RERANK_REV, device="cpu")


def rerank(query: str, k: int) -> list[int]:
    cand = hybrid(query, RERANK_DEPTH)
    s = _reranker().predict([(query, chunks()[i]["text"]) for i in cand])
    return [i for _, i in sorted(zip(s, cand), key=lambda t: (-t[0], t[1]))][:k]


RUNGS = {"R1": bm25, "R2": dense, "R3": hybrid, "R4": rerank}
