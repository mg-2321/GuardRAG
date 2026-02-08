from __future__ import annotations

from typing import List

from .base import BaseRetriever, RetrieverResult
from .bm25 import BM25Retriever
from .dense import DenseRetriever


class HybridRetriever(BaseRetriever):
    """
    Linear interpolation between BM25 and dense scores.
    """

    name = "hybrid"

    def __init__(self, store, *, alpha: float = 0.5, dense_model: str = "intfloat/e5-large-v2", cache_dir: str = None):
        super().__init__(store)
        self.alpha = alpha
        self.bm25 = BM25Retriever(store)
        self.dense = DenseRetriever(store, model_name=dense_model, cache_dir=cache_dir)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResult]:
        bm25_results = self.bm25.retrieve(query, top_k=top_k)
        dense_results = self.dense.retrieve(query, top_k=top_k)

        scores = {}
        for doc, score in bm25_results:
            scores.setdefault(doc.doc_id, 0.0)
            scores[doc.doc_id] += self.alpha * score

        for doc, score in dense_results:
            scores.setdefault(doc.doc_id, 0.0)
            scores[doc.doc_id] += (1 - self.alpha) * score

        ranked_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
        return self._collect_documents(ranked_ids, [scores[i] for i in ranked_ids])


