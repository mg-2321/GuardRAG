"""
Hybrid retriever that combines sparse and dense retrieval.

Author: Gayatri Malladi
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import BaseRetriever, RetrieverResult
from .bm25 import BM25Retriever
from .dense import DenseRetriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever: min-max-normalized linear interpolation of BM25 and dense scores.

    BM25 raw scores are in the range ~5–50; dense cosine similarities are in ~0–1.
    Direct linear combination (alpha*bm25 + (1-alpha)*dense) without normalization
    causes BM25 to completely dominate regardless of alpha, making the dense component
    meaningless.  This implementation normalises each retriever's scores to [0, 1]
    independently before combining, so alpha truly controls the trade-off.

    Candidate pool:
        Each sub-retriever is queried for `pool_size` docs (default = max(3*top_k, 50))
        rather than just top_k.  This avoids the edge case where a doc ranked 11th by
        BM25 and 11th by dense would have been a top-5 combined result but never
        appeared in either individual top_k list.

    Combining:
        combined(doc) = alpha * norm_bm25(doc) + (1 - alpha) * norm_dense(doc)
        Missing score (doc not in one retriever's pool) is treated as 0.0.
    """

    name = "hybrid"

    def __init__(
        self,
        store,
        *,
        alpha: float = 0.5,
        dense_model: str = "intfloat/e5-large-v2",
        cache_dir: str = None,
        device: str = None,
        require_gpu: bool = False,
        index_backend: str = "auto",
        pool_size: int = None,
    ):
        """
        Parameters
        ----------
        alpha:       Weight on dense scores (0 = pure BM25, 1 = pure dense).
        dense_model: HuggingFace model id for the dense retriever.
        cache_dir:   Optional embedding cache directory.
        pool_size:   Candidate pool per sub-retriever before combining.
                     Defaults to max(3 * top_k, 50) at query time.
        """
        super().__init__(store)
        self.alpha      = alpha
        self.pool_size  = pool_size
        self.bm25  = BM25Retriever(store)
        self.dense = DenseRetriever(
            store,
            model_name=dense_model,
            cache_dir=cache_dir,
            device=device,
            require_gpu=require_gpu,
            index_backend=index_backend,
        )

    # internal helpers 

    @staticmethod
    def _min_max_normalize(results: List[RetrieverResult]) -> Dict[str, float]:
        """Return {doc_id: normalized_score} with scores in [0, 1]."""
        if not results:
            return {}
        score_map = {doc.doc_id: score for doc, score in results}
        min_s = min(score_map.values())
        max_s = max(score_map.values())
        denom = max_s - min_s
        if denom == 0.0:
            # All scores equal (e.g. single-doc corpus or all-zero BM25) → map to 1.0
            return {k: 1.0 for k in score_map}
        return {k: (v - min_s) / denom for k, v in score_map.items()}

    # public API 

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResult]:
        pool = self.pool_size or max(top_k * 3, 50)

        bm25_results  = self.bm25.retrieve(query,  top_k=pool)
        dense_results = self.dense.retrieve(query, top_k=pool)

        # Normalize each retriever's scores independently to [0, 1]
        bm25_norm  = self._min_max_normalize(bm25_results)
        dense_norm = self._min_max_normalize(dense_results)

        # Union of all candidate doc_ids from both retrievers
        all_ids = set(bm25_norm) | set(dense_norm)

        # Combine: docs absent from one retriever's pool get score 0.0 for that component
        combined: Dict[str, float] = {
            doc_id: (
                self.alpha * bm25_norm.get(doc_id, 0.0)
                + (1.0 - self.alpha) * dense_norm.get(doc_id, 0.0)
            )
            for doc_id in all_ids
        }

        ranked_ids = sorted(combined, key=combined.__getitem__, reverse=True)[:top_k]
        return self._collect_documents(ranked_ids, [combined[i] for i in ranked_ids])
