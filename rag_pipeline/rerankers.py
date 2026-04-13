"""
Reranking utilities.

Author: Gayatri Malladi
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

from .types import Document

Candidate = Tuple[Document, float]


class BaseReranker(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: Iterable[Candidate]) -> List[Candidate]:
        ...


class CrossEncoderReranker(BaseReranker):
    name = "cross_encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", top_n: int = 10):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("sentence-transformers is required for CrossEncoderReranker.")
        except Exception as exc:
            raise RuntimeError(
                "Failed to import sentence_transformers for CrossEncoderReranker."
            ) from exc
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query: str, candidates: Iterable[Candidate]) -> List[Candidate]:
        candidates = list(candidates)
        if not candidates:
            return []

        pairs = [(query, doc.text) for doc, _ in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for (doc, _), score in zip(candidates, scores):
            reranked.append((doc, float(score)))

        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[: self.top_n]


RERANKER_REGISTRY = {
    CrossEncoderReranker.name: CrossEncoderReranker,
}


def get_reranker(name: str, **kwargs):
    try:
        cls = RERANKER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown reranker '{name}'. Available: {list(RERANKER_REGISTRY)}") from exc
    return cls(**kwargs)
