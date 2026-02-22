from .base import BaseRetriever  # noqa: F401
from .dense import DenseRetriever  # noqa: F401
from .hybrid import HybridRetriever  # noqa: F401
from .splade import SPLADERetriever  # noqa: F401

RETRIEVER_REGISTRY = {
    DenseRetriever.name: DenseRetriever,
    HybridRetriever.name: HybridRetriever,
    SPLADERetriever.name: SPLADERetriever,
}

# Optional BM25 support (rank-bm25 may not be installed in all environments)
try:
    from .bm25 import BM25Retriever  # noqa: F401
except Exception:
    BM25Retriever = None  # type: ignore[assignment]
else:
    RETRIEVER_REGISTRY[BM25Retriever.name] = BM25Retriever


def get_retriever(name: str):
    try:
        return RETRIEVER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown retriever {name}. Available: {list(RETRIEVER_REGISTRY)}") from exc

