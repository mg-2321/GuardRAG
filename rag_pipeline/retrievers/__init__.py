from .base import BaseRetriever  # noqa: F401
from .bm25 import BM25Retriever  # noqa: F401
from .dense import DenseRetriever  # noqa: F401
from .hybrid import HybridRetriever  # noqa: F401
from .splade import SPLADERetriever  # noqa: F401


RETRIEVER_REGISTRY = {
    BM25Retriever.name: BM25Retriever,
    DenseRetriever.name: DenseRetriever,
    HybridRetriever.name: HybridRetriever,
    SPLADERetriever.name: SPLADERetriever,
}


def get_retriever(name: str):
    try:
        return RETRIEVER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown retriever {name}. Available: {list(RETRIEVER_REGISTRY)}") from exc


