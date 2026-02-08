"""
Query processing utilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List


class QueryProcessor(ABC):
    """Abstract processor applied to incoming queries."""

    name: str = "base"

    @abstractmethod
    def __call__(self, query: str) -> str:
        ...


class LowercaseProcessor(QueryProcessor):
    name = "lowercase"

    def __call__(self, query: str) -> str:
        return query.lower()


class StripWhitespaceProcessor(QueryProcessor):
    name = "strip_whitespace"

    def __call__(self, query: str) -> str:
        return " ".join(query.split())


PROCESSOR_REGISTRY = {
    LowercaseProcessor.name: LowercaseProcessor,
    StripWhitespaceProcessor.name: StripWhitespaceProcessor,
}


def build_processors(names: Iterable[str]) -> List[QueryProcessor]:
    processors: List[QueryProcessor] = []
    for name in names:
        try:
            processors.append(PROCESSOR_REGISTRY[name]())
        except KeyError as exc:
            raise ValueError(f"Unknown query processor '{name}'. Available: {list(PROCESSOR_REGISTRY)}") from exc
    return processors


def apply_processors(query: str, processors: Iterable[QueryProcessor]) -> str:
    for processor in processors:
        query = processor(query)
    return query


