"""
Document chunking utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .types import Document


@dataclass
class ChunkerConfig:
    chunk_size: int = 1000
    overlap: int = 200
    separator: str = "\n\n"


class Chunker:
    """Splits documents into overlapping chunks measured in tokens."""

    def __init__(self, config: ChunkerConfig):
        self.config = config

    def chunk(self, document: Document) -> List[Document]:
        text = document.text
        size = max(self.config.chunk_size, 1)
        overlap = max(min(self.config.overlap, size - 1), 0)

        tokens = self._split(text)
        spans = self._compute_spans(tokens, size, overlap)

        chunks: List[Document] = []
        for idx, (start, end) in enumerate(spans):
            chunk_tokens = tokens[start:end]
            chunk_text = self._join(chunk_tokens)
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "_source_id": document.doc_id,
                    "_chunk_index": idx,
                    "_chunk_span": [start, end],
                    "_chunk_size": size,
                    "_chunk_overlap": overlap,
                }
            )
            chunks.append(
                Document(
                    doc_id=f"{document.doc_id}::chunk{idx}",
                    title=document.title,
                    text=chunk_text,
                    metadata=metadata,
                )
            )
        return chunks or [document]

    @staticmethod
    def _split(text: str) -> List[str]:
        return text.split()

    @staticmethod
    def _join(tokens: List[str]) -> str:
        return " ".join(tokens)

    @staticmethod
    def _compute_spans(tokens: List[str], size: int, overlap: int) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        start = 0
        total = len(tokens)

        while start < total:
            end = min(start + size, total)
            spans.append((start, end))
            if end == total:
                break
            start = end - overlap
            if start < 0:
                start = 0
        return spans

