"""
Document store abstraction for GuardRAG.

This module follows the component layout suggested by RAG'n'Roll:
documents live in a unified store, regardless of the downstream
retriever.  Stores can be backed by JSONL files, in-memory dictionaries,
or vector databases in future iterations.  For now we support a simple
JSONL-backed store with optional poisoning metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .chunking import Chunker
from .types import Document


class DocumentStore:
    """
    Lightweight document store that can be wrapped by different retrieval
    components.  Designed to mirror the 'DocumentStore' block from the
    RAG'n'Roll blueprint.
    """

    def __init__(self, documents: Iterable[Document]):
        self._documents: Dict[str, Document] = {}
        for doc in documents:
            if doc.doc_id in self._documents:
                raise ValueError(f"Duplicate document id detected: {doc.doc_id}")
            self._documents[doc.doc_id] = doc

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        chunker: Optional[Chunker] = None,
    ) -> "DocumentStore":
        path = Path(path)
        documents: List[Document] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                metadata = data.get("metadata", {}).copy()
                for key, value in data.items():
                    if key not in {"_id", "title", "text", "metadata"}:
                        metadata[key] = value
                doc = Document(
                    doc_id=data["_id"],
                    title=data.get("title", ""),
                    text=data.get("text", ""),
                    metadata=metadata,
                )
                if chunker:
                    documents.extend(chunker.chunk(doc))
                else:
                    documents.append(doc)

        return cls(documents)

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents.values())

    def get(self, doc_id: str) -> Optional[Document]:
        return self._documents.get(doc_id)

    def filter(self, *, poisoned: Optional[bool] = None) -> List[Document]:
        """
        Filter documents by poisoning status.
        """
        if poisoned is None:
            return list(self._documents.values())
        return [doc for doc in self._documents.values() if doc.is_poisoned == poisoned]

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for doc in self._documents.values():
                f.write(json.dumps(doc.to_dict()) + "\n")

