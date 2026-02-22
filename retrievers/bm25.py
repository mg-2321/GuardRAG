from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover
    raise ImportError("rank-bm25 is required for BM25Retriever. Install via pip.") from exc

from .base import BaseRetriever, RetrieverResult


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, store, *, tokenizer=None):
        super().__init__(store)
        self._tokenizer = tokenizer or (lambda text: text.lower().split())
        # Tokenize corpus with progress for large corpora
        store_size = len(self.store)
        if store_size > 1000:
            print(f"Building BM25 index for {store_size} documents...", flush=True)
        corpus_tokens = []
        for i, doc in enumerate(self.store):
            corpus_tokens.append(self._tokenizer(f"{doc.title} {doc.text}"))
            if store_size > 1000 and (i + 1) % 1000 == 0:
                print(f"  Tokenized {i + 1}/{store_size} documents...", flush=True)
        if store_size > 1000:
            print("Creating BM25 index...", flush=True)
        self._index = BM25Okapi(corpus_tokens)
        self._ids = [doc.doc_id for doc in self.store]
        if store_size > 1000:
            print("BM25 index built!", flush=True)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResult]:
        query_tokens = self._tokenizer(query)
        scores = self._index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        doc_ids = [self._ids[i] for i in top_indices]
        doc_scores = [scores[i] for i in top_indices]
        return self._collect_documents(doc_ids, doc_scores)


