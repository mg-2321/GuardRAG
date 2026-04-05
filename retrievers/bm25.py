from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi as _ExternalBM25Okapi
except Exception:  # pragma: no cover
    _ExternalBM25Okapi = None

from .base import BaseRetriever, RetrieverResult


class _FallbackBM25Okapi:
    """Minimal BM25 implementation used when rank_bm25 is unavailable/broken."""

    def __init__(self, corpus_tokens, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_freqs = []
        self.doc_len = []
        df = Counter()

        total_len = 0
        for tokens in corpus_tokens:
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            doc_len = len(tokens)
            self.doc_len.append(doc_len)
            total_len += doc_len
            for token in freqs:
                df[token] += 1

        self.avgdl = (total_len / self.corpus_size) if self.corpus_size else 0.0
        self.idf = {
            token: max(0.0, math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5)))
            for token, freq in df.items()
        }

    def get_scores(self, query_tokens):
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if not self.corpus_size:
            return scores
        avgdl = self.avgdl or 1.0
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for idx, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(token, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_len[idx] / avgdl))
                scores[idx] += idf * ((freq * (self.k1 + 1.0)) / denom)
        return scores


BM25Okapi = _ExternalBM25Okapi or _FallbackBM25Okapi


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

