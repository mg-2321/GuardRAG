from __future__ import annotations

from typing import List
from pathlib import Path
import pickle
import hashlib

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required for DenseRetriever. Install via pip."
    ) from exc

from .base import BaseRetriever, RetrieverResult


class DenseRetriever(BaseRetriever):
    name = "dense"

    def __init__(self, store, model_name: str = "intfloat/e5-large-v2", cache_dir: str = None):
        super().__init__(store)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._ids = [doc.doc_id for doc in self.store]
        
        # E5 models require "passage: " prefix for documents
        is_e5 = "e5" in model_name.lower()
        corpus_texts = [
            f"passage: {doc.title} {doc.text}" if is_e5 else f"{doc.title} {doc.text}"
            for doc in self.store
        ]
        
        # Try to load cached embeddings
        cache_path = None
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Create cache key from model name and corpus hash
            corpus_hash = hashlib.md5("".join(corpus_texts).encode()).hexdigest()[:16]
            cache_path = cache_dir / f"dense_{model_name.replace('/', '_')}_{corpus_hash}.pkl"
            
            if cache_path.exists():
                print(f"Loading cached embeddings from {cache_path}")
                with open(cache_path, 'rb') as f:
                    self._embeddings = pickle.load(f)
                print(f"✅ Loaded {len(self._embeddings)} cached embeddings")
            else:
                print(f"Encoding corpus with {model_name}...")
                print(f"  This may take 2-5 minutes for large models...")
                self._embeddings = self.model.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=True)
                # Save cache
                with open(cache_path, 'wb') as f:
                    pickle.dump(self._embeddings, f)
                print(f"✅ Saved embeddings to cache: {cache_path}")
        else:
            print(f"Encoding corpus with {model_name}...")
            print(f"  This may take 2-5 minutes for large models...")
            self._embeddings = self.model.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=True)
        
        self._is_e5 = is_e5

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResult]:
        # E5 models require "query: " prefix for queries
        query_text = f"query: {query}" if self._is_e5 else query
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = np.dot(self._embeddings, query_embedding) / np.clip(norms, 1e-10, None)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        doc_ids = [self._ids[i] for i in top_indices]
        scores = [similarities[i] for i in top_indices]
        return self._collect_documents(doc_ids, scores)
