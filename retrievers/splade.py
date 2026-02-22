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
        "sentence-transformers is required for SPLADERetriever. Install via pip."
    ) from exc

from .base import BaseRetriever, RetrieverResult


class SPLADERetriever(BaseRetriever):
    name = "splade"

    def __init__(self, store, model_name: str = "naver/splade-cocondenser-ensembledistil", cache_dir: str = None):
        super().__init__(store)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._ids = [doc.doc_id for doc in self.store]
        
        corpus_texts = [f"{doc.title} {doc.text}" for doc in self.store]
        
        # Try to load cached embeddings
        cache_path = None
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Create cache key from model name and corpus hash
            corpus_hash = hashlib.md5("".join(corpus_texts).encode()).hexdigest()[:16]
            cache_path = cache_dir / f"splade_{model_name.replace('/', '_')}_{corpus_hash}.pkl"
            
            if cache_path.exists():
                print(f"Loading cached SPLADE embeddings from {cache_path}")
                with open(cache_path, 'rb') as f:
                    self._embeddings = pickle.load(f)
                print(f"✅ Loaded {len(self._embeddings)} cached SPLADE embeddings")
            else:
                print(f"Encoding corpus with SPLADE (this may take a few minutes)...")
                self._embeddings = self.model.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=True)
                # Save cache
                with open(cache_path, 'wb') as f:
                    pickle.dump(self._embeddings, f)
                print(f"✅ Saved SPLADE embeddings cache to {cache_path}")
        else:
            print(f"Encoding corpus with SPLADE (this may take a few minutes)...")
            self._embeddings = self.model.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=True)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResult]:
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        # SPLADE uses dot product for sparse vectors
        similarities = np.dot(self._embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        doc_ids = [self._ids[i] for i in top_indices]
        scores = [similarities[i] for i in top_indices]
        return self._collect_documents(doc_ids, scores)

