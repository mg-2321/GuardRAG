from __future__ import annotations

from typing import List
from pathlib import Path
import pickle
import hashlib
import os
import time

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required for DenseRetriever. Install via pip."
    ) from exc

from .base import BaseRetriever, RetrieverResult


def _resolve_device(device: str | None, require_gpu: bool) -> str:
    try:
        import torch
    except ImportError as exc:
        if require_gpu:
            raise RuntimeError(
                "GPU retrieval was requested, but torch is not installed."
            ) from exc
        return "cpu"

    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if require_gpu:
            raise RuntimeError("GPU retrieval was requested, but CUDA is not available.")
        return "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Retriever requested device '{device}', but CUDA is not available."
        )
    return device


class DenseRetriever(BaseRetriever):
    name = "dense"

    @staticmethod
    def _load_cached_embeddings(cache_path: Path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"✅ Loaded {len(embeddings)} cached embeddings")
        return embeddings

    @staticmethod
    def _save_cached_embeddings(cache_path: Path, embeddings) -> None:
        tmp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
        with open(tmp_path, 'wb') as f:
            pickle.dump(embeddings, f)
        os.replace(tmp_path, cache_path)
        print(f"Saved embeddings to cache: {cache_path}")

    @classmethod
    def _wait_for_cache(cls, cache_path: Path, lock_path: Path, poll_s: float = 5.0, timeout_s: float = 4 * 3600.0):
        start = time.time()
        announced = False
        while True:
            if cache_path.exists():
                return cls._load_cached_embeddings(cache_path)
            if not lock_path.exists():
                return None
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for cache build: {cache_path}")
            if not announced:
                print(f"Waiting for embedding cache to be built by another job: {cache_path}")
                announced = True
            time.sleep(poll_s)

    def __init__(
        self,
        store,
        model_name: str = "intfloat/e5-large-v2",
        cache_dir: str = None,
        device: str | None = None,
        require_gpu: bool = False,
    ):
        super().__init__(store)
        self.model_name = model_name
        self.device = _resolve_device(device, require_gpu)
        print(f"Loading dense model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self._ids = [doc.doc_id for doc in self.store]
        
        # Prefix rules:
        #   E5 models (intfloat/e5-*)       : "passage: {text}" for docs, "query: {q}" for queries
        #   BGE models (BAAI/bge-*)         : no prefix for docs, instruction prefix for queries
        #   All others                      : no prefix
        is_e5  = "e5"  in model_name.lower()
        is_bge = "bge" in model_name.lower()
        corpus_texts = [
            f"passage: {doc.title} {doc.text}" if is_e5
            else f"{doc.title} {doc.text}"           # BGE and others: no doc prefix
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
            lock_path = cache_path.with_suffix(cache_path.suffix + '.lock')

            if cache_path.exists():
                self._embeddings = self._load_cached_embeddings(cache_path)
            else:
                while True:
                    try:
                        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    except FileExistsError:
                        waited = self._wait_for_cache(cache_path, lock_path)
                        if waited is not None:
                            self._embeddings = waited
                            break
                        continue

                    try:
                        with os.fdopen(fd, 'w') as lock_file:
                            lock_file.write(f"pid={os.getpid()}\n")
                        print(f"Encoding corpus with {model_name}...")
                        print(f"  This may take 2-5 minutes for large models...")
                        self._embeddings = self.model.encode(
                            corpus_texts,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )
                        self._save_cached_embeddings(cache_path, self._embeddings)
                        break
                    finally:
                        try:
                            lock_path.unlink()
                        except FileNotFoundError:
                            pass
        else:
            print(f"Encoding corpus with {model_name}...")
            print(f"  This may take 2-5 minutes for large models...")
            self._embeddings = self.model.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=False)
        
        self._is_e5  = is_e5
        self._is_bge = is_bge

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResult]:
        # E5:  "query: {q}"
        # BGE: "Represent this sentence for searching relevant passages: {q}"
        # Others: no prefix
        if self._is_e5:
            query_text = f"query: {query}"
        elif self._is_bge:
            query_text = f"Represent this sentence for searching relevant passages: {query}"
        else:
            query_text = query
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = np.dot(self._embeddings, query_embedding) / np.clip(norms, 1e-10, None)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        doc_ids = [self._ids[i] for i in top_indices]
        scores = [similarities[i] for i in top_indices]
        return self._collect_documents(doc_ids, scores)
