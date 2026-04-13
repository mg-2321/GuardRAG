"""
True SPLADE retriever using the masked-LM head aggregation.

Author: Gayatri Malladi

SPLADE (Sparse Lexical And Expansion) encodes text into a sparse vector over
the full vocabulary (~30 k dimensions for BERT-base).  The encoding is:

    v(text) = max_{tokens} log(1 + ReLU(MLM_logit(token)))

Documents are stored as a scipy.sparse.csr_matrix; retrieval is a dot product
between the query sparse vector and the corpus matrix.

Model choices (from https://huggingface.co/naver):
  naver/splade-v2-max                        ← recommended default
  naver/splade-cocondenser-ensembledistil    ← original paper model
  naver/splade-v2-distil

The old implementation used SentenceTransformer.encode(), which produces
dense CLS/mean-pooled embeddings — NOT sparse SPLADE vectors.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import List

import numpy as np

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    _tqdm = None

from .base import BaseRetriever, RetrieverResult


class SPLADERetriever(BaseRetriever):
    """True sparse SPLADE retriever."""

    name = "splade"

    def __init__(
        self,
        store,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        cache_dir: str = None,
        batch_size: int = 32,
        device: str = None,
        require_gpu: bool = False,
    ):
        super().__init__(store)
        self.model_name = model_name
        self.batch_size  = batch_size

        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for SPLADERetriever. "
                "Install via: pip install transformers torch"
            ) from exc

        import torch as _torch
        self._torch = _torch

        if device is None:
            if _torch.cuda.is_available():
                device = "cuda"
            elif require_gpu:
                raise RuntimeError("GPU retrieval was requested, but CUDA is not available.")
            else:
                device = "cpu"
        elif device.startswith("cuda") and not _torch.cuda.is_available():
            raise RuntimeError(
                f"Retriever requested device '{device}', but CUDA is not available."
            )
        self.device = device

        print(f"Loading SPLADE model '{model_name}' on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()

        self._ids = [doc.doc_id for doc in self.store]
        corpus_texts = [f"{doc.title} {doc.text}" for doc in self.store]

        # Cache handling
        cache_path = None
        if cache_dir:
            _cache_dir = Path(cache_dir)
            _cache_dir.mkdir(parents=True, exist_ok=True)
            corpus_hash = hashlib.md5("".join(corpus_texts).encode()).hexdigest()[:16]
            cache_path  = _cache_dir / f"splade_{model_name.replace('/', '_')}_{corpus_hash}.pkl"

        if cache_path and cache_path.exists():
            print(f"  Loading cached SPLADE corpus matrix from {cache_path}")
            with open(cache_path, "rb") as fh:
                self._corpus_matrix = pickle.load(fh)
            print(f"  Loaded corpus matrix {self._corpus_matrix.shape}")
        else:
            self._corpus_matrix = self._encode_corpus(corpus_texts)
            if cache_path:
                with open(cache_path, "wb") as fh:
                    pickle.dump(self._corpus_matrix, fh)
                print(f"  Saved SPLADE corpus matrix to {cache_path}")

    # ── encoding ─────────────────────────────────────────────────────────────

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into SPLADE sparse vectors.

        Returns dense numpy array of shape (len(texts), vocab_size).
        In practice ~95% of values will be 0 after ReLU.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with self._torch.no_grad():
            logits = self.model(**inputs).logits  # (batch, seq_len, vocab_size)

        # SPLADE aggregation: log(1 + ReLU(logit)), max over sequence tokens
        activated  = self._torch.log1p(self._torch.relu(logits))       # (B, L, V)
        mask       = inputs["attention_mask"].unsqueeze(-1).float()     # (B, L, 1)
        sparse_vec = (activated * mask).max(dim=1).values               # (B, V)

        return sparse_vec.cpu().float().numpy()

    def _encode_corpus(self, texts: List[str]):
        """Encode all corpus texts and return a scipy sparse CSR matrix."""
        try:
            import scipy.sparse
        except ImportError as exc:
            raise ImportError(
                "scipy is required for SPLADERetriever. "
                "Install via: pip install scipy"
            ) from exc

        print(f"  Encoding {len(texts)} docs with SPLADE (batch_size={self.batch_size})...")
        batches = []
        batch_indices = range(0, len(texts), self.batch_size)
        _iter = _tqdm(batch_indices, desc='SPLADE encode', unit='batch') if _HAS_TQDM else batch_indices
        for i in _iter:
            batches.append(self._encode_batch(texts[i : i + self.batch_size]))

        dense = np.vstack(batches)          # (n_docs, vocab_size)
        sparse = scipy.sparse.csr_matrix(dense)
        sparsity = 1.0 - sparse.nnz / (dense.shape[0] * dense.shape[1])
        print(f"  Corpus matrix: {dense.shape}  sparsity={sparsity:.1%}")
        return sparse                        # memory-efficient for large corpora

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrieverResult]:
        query_vec = self._encode_batch([query])[0]   # (vocab_size,)
        # Dot product: sparse_matrix (n_docs × V) · dense_vec (V,) → (n_docs,)
        scores    = np.asarray(self._corpus_matrix.dot(query_vec)).ravel()
        top_idx   = np.argsort(scores)[::-1][:top_k]
        doc_ids   = [self._ids[i] for i in top_idx]
        doc_scores = [float(scores[i]) for i in top_idx]
        return self._collect_documents(doc_ids, doc_scores)
