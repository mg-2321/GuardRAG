"""
Persistent dense vector index backends used by dense retrieval.

Author: Gayatri Malladi
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector
    return vector / norm


class PersistentVectorIndex:
    """
    Persistent vector index abstraction.

    Backends:
      - ``qdrant_remote`` if Qdrant credentials are configured
      - ``qdrant_local`` for a persistent local Qdrant store
      - ``faiss_hnsw`` if FAISS is installed
      - ``faiss_flat`` if FAISS is installed
      - ``sklearn`` as the portable local fallback

    ``auto`` prefers hosted Qdrant when configured, then FAISS, then falls
    back to ``sklearn``.
    """

    def __init__(
        self,
        *,
        ids: List[str],
        backend: str,
        metric: str,
        index,
        embeddings: np.ndarray | None = None,
    ):
        self.ids = ids
        self.backend = backend
        self.metric = metric
        self.index = index
        self.embeddings = embeddings

    @staticmethod
    def qdrant_remote_config() -> dict | None:
        url = _env_first("GUARDRAG_QDRANT_URL", "QDRANT_URL")
        if not url:
            return None
        config = {
            "url": url,
            "api_key": _env_first("GUARDRAG_QDRANT_API_KEY", "QDRANT_API_KEY"),
        }
        timeout = _env_first("GUARDRAG_QDRANT_TIMEOUT", "QDRANT_TIMEOUT")
        if timeout:
            try:
                config["timeout"] = int(timeout)
            except ValueError:
                pass
        prefix = _env_first("GUARDRAG_QDRANT_PREFIX", "QDRANT_PREFIX")
        if prefix:
            config["prefix"] = prefix
        return config

    @staticmethod
    def resolve_backend(requested: str | None) -> str:
        requested = (requested or "auto").strip().lower()
        if requested == "auto":
            if PersistentVectorIndex.qdrant_remote_config() and _is_available("qdrant_client"):
                return "qdrant_remote"
            if _is_available("faiss"):
                return "faiss_hnsw"
            return "sklearn"
        if requested in {"qdrant_remote", "qdrant_local"} and not _is_available("qdrant_client"):
            raise RuntimeError(
                f"Vector index backend '{requested}' was requested, but qdrant-client is not installed."
            )
        if requested == "qdrant_remote" and not PersistentVectorIndex.qdrant_remote_config():
            raise RuntimeError(
                "Vector index backend 'qdrant_remote' was requested, but no "
                "GUARDRAG_QDRANT_URL/QDRANT_URL is configured."
            )
        if requested in {"faiss_hnsw", "faiss_flat"} and not _is_available("faiss"):
            raise RuntimeError(
                f"Vector index backend '{requested}' was requested, but faiss is not installed."
            )
        if requested == "sklearn" and not _is_available("sklearn"):
            raise RuntimeError(
                "Vector index backend 'sklearn' was requested, but scikit-learn is not installed."
            )
        if requested not in {"qdrant_remote", "qdrant_local", "faiss_hnsw", "faiss_flat", "sklearn"}:
            raise ValueError(
                f"Unknown vector index backend '{requested}'. "
                "Available: auto, qdrant_remote, qdrant_local, faiss_hnsw, faiss_flat, sklearn"
            )
        return requested

    @staticmethod
    def _qdrant_collection_name(index_dir: Path) -> str:
        prefix = _env_first("GUARDRAG_QDRANT_COLLECTION_PREFIX")
        stem = index_dir.name.lower().replace(".", "_").replace("-", "_")
        stem = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem)
        stem = "_".join(part for part in stem.split("_") if part)
        stem = stem[:36] or "guardrag"
        suffix = hashlib.md5(str(index_dir.resolve()).encode("utf-8")).hexdigest()[:12]
        base = f"{stem}_{suffix}"
        if prefix:
            prefix = prefix.lower().replace("-", "_")
            return f"{prefix}_{base}"[:63]
        return base[:63]

    @staticmethod
    def _qdrant_distance(metric: str):
        from qdrant_client.http import models as qdrant_models  # type: ignore

        if metric == "cosine":
            return qdrant_models.Distance.COSINE
        if metric == "dot":
            return qdrant_models.Distance.DOT
        raise ValueError(f"Unsupported Qdrant metric '{metric}'")

    @staticmethod
    def _qdrant_index_marker(index_dir: Path) -> Path:
        return index_dir / "index.qdrant.json"

    @classmethod
    def _qdrant_client_and_collection(
        cls,
        *,
        backend: str,
        index_dir: Path,
        metric: str,
        dimension: int,
        collection_name: str | None = None,
        create_if_missing: bool = False,
    ):
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.http import models as qdrant_models  # type: ignore

        if backend == "qdrant_local":
            storage_path = index_dir / "qdrant_store"
            storage_path.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(storage_path))
        elif backend == "qdrant_remote":
            config = cls.qdrant_remote_config()
            if not config:
                raise RuntimeError(
                    "Qdrant remote backend selected, but no GUARDRAG_QDRANT_URL/QDRANT_URL is configured."
                )
            client = QdrantClient(**config)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported Qdrant backend '{backend}'")

        collection_name = collection_name or cls._qdrant_collection_name(index_dir)
        if create_if_missing:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=int(dimension),
                    distance=cls._qdrant_distance(metric),
                ),
            )
        elif not client.collection_exists(collection_name):
            raise FileNotFoundError(
                f"Qdrant collection '{collection_name}' does not exist for backend '{backend}'."
            )
        return client, collection_name

    @staticmethod
    def _qdrant_points(ids: List[str], embeddings: np.ndarray, *, start_index: int = 0):
        from qdrant_client.http import models as qdrant_models  # type: ignore

        points = []
        for offset, (doc_id, vector) in enumerate(zip(ids, embeddings)):
            points.append(
                qdrant_models.PointStruct(
                    id=int(start_index + offset),
                    vector=vector.tolist(),
                    payload={"doc_id": doc_id},
                )
            )
        return points

    @classmethod
    def _qdrant_search(cls, index_state, query_vector: np.ndarray, top_k: int):
        client = index_state["client"]
        collection_name = index_state["collection_name"]
        query_payload = cls._prepare_query(query_vector, index_state["metric"]).astype(np.float32).tolist()

        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=collection_name,
                query=query_payload,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            points = getattr(response, "points", response)
        else:
            points = client.search(
                collection_name=collection_name,
                query_vector=query_payload,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

        doc_ids = []
        scores = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            point_id = payload.get("doc_id", getattr(point, "id", None))
            if point_id is None:
                continue
            doc_ids.append(str(point_id))
            scores.append(float(getattr(point, "score", 0.0)))
        return doc_ids, scores

    @staticmethod
    def _prepare_embeddings(embeddings: np.ndarray, metric: str) -> np.ndarray:
        arr = np.asarray(embeddings, dtype=np.float32)
        if metric == "cosine":
            return _normalize_rows(arr)
        return arr

    @staticmethod
    def _prepare_query(query_vector: np.ndarray, metric: str) -> np.ndarray:
        vec = np.asarray(query_vector, dtype=np.float32)
        if metric == "cosine":
            return _normalize_vector(vec)
        return vec

    @classmethod
    def exists(cls, index_dir: Path) -> bool:
        metadata_path = index_dir / "metadata.json"
        doc_ids_path = index_dir / "doc_ids.json"
        if not metadata_path.exists() or not doc_ids_path.exists():
            return False
        if (index_dir / "index.pkl").exists() or (index_dir / "index.faiss").exists():
            return True
        if cls._qdrant_index_marker(index_dir).exists():
            return True
        try:
            backend = json.loads(metadata_path.read_text(encoding="utf-8")).get("backend")
        except Exception:
            return False
        return backend in {"qdrant_local", "qdrant_remote"}

    @classmethod
    def build_in_memory(
        cls,
        *,
        ids: Iterable[str],
        embeddings: np.ndarray,
        backend: str | None = None,
        metric: str = "cosine",
    ) -> "PersistentVectorIndex":
        backend = cls.resolve_backend(backend)
        ids = list(ids)
        arr = cls._prepare_embeddings(embeddings, metric)

        if backend == "sklearn":
            from sklearn.neighbors import NearestNeighbors

            index = NearestNeighbors(metric="cosine", algorithm="brute")
            index.fit(arr)
            return cls(ids=ids, backend=backend, metric=metric, index=index, embeddings=arr)

        if backend in {"qdrant_local", "qdrant_remote"}:
            raise RuntimeError(
                f"Backend '{backend}' requires a persistent index directory; use build() or build_or_load()."
            )

        import faiss  # type: ignore

        dim = int(arr.shape[1]) if arr.ndim == 2 else 0
        if backend == "faiss_hnsw":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 64
        elif backend == "faiss_flat":
            index = faiss.IndexFlatIP(dim)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported backend {backend}")
        index.add(arr)
        return cls(ids=ids, backend=backend, metric=metric, index=index, embeddings=None)

    @classmethod
    def build_or_load(
        cls,
        *,
        index_dir: Path,
        ids: Iterable[str],
        embeddings: np.ndarray,
        backend: str | None = None,
        metric: str = "cosine",
    ) -> "PersistentVectorIndex":
        backend = cls.resolve_backend(backend)
        ids = list(ids)
        index_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = index_dir / "metadata.json"
        if metadata_path.exists():
            try:
                return cls.load(index_dir)
            except Exception:
                pass
        return cls.build(
            index_dir=index_dir,
            ids=ids,
            embeddings=embeddings,
            backend=backend,
            metric=metric,
        )

    @classmethod
    def build(
        cls,
        *,
        index_dir: Path,
        ids: List[str],
        embeddings: np.ndarray,
        backend: str,
        metric: str = "cosine",
    ) -> "PersistentVectorIndex":
        backend = cls.resolve_backend(backend)
        arr = cls._prepare_embeddings(embeddings, metric)
        index_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "backend": backend,
            "metric": metric,
            "count": len(ids),
            "dimension": int(arr.shape[1]) if arr.ndim == 2 and len(arr) else 0,
        }
        (index_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (index_dir / "doc_ids.json").write_text(json.dumps(ids), encoding="utf-8")

        if backend == "sklearn":
            from sklearn.neighbors import NearestNeighbors

            np.save(index_dir / "embeddings.npy", arr)
            index = NearestNeighbors(metric="cosine", algorithm="brute")
            index.fit(arr)
            with (index_dir / "index.pkl").open("wb") as f:
                pickle.dump(index, f)
            return cls(ids=ids, backend=backend, metric=metric, index=index, embeddings=arr)

        if backend in {"qdrant_local", "qdrant_remote"}:
            collection_name = cls._qdrant_collection_name(index_dir)
            client, collection_name = cls._qdrant_client_and_collection(
                backend=backend,
                index_dir=index_dir,
                metric=metric,
                dimension=metadata["dimension"],
                collection_name=collection_name,
                create_if_missing=True,
            )
            batch_size = 1024
            for start in range(0, len(ids), batch_size):
                stop = start + batch_size
                client.upsert(
                    collection_name=collection_name,
                    points=cls._qdrant_points(
                        ids[start:stop],
                        arr[start:stop],
                        start_index=start,
                    ),
                    wait=True,
                )
            qdrant_metadata = dict(metadata)
            qdrant_metadata["collection_name"] = collection_name
            qdrant_metadata["qdrant_mode"] = "remote" if backend == "qdrant_remote" else "local"
            (index_dir / "metadata.json").write_text(
                json.dumps(qdrant_metadata, indent=2), encoding="utf-8"
            )
            cls._qdrant_index_marker(index_dir).write_text(
                json.dumps(
                    {
                        "collection_name": collection_name,
                        "backend": backend,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return cls(
                ids=ids,
                backend=backend,
                metric=metric,
                index={
                    "client": client,
                    "collection_name": collection_name,
                    "metric": metric,
                },
                embeddings=None,
            )

        import faiss  # type: ignore

        np.save(index_dir / "embeddings.npy", arr)
        dim = int(arr.shape[1]) if arr.ndim == 2 else 0
        if backend == "faiss_hnsw":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 64
        elif backend == "faiss_flat":
            index = faiss.IndexFlatIP(dim)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported backend {backend}")
        index.add(arr)
        faiss.write_index(index, str(index_dir / "index.faiss"))
        return cls(ids=ids, backend=backend, metric=metric, index=index, embeddings=None)

    @classmethod
    def load(cls, index_dir: Path) -> "PersistentVectorIndex":
        metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        ids = json.loads((index_dir / "doc_ids.json").read_text(encoding="utf-8"))
        backend = metadata["backend"]
        metric = metadata.get("metric", "cosine")
        if backend == "sklearn":
            with (index_dir / "index.pkl").open("rb") as f:
                index = pickle.load(f)
            embeddings = np.load(index_dir / "embeddings.npy")
            return cls(ids=ids, backend=backend, metric=metric, index=index, embeddings=embeddings)

        if backend in {"qdrant_local", "qdrant_remote"}:
            client, collection_name = cls._qdrant_client_and_collection(
                backend=backend,
                index_dir=index_dir,
                metric=metric,
                dimension=int(metadata.get("dimension", 0)),
                collection_name=metadata.get("collection_name"),
                create_if_missing=False,
            )
            return cls(
                ids=ids,
                backend=backend,
                metric=metric,
                index={
                    "client": client,
                    "collection_name": collection_name,
                    "metric": metric,
                },
                embeddings=None,
            )

        import faiss  # type: ignore

        index = faiss.read_index(str(index_dir / "index.faiss"))
        return cls(ids=ids, backend=backend, metric=metric, index=index, embeddings=None)

    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[str], List[float]]:
        if not self.ids:
            return [], []
        k = min(max(int(top_k), 1), len(self.ids))
        q = self._prepare_query(query_vector, self.metric)

        if self.backend == "sklearn":
            distances, indices = self.index.kneighbors(q.reshape(1, -1), n_neighbors=k)
            idxs = indices[0].tolist()
            if self.metric == "cosine":
                scores = (1.0 - distances[0]).tolist()
            else:
                scores = (-distances[0]).tolist()
            return [self.ids[i] for i in idxs], [float(s) for s in scores]

        if self.backend in {"qdrant_local", "qdrant_remote"}:
            return self._qdrant_search(self.index, q, k)

        scores, indices = self.index.search(q.reshape(1, -1).astype(np.float32), k)
        idxs = [int(i) for i in indices[0] if int(i) >= 0]
        vals = [float(scores[0][j]) for j, i in enumerate(indices[0]) if int(i) >= 0]
        return [self.ids[i] for i in idxs], vals
