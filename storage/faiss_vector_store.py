# -*- coding: utf-8 -*-
# @File: __init__.py
# @Author: yaccii
# @Time: 2025-11-13 20:28
# @Description:
import json
import os
from dataclasses import dataclass
from os.path import exists
from typing import Dict, Any, Protocol, List, Tuple

import faiss
import numpy as np

from infrastructure.mlogger import mlogger


@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any]

class IVectorStore(Protocol):
    def add_embeddings(self, records: List[VectorRecord]) -> None:
        ...

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[VectorRecord, float]]:
        ...

    def delete(self, ids: List[str]) -> None:
        ...

    def persist(self) -> None:
        ...

    def load(self) -> None:
        ...

    def reset(self) -> None:
        ...

    @property
    def size(self) -> int:
        ...


class FaissVectorStore(IVectorStore):
    def __init__(self, dim: int, root_path: str) -> None:
        if dim <= 0:
            raise ValueError("Vector dimension must be a positive integer.")

        self.dim = dim
        self.root_path = os.path.abspath(root_path)
        os.makedirs(self.root_path, exist_ok=True)

        self._vectors: np.ndarray = np.empty((0, self.dim), dtype=np.float32)
        self._ids: List[str] = []
        self._metas: List[Dict[str, Any]] = []

        self._id2index: Dict[str, int] = {}
        self.index = faiss.IndexFlatIP(self.dim)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector, axis=-1, keepdims=True)
        norm = norm + 1e-8
        return vector / norm

    def _rebuild_index(self) -> None:
        self.index.reset()
        if self._vectors.shape[0] == 0:
            return

        _vectors = self._normalize(self._vectors.astype(np.float32))
        self.index.add(_vectors)  # type: ignore[arg-type]

    def _save_arrays(self) -> None:
        vectors_path = os.path.join(self.root_path, "vectors.npy")
        metadata_path = os.path.join(self.root_path, "metadata.json")

        np.save(vectors_path, self._vectors)

        payload = {
            "ids": self._ids,
            "metadata": self._metas,
            "dim": self.dim,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)

    def _load_arrays(self) -> None:
        vectors_path = os.path.join(self.root_path, "vectors.npy")
        meta_path = os.path.join(self.root_path, "metadata.json")

        if not os.path.isfile(vectors_path) or not os.path.isfile(meta_path):
            mlogger.info(
                "[FaissVectorStore] No existing index files found; start with empty index."
            )
            self._vectors = np.empty((0, self.dim), dtype="float32")
            self._ids = []
            self._metas = []
            self._id_to_index = {}
            self.index.reset()
            return

        try:
            vectors = np.load(vectors_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            mlogger.warning(
                f"[FaissVectorStore] Failed to read index files, "
                f"start with empty index instead. error={e}"
            )
            self._vectors = np.empty((0, self.dim), dtype="float32")
            self._ids = []
            self._metas = []
            self._id_to_index = {}
            self.index.reset()
            return

        ids = meta.get("ids") or []
        metas = meta.get("metadatas") or []

        if vectors.shape[0] != len(ids) or len(ids) != len(metas):
            mlogger.warning(
                "[FaissVectorStore] Vector size and metadata size mismatch; "
                "reset index to empty and rebuild on demand."
            )
            self._vectors = np.empty((0, self.dim), dtype="float32")
            self._ids = []
            self._metas = []
            self._id_to_index = {}
            self.index.reset()
            return

        self._vectors = vectors.astype("float32")
        self._ids = ids
        self._metas = metas
        self._id_to_index = {str(i): idx for idx, i in enumerate(ids)}

        if self._vectors.shape[0] > 0:
            x = self._normalize(self._vectors)
            self.index.reset()
            self.index.add(x) # type: ignore[arg-type]

        mlogger.info(f"[FaissVectorStore] Loaded existing index: count={len(self._ids)}")


    @property
    def size(self) -> int:
        return self._vectors.shape[0]

    def reset(self) -> None:
        self._vectors = np.empty((0, self.dim), dtype=np.float32)
        self._ids = []
        self._metas = []
        self._id2index = {}
        self.index.reset()

    def add_embeddings(self, records: List[VectorRecord]) -> None:
        if not records:
            return

        new_vectors: List[np.ndarray] = []
        new_ids: List[str] = []
        new_metas: List[Dict[str, Any]] = []

        _vectors = self._vectors
        _ids = list(self._ids)
        _metas = list(self._metas)
        _id2index = dict(self._id2index)

        for record in records:
            if record.vector.ndim != 1 or record.vector.shape[0] != self.dim:
                raise ValueError(f"Vector dimension does not match vector size. Excepted {self.dim}, got {record.vector.ndim}.")

            v = record.vector.astype(np.float32)
            if record.id in _id2index:
                idx = _id2index[record.id]
                _vectors[idx] = v
                _metas[idx] = record.metadata or {}

            else:
                new_vectors.append(v)
                new_ids.append(record.id)
                new_metas.append(record.metadata or {})

        if new_vectors:
            to_add = np.stack(new_vectors, axis=0)
            if _vectors.shape[0] == 0:
                _vectors = to_add

            else:
                _vectors = np.concatenate((_vectors, to_add), axis=0)

            _ids.extend(new_ids)
            _metas.extend(new_metas)

        self._vectors = _vectors
        self._metas = _metas
        self._ids = _ids
        self._id2index = {id_: idx for idx, id_ in enumerate(self._ids)}

        self._rebuild_index()

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[VectorRecord, float]]:
        if self.size == 0:
            return []

        if query_vector.ndim != 1 or query_vector.shape[0] != self.dim:
            raise ValueError("Query vector dimension does not match vector size.")

        k = max(1, top_k)

        q = query_vector.astype(np.float32).reshape(1, self.dim)
        q = self._normalize(q)

        distances, indices = self.index.search(q, k) # type: ignore[arg-type]

        results: List[Tuple[VectorRecord, float]] = []
        for pos, score in zip(indices[0], distances[0]):
            if pos < 0 or pos >= self.size:
                continue
            rec = VectorRecord(
                id=self._ids[pos],
                vector=self._vectors[pos],
                metadata=self._metas[pos],
            )

            sim = float(max(0.0, min(1.0, score)))
            results.append((rec, sim))

        return results

    def delete(self, ids: List[str]) -> None:
        if not ids or self.size == 0:
            return

        to_delete = set(ids)
        keep_vectors: List[np.ndarray] = []
        keep_ids: List[str] = []
        keep_metas: List[Dict[str, Any]] = []

        for idx, (id_, v, m) in enumerate(zip(self._ids, self._vectors, self._metas)):
            if id_ in to_delete:
                continue
            keep_ids.append(id_)
            keep_vectors.append(v)
            keep_metas.append(m)

        if keep_vectors:
            self._vectors = np.stack(keep_vectors, axis=0)
        else:
            self._vectors = np.empty((0, self.dim), dtype=np.float32)

        self._ids = keep_ids
        self._metas = keep_metas
        self._id2index = {id_: idx for idx, id_ in enumerate(self._ids)}

    def persist(self) -> None:
        self._save_arrays()

    def load(self) -> None:
        self._load_arrays()
