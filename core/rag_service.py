# -*- coding: utf-8 -*-
# @File: rag_service.py
# @Author: yaccii
# @Time: 2025-11-10 08:57
# @Description:
import time
import uuid
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import UploadFile
from openai.types import VectorStore

from domain.rag import RagDocument, RagChunk
from infrastructure.config_manager import config
from infrastructure.mlogger import mlogger
from infrastructure.vector_store_manager import vector_store_manager
from rag.embeddings import Embeddings
from rag.loader import load_text_from_upload_file, load_text_from_url, load_text_from_file_path
from rag.splitter import split_text
from storage.faiss_vector_store import VectorRecord


class RagService:
    def __init__(self):
        self._config = config.as_dict()
        self._storage = None
        self._embedding = Embeddings()
        self._vector_store: Optional[VectorStore] = None
        self._faiss_index_built_once: bool = False

    @property
    def storage(self):
        if not self._storage:
            from infrastructure.storage_manager import storage_manager
            self._storage = storage_manager.get()
        return self._storage

    def _get_vector_store(self) -> Optional[VectorStore]:
        if self._vector_store is not None:
            return self._vector_store

        store = vector_store_manager.get()
        if store is None:
            mlogger.info("[RagService] vector store is not enabled or not initialized")
            return None

        self._vector_store = store
        return store


    async def ingest_from_url(
            self, *, user_id: int, url: str, title: Optional[str], tags: Optional[List[str]], scope: str
    ) -> str:
        got_title, content = await load_text_from_url(url)
        final_title = (title or got_title or url)[:255]
        return await self._ingest_raw_text(
            user_id=user_id, source="web", url=url,
            title=final_title, tags=(tags or []), scope=scope,
            content=content,
        )

    async def ingest_from_file(
            self, *, user_id: int, file: UploadFile, title: Optional[str], tags: Optional[List[str]], scope: str
    ) -> str:
        got_title, content = await load_text_from_upload_file(file)
        final_title = (title or got_title or file.filename or "Untitled")[:255]
        return await self._ingest_raw_text(
            user_id=user_id, source="upload", url=None,
            title=final_title, tags=(tags or []), scope=scope,
            content=content,
        )

    async def ingest_from_file_path(
            self, *, user_id: int, file_path: str, title: Optional[str], tags: Optional[List[str]], scope: str
    ) -> str:
        got_title, content = load_text_from_file_path(file_path)
        final_title = (title or got_title or "Untitled")[:255]
        return await self._ingest_raw_text(
            user_id=user_id, source="upload", url=None,
            title=final_title, tags=(tags or []), scope=scope,
            content=content,
        )

    async def _ingest_raw_text(
            self,
            *,
            user_id: int,
            title: str,
            content: str,
            source: str = "upload",  # upload | web | sync
            url: Optional[str] = None,
            tags: Optional[List[str]] = None,
            scope: str = "global",
    ) -> str:

        if not title or not content:
            raise ValueError("[RagService] Title or content must be provided")

        split_config = (self._config.get("rag_split") or {})
        parts: List[str] = split_text(
            content,
            target_tokens=int(split_config.get("target_tokens", 400)),
            max_tokens=int(split_config.get("max_tokens", 800)),
            sentence_overlap=int(split_config.get("sentence_overlap", 2)),
        )

        if not parts:
            raise ValueError("[RagService]Content must be provided")

        vectors: List[List[float]] = await self._embedding.encode(texts=parts)
        if not vectors:
            raise RuntimeError("[RagService] Embedding result is empty")

        dim = len(vectors[0])
        now = int(time.time())
        doc_id = str(uuid.uuid4())

        embed_config = self._config.get("embedding", {}) or {}
        doc = RagDocument(
            doc_id=doc_id,
            user_id=user_id,
            title=title[:255],
            source=source,
            url=url,
            tags=tags or [],
            scope=scope,
            is_deleted=0,
            created_at=now,
            updated_at=now,
            embed_provider=embed_config.get("provider") or "openai",
            embed_model=embed_config.get("model") or "text-embedding-3-small",
            embed_dim=dim,
            embed_version=int(embed_config.get("version", 1)),
            split_params={
                "target_tokens": int(embed_config.get("target_tokens", 400)),
                "max_tokens": int(embed_config.get("max_tokens", 800)),
                "sentence_overlap": int(embed_config.get("sentence_overlap", 2)),
            },
            preprocess_flags="strip=1,html=basic"
        )

        chunk: List[RagChunk] = []
        for i, (txt, vector) in enumerate(zip(parts, vectors)):
            chunk.append(RagChunk(
                doc_id=doc_id,
                user_id=user_id,
                chunk_index=i,
                content=txt,
                embedding=vector,
                created_at=now
            ))
        await self.storage.upsert_rag_document(doc, chunk)

        return doc_id

    async def list_documents(self, user_id: int) -> List[Dict[str, Any]]:
        docs: List[RagDocument] = await self.storage.list_rag_documents(user_id=user_id)
        return [d.model_dump() for d in docs]

    async def delete_document(
            self,
            *,
            user_id: int,
            doc_id: str,
    ) -> None:
        await self.storage.delete_rag_document(user_id=user_id, doc_id=doc_id)

    async def _semantic_search_mysql(
            self,
            *,
            query: str,
            top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        qv = await self._embedding.encode([query])
        if not qv or not qv[0]:
            return []

        q = np.asarray(qv[0], dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return []

        q = q / (q_norm + 1e-8)

        scan_limit = int(self._config.get("embeddings", {}).get("max_chunks_scan", 5000))
        candidates: List[Dict[str, Any]] = await self.storage.get_rag_chunks_with_embeddings(
            scan_limit=scan_limit
        )

        want_dim = q.shape[0]
        matrix: List[List[float]] = []
        keep: List[Dict[str, Any]] = []
        for c in candidates:
            vec = c.get("embedding")
            if isinstance(vec, list) and len(vec) == want_dim:
                matrix.append(vec)
                keep.append(c)

        if not matrix:
            return []

        M = np.asarray(matrix, dtype=np.float32)  # (N, dim)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        sims = (M @ q).tolist()  # (N,)

        k = max(1, int(top_k))
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]

        def _mk_snippet(s: str, limit: int = 200) -> str:
            t = " ".join((s or "").split())
            return t if len(t) <= limit else (t[:limit] + "…")

        results: List[Dict[str, Any]] = []
        for i in top_idx:
            c = keep[i]
            sim = float(sims[i])
            results.append({
                "title": c.get("title"),
                "url": c.get("url"),
                "snippet": _mk_snippet(c.get("content") or ""),
                "score": max(0, min(100, round(sim * 100))),
                "meta": {"doc_id": c["doc_id"], "chunk_index": int(c["chunk_index"])},
                "content": c.get("content") or "",
            })
        return results

    async def _ensure_faiss_index_built(self) -> None:
        store = self._get_vector_store()
        if store is None:
            return

        if store.size > 0:
            return

        if self._faiss_index_built_once:
            return

        self._faiss_index_built_once = True

        embed_config = self._config.get("embeddings", {})
        scan_limit = int(embed_config.get("max_chunks_scan", 5000))

        rows: List[Dict[str, Any]] = await self.storage.get_rag_chunks_with_embeddings(
            scan_limit=scan_limit,
            user_id=None
        )

        if not rows:
            return

        texts: List[str] = []
        embeddings: List[Optional[np.ndarray]] = []
        has_emb: List[bool] = []

        for row in rows:
            text = row.get("content") or ""
            texts.append(text)

            _vector = row.get("embedding")
            if isinstance(_vector, list) and len(_vector) > 0:
                embeddings.append(np.asarray(_vector, dtype=np.float32))
                has_emb.append(True)
            else:
                embeddings.append(None)
                has_emb.append(False)

        if not all(has_emb):
            texts_to_embed = [
                t for t, flag in zip(texts, has_emb) if not flag
            ]
            if texts_to_embed:
                mlogger.info(
                    f"[RagService] Found {len(texts_to_embed)} chunks without embeddings; embedding now..."
                )
                new_embs = await self._embedding.encode(texts_to_embed)
                if len(new_embs) != len(texts_to_embed):
                    raise RuntimeError(
                        "Embedding service returned unexpected vector count."
                    )
                it = iter(new_embs)
                for i, flag in enumerate(has_emb):
                    if not flag:
                        embeddings[i] = np.asarray(next(it), dtype=np.float32)

        records: List[VectorRecord] = []
        for row, emb in zip(rows, embeddings):
            if emb is None:
                continue

            doc_id = str(row.get("doc_id") or "")
            chunk_index_val = row.get("chunk_index")

            if not doc_id or chunk_index_val is None:
                continue

            chunk_id = f"{doc_id}:{int(chunk_index_val)}"
            user_id = row.get("user_id")

            metadata: Dict[str, Any] = {
                "text": row.get("content") or "",
                "title": row.get("title"),
                "url": row.get("url"),
                "doc_id": doc_id,
                "user_id": user_id,
                "chunk_index": int(chunk_index_val),
                "source": "mysql_rag",
            }

            records.append(
                VectorRecord(
                    id=chunk_id,
                    vector=emb,
                    metadata=metadata,
                )
            )

        if not records:
            mlogger.info("[RagService] No valid RAG chunk records found to build Faiss index.")
            return

        store.add_embeddings(records)
        store.persist()

        mlogger.info(f"[RagService] Faiss index has been built, current vector count={store.size}")

    async def _semantic_search_faiss(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        store = self._get_vector_store()
        if store is None:
            raise RuntimeError("Vector store is not available.")

        await self._ensure_faiss_index_built()
        if store.size == 0:
            return []

        qv = await self._embedding.encode([query])
        if not qv or not qv[0]:
            return []

        q = np.asarray(qv[0], dtype=np.float32)

        rag_cfg = self._config.get("rag", {}) or {}
        top_k_raw = int(rag_cfg.get("faiss_top_k_raw", max(10, top_k * 3)))

        candidates = store.search(
            query_vector=q,
            top_k=top_k_raw,
        )
        if not candidates:
            return []

        filtered: List[tuple[VectorRecord, float]] = []
        for rec, sim in candidates:
            filtered.append((rec, sim))

        k = max(1, int(top_k))
        filtered = filtered[:k]

        def _mk_snippet(s: str, limit: int = 200) -> str:
            t = " ".join((s or "").split())
            return t if len(t) <= limit else (t[:limit] + "…")

        results: List[Dict[str, Any]] = []
        for rec, sim in filtered:
            meta = rec.metadata or {}
            score = max(0, min(100, round(sim * 100)))  # 0~1 -> 0~100

            results.append(
                {
                    "title": meta.get("title"),
                    "url": meta.get("url"),
                    "snippet": _mk_snippet(meta.get("text") or ""),
                    "score": score,
                    "meta": {
                        "doc_id": meta.get("doc_id"),
                        "chunk_index": int(meta.get("chunk_index") or 0),
                    },
                    "content": meta.get("text") or "",
                }
            )

        return results

    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        rag_cfg = self._config.get("rag", {}) or {}
        retriever = str(rag_cfg.get("retriever") or "").lower()

        if retriever == "faiss":
            try:
                return await self._semantic_search_faiss(
                    query=query,
                    top_k=top_k,
                )
            except Exception as e:
                mlogger.warning(
                    f"[RagService] Faiss search failed, fallback to MySQL. error={e}"
                )

            return await self._semantic_search_mysql(
                query=query,
                top_k=top_k,
            )

        return []