# -*- coding: utf-8 -*-
# @File: rag_service.py
# @Author: yaccii
# @Time: 2025-11-10 08:57
# @Description: RAG 文档上传 / 检索 / 删除（依赖向量库：FAISS / Milvus）
from __future__ import annotations

import math
import uuid
from typing import Optional, List, Dict, Any

from fastapi import UploadFile

from infrastructure.config_manager import config
from infrastructure.mlogger import mlogger
from infrastructure.vector_store_manager import get_vector_store
from rag.embeddings import Embeddings
from rag.loader import (
    load_text_from_upload_file,
    load_text_from_url,
    load_text_from_file_path,
)
from rag.splitter import split_text
from storage.vector_store_base import VectorSearchResult


class RagService:
    """
    - 检索阶段：TopN 候选（可配）→ MMR 去重（无额外依赖）→（可选）交叉编码精排
    - 分数规约、最小阈值过滤、更稳健的 snippet 生成
    """
    def __init__(self) -> None:
        self._config: Dict[str, Any] = config.as_dict()
        self._embedding = Embeddings()
        self._reranker = self._init_reranker()

    # ---------------- Ingest ----------------

    async def ingest_from_url(
        self,
        user_id: int,
        url: str,
        title: Optional[str],
        tags: Optional[List[str]],
        scope: str,
    ) -> str:
        got_title, content = await load_text_from_url(url)
        final_title = (title or got_title or url or "Untitled")[:255]
        return await self._ingest_raw_text(
            user_id=user_id,
            url=url,
            title=final_title,
            tags=(tags or []),
            scope=scope,
            content=content,
        )

    async def ingest_from_file(
        self,
        user_id: int,
        file: UploadFile,
        title: Optional[str],
        tags: Optional[List[str]],
        scope: str,
    ) -> str:
        """
        从上传文件读取内容并写入向量库。
        """
        got_title, content = await load_text_from_upload_file(file)
        final_title = (title or got_title or file.filename or "Untitled")[:255]
        return await self._ingest_raw_text(
            user_id=user_id,
            url=None,
            title=final_title,
            tags=(tags or []),
            scope=scope,
            content=content,
        )

    async def ingest_from_file_path(
        self,
        user_id: int,
        file_path: str,
        title: Optional[str],
        tags: Optional[List[str]],
        scope: str,
    ) -> str:
        got_title, content = load_text_from_file_path(file_path)
        final_title = (title or got_title or "Untitled")[:255]
        return await self._ingest_raw_text(
            user_id=user_id,
            url=None,
            title=final_title,
            tags=(tags or []),
            scope=scope,
            content=content,
        )

    async def _ingest_raw_text(
        self,
        user_id: int,
        title: str,
        content: str,
        url: Optional[str] = None,
        tags: Optional[List[str]] = None,
        scope: str = "global",
    ) -> str:
        title = (title or "").strip()
        content = (content or "").strip()
        if not title or not content:
            raise ValueError("[RagService] Title or content must be provided")

        # 1) 文本切分
        split_config = self._config.get("rag_split") or {}
        parts: List[str] = split_text(
            content,
            target_tokens=int(split_config.get("target_tokens", 400)),
            max_tokens=int(split_config.get("max_tokens", 800)),
            sentence_overlap=int(split_config.get("sentence_overlap", 2)),
        )
        if not parts:
            raise ValueError("[RagService] Content must be provided after splitting")

        # 2) Embedding
        vectors: List[List[float]] = await self._embedding.encode(texts=parts)
        if not vectors:
            raise RuntimeError("[RagService] Embedding result is empty")

        dim = len(vectors[0])
        embed_cfg = self._config.get("embedding", {}) or {}
        expect_dim = int(embed_cfg.get("dim") or 0)
        if expect_dim and dim != expect_dim:
            mlogger.warning(
                self.__class__.__name__,
                "_ingest_raw_text",
                msg="dim mismatch",
                config=embed_cfg,
                got=dim,
            )

        doc_id = str(uuid.uuid4())

        store = get_vector_store()
        if store is None:
            mlogger.warning(
                self.__class__.__name__,
                "vector store not available",
                msg="skip RAG ingest",
                doc_id=doc_id,
            )
            return doc_id

        try:
            store.upsert_document(
                doc_id=doc_id,
                user_id=user_id,
                title=title[:255],
                url=url,
                scope=scope,
                tags=tags or [],
                chunks=parts,
                embeddings=vectors,
            )
        except Exception as e:
            mlogger.exception(
                self.__class__.__name__,
                "upsert_document",
                user_id=user_id,
                doc_id=doc_id,
                msg=e,
            )
            raise

        mlogger.info(self.__class__.__name__, "upsert success", user_id=user_id, doc_id=doc_id)
        return doc_id

    async def list_documents(self, user_id: int) -> List[Dict[str, Any]]:
        """ 暂不实现 """
        return []

    async def delete_document(self, user_id: int, doc_id: str) -> None:
        doc_id = (doc_id or "").strip()
        if not doc_id:
            return

        store = get_vector_store()
        if store is None:
            mlogger.warning(
                self.__class__.__name__,
                "delete_document",
                msg="vector store is None",
                user_id=user_id,
                doc_id=doc_id,
            )
            return

        try:
            store.delete_document(doc_id)
            mlogger.info(self.__class__.__name__, "delete success", user_id=user_id, doc_id=doc_id)
        except Exception as e:
            mlogger.exception(self.__class__.__name__, "delete_document", msg=e, user_id=user_id, doc_id=doc_id)
            raise

    # ---------------- Search（含 MMR + 可选精排） ----------------

    @staticmethod
    def _mk_snippet(s: str, limit: int = 200) -> str:
        t = " ".join((s or "").split())
        return t if len(t) <= limit else (t[:limit] + "…")

    @staticmethod
    def _score_to_unit(v: Optional[float]) -> float:
        """
        把不同实现下的相似度/分数规约到 [0,1]：
        - 若 <=1 认为已是相似度，截断至 [0,1]
        - 若 >1 认为是百分数（0~100），先归一化到 [0,1]
        - None 视为 0
        """
        if v is None:
            return 0.0
        try:
            v = float(v)
        except Exception:
            return 0.0
        if v <= 1.0:
            return max(0.0, min(1.0, v))
        return max(0.0, min(1.0, v / 100.0))

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return 0.0 if (na == 0.0 or nb == 0.0) else (dot / (na * nb))

    def _init_reranker(self):
        try:
            rag_cfg = (self._config.get("rag") or {})
            enable = bool(rag_cfg.get("enable_rerank", False))  # 默认关闭，避免引新依赖
            if not enable:
                return None
            rr = rag_cfg.get("reranker") or {}
            model = rr.get("model") or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            device = rr.get("device") or "cpu"
            from sentence_transformers import CrossEncoder  # 延迟导入
            enc = CrossEncoder(model, device=device)
            mlogger.info(self.__class__.__name__, "reranker loaded", model=model, device=device)
            return enc
        except Exception as e:
            mlogger.warning(self.__class__.__name__, "init_reranker", msg=str(e))
            return None

    def _mmr(self, query_emb: List[float], items: List[Dict[str, Any]], k: int, lambda_: float = 0.7) -> List[Dict[str, Any]]:
        """
        在候选 items（每个包含 'emb'）上执行 MMR，输出多样化的 Top-k。
        items: [{..., 'emb': [float,...], 'score': 0~1}, ...]
        """
        selected: List[Dict[str, Any]] = []
        candidates = items[:]
        while candidates and len(selected) < k:
            best, best_val = None, -1e9
            for c in candidates:
                rel = float(c.get("score", 0.0))
                div = 0.0
                for s in selected:
                    div = max(div, self._cosine(c["emb"], s["emb"]))
                val = lambda_ * rel - (1.0 - lambda_) * div
                if val > best_val:
                    best, best_val = c, val
            selected.append(best)
            candidates.remove(best)
        return selected

    def _rerank(self, query: str, items: List[Dict[str, Any]], final_k: int) -> List[Dict[str, Any]]:
        if not self._reranker or not items:
            return items[:final_k]
        try:
            pairs = [(query, it["content"]) for it in items]
            scores = self._reranker.predict(pairs)
            for it, sc in zip(items, scores):
                it["score_rerank"] = float(sc)
            items.sort(key=lambda x: x.get("score_rerank", 0.0), reverse=True)
            return items[:final_k]
        except Exception as e:
            mlogger.warning(self.__class__.__name__, "rerank", msg=str(e))
            return items[:final_k]

    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        store = get_vector_store()
        if store is None:
            mlogger.warning(self.__class__.__name__, "semantic_search", msg="store is None", query=query)
            return []

        # 1) Query -> Embedding
        qv = await self._embedding.encode([query])
        if not qv or not qv[0]:
            return []
        query_emb: List[float] = qv[0]

        rag_cfg = self._config.get("rag") or {}
        default_top_k = int(rag_cfg.get("top_k", 5) or 5)
        K = int(top_k or default_top_k or 5)

        # 候选规模（TopN）
        N = int(rag_cfg.get("candidate_k", max(10, K * 8)))
        try:
            raw_results: List[VectorSearchResult] = store.search(
                query_embedding=query_emb,
                top_k=N,
            )
        except Exception as e:
            mlogger.exception(self.__class__.__name__, "semantic_search", msg=e, query=query)
            return []

        if not raw_results:
            return []

        # 2) 构建候选 + 批量编码内容用于 MMR
        contents: List[str] = []
        items: List[Dict[str, Any]] = []
        for r in raw_results:
            score_unit = self._score_to_unit(r.score if hasattr(r, "score") else None)
            text = getattr(r, "content", None) or ""
            contents.append(text)
            items.append({
                "doc_id": getattr(r, "doc_id", ""),
                "chunk_index": getattr(r, "chunk_index", 0),
                "title": getattr(r, "title", None),
                "url": getattr(r, "url", None),
                "content": text,
                "score": score_unit,
            })

        try:
            embs = await self._embedding.encode(contents) if contents else []
        except Exception as e:
            mlogger.warning(self.__class__.__name__, "encode_candidates", msg=str(e))
            embs = [[] for _ in contents]

        for it, emb in zip(items, embs):
            it["emb"] = emb or []

        # 3) MMR 多样化选择（TopM）
        M = max(K * 2, min(len(items), 12))  # 控制计算量，足够体现思路
        lambda_ = float(rag_cfg.get("mmr_lambda", 0.7))
        mmr_selected = self._mmr(query_emb, items, k=M, lambda_=lambda_)

        # 4) 可选交叉编码精排 → TopK
        final_items = self._rerank(query, mmr_selected, final_k=K)

        # 5) 最小阈值过滤并输出（与原结构一致）
        min_score = float(rag_cfg.get("min_score", 0.0))
        out: List[Dict[str, Any]] = []
        for it in final_items:
            if float(it.get("score", 0.0)) < min_score:
                continue
            out.append(
                {
                    "title": it["title"],
                    "url": it["url"],
                    "snippet": self._mk_snippet(it["content"]),
                    "score": int(round(it.get("score", 0.0) * 100)),
                    "meta": {
                        "doc_id": it["doc_id"],
                        "chunk_index": it["chunk_index"],
                    },
                    "content": it["content"],
                }
            )
        return out
