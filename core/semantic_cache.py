# -*- coding: utf-8 -*-
# @File: semantic_cache.py
# @Author: yaccii
# @Time: 2025-11-23 22:14
# @Description: 语义缓存（会话内相似问答命中后直接返回）

from __future__ import annotations
import math
import time
from typing import Dict, List, Any, Optional


class _CacheEntry:
    __slots__ = ("emb", "reply", "sources", "created_at")

    def __init__(self, emb: List[float], reply: str, sources: List[Dict[str, Any]]):
        self.emb = emb
        self.reply = reply
        self.sources = sources or []
        self.created_at = int(time.time())


class SemanticCache:
    def __init__(self):
        self._buckets: Dict[str, List[_CacheEntry]] = {}
        self._default_max = 200

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return 0.0 if (na == 0.0 or nb == 0.0) else (dot / (na * nb))

    def _key(self, user_id: int, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    async def find_similar(
        self,
        user_id: int,
        session_id: str,
        query_emb: List[float],
        threshold: float = 0.92,
    ) -> Optional[Dict[str, Any]]:
        bucket = self._buckets.get(self._key(user_id, session_id)) or []
        if not bucket or not query_emb:
            return None

        best_idx, best_sim = -1, -1.0
        for i, e in enumerate(bucket):
            sim = self._cos(query_emb, e.emb)
            if sim > best_sim:
                best_idx, best_sim = i, sim

        if best_sim >= threshold and best_idx >= 0:
            e = bucket[best_idx]
            return {"reply": e.reply, "sources": e.sources, "similarity": best_sim}
        return None

    async def put(
        self,
        user_id: int,
        session_id: str,
        query_emb: List[float],
        reply: str,
        sources: List[Dict[str, Any]],
        max_entries: Optional[int] = None,
    ) -> None:
        if not query_emb or not reply:
            return
        k = self._key(user_id, session_id)
        bucket = self._buckets.setdefault(k, [])
        bucket.append(_CacheEntry(query_emb, reply, sources or []))

        limit = int(max_entries or self._default_max or 200)
        if len(bucket) > limit:
            # 简单淘汰：按创建时间删除最旧的前若干条
            bucket.sort(key=lambda x: x.created_at, reverse=True)
            del bucket[limit:]


# 单例
semantic_cache = SemanticCache()
