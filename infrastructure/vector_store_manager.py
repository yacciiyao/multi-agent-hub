# -*- coding: utf-8 -*-
# @File: vector_store_manager.py
# @Author: yaccii
# @Time: 2025-11-14 10:00
# @Description:
from typing import Optional

from openai.types import VectorStore

from infrastructure.config_manager import config
from infrastructure.mlogger import mlogger
from storage.faiss_vector_store import IVectorStore, FaissVectorStore


class VectorStoreManager:

    _instance: Optional["VectorStoreManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.store: Optional[IVectorStore] = None
        self.initialized: bool = False

    async def init(self) -> None:
        if self.initialized:
            return

        _config = config.as_dict()
        vs_config = _config["vector_store"] or {}

        enabled = vs_config.get("enabled", False)
        if not enabled:
            self.store = None
            self.initialized = True
            return

        vs_type = vs_config.get("type", "faiss")

        dim = int(vs_config.get("dim", 1536))
        root_path = vs_config.get("root_path", "data/vector_store")
        mlogger.info(f"[VectorStoreManager] Initializing vector store with type {vs_type}")

        store = FaissVectorStore(dim, root_path)
        try:
            store.load()
        except FileNotFoundError:
            pass

        self.store = store
        self.initialized = True

    def get(self) -> Optional[VectorStore]:
        return self.store

vector_store_manager = VectorStoreManager()