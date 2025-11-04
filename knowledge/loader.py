# -*- coding: utf-8 -*-
"""
@Author: yaccii
@Date: 2025-10-29 14:53
@Desc:
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader

from infrastructure.logger import logger


def load_documents(file_path: str) -> List:
    """按文件类型加载文本为 Langchain Documents"""

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in [".txt", ".md", ".csv"]:
        docs = TextLoader(str(path), encoding="utf-8").load()

    elif suffix in [".docx"]:
        docs = Docx2txtLoader(str(path)).load()

    elif suffix in [".pdf"]:
        docs = PyPDFLoader(str(path)).load()

    else:
        raise ValueError(f"不支持文件类型: {suffix}")

    logger.info(f"[Loader] {file_path} -> {len(docs)} documents")

    return docs