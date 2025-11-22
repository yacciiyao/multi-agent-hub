# -*- coding: utf-8 -*-
# @File: storage_file.py
# @Author: yaccii
# @Time: 2025-11-21 21:29
# @Description:
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Dict, Any

from domain.message import Attachment
from domain.enums import AttachmentType


@runtime_checkable
class FileStorage(Protocol):
    """
    文件存储抽象接口。
    目前只需要保存文件并返回 Attachment，后续如果有删除/获取等再扩展。
    """

    def save_file(
        self,
        user_id: int,
        session_id: str,
        file_bytes: bytes,
        file_name: str,
        mime_type: Optional[str] = None,
    ) -> Attachment:
        ...


class LocalFileStorage:
    """
    本地文件存储实现：
    - 把文件写到本地磁盘
    - 生成可用于前端访问的 URL
    """

    def __init__(self, base_dir: str, public_base_url: str) -> None:
        # 文件根目录，例如 "./uploads"
        self._base_dir = Path(base_dir)
        # 对外访问的 URL 前缀，例如 "/static/uploads"
        self._public_base_url = public_base_url.rstrip("/static/uploads")

        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save_file(
        self,
        user_id: int,
        session_id: str,
        file_bytes: bytes,
        file_name: str,
        mime_type: Optional[str] = None,
    ) -> Attachment:
        if not file_bytes:
            raise ValueError("empty file content")

        # 生成唯一 id
        attachment_id = uuid.uuid4().hex

        # 保留原始扩展名
        _, ext = os.path.splitext(file_name or "")
        ext = ext.lower()

        # 目录结构: base_dir / {user_id} / {session_id} /
        rel_dir = Path(str(user_id)) / session_id
        dir_path = self._base_dir / rel_dir
        dir_path.mkdir(parents=True, exist_ok=True)

        # 文件名: {attachment_id}{ext}
        disk_name = f"{attachment_id}{ext}"
        disk_path = dir_path / disk_name

        with open(disk_path, "wb") as f:
            f.write(file_bytes)

        # URL: {public_base_url}/{user_id}/{session_id}/{attachment_id}{ext}
        rel_url_path = str(rel_dir / disk_name).replace(os.sep, "/")
        public_url = f"{self._public_base_url}/{rel_url_path}"

        meta: Dict[str, Any] = {
            "file_name": file_name,
            "mime_type": mime_type,
            "size_bytes": len(file_bytes),
        }

        return Attachment(
            id=attachment_id,
            type=AttachmentType.image,
            url=public_url,
            mime_type=mime_type,
            file_name=file_name,
            size_bytes=len(file_bytes),
            meta=meta,
        )


# 简单的单例管理，避免到处 new
_file_storage: Optional[FileStorage] = None


def get_file_storage() -> FileStorage:
    """
    获取全局 FileStorage 实例。
    默认从环境变量读取基础配置，也可以根据需要改成从 config_manager 读取。
    """
    global _file_storage
    if _file_storage is not None:
        return _file_storage

    # 可以按需改成从 infrastructure.config_manager 读取
    base_dir = os.getenv("UPLOAD_BASE_DIR", "./data/uploads")
    public_base_url = os.getenv("UPLOAD_PUBLIC_BASE", "/static/uploads")

    _file_storage = LocalFileStorage(base_dir=base_dir, public_base_url=public_base_url)
    return _file_storage
