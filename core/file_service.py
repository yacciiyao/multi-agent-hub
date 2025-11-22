# -*- coding: utf-8 -*-
# @File: file_service.py
# @Author: yaccii
# @Time: 2025-11-21 21:50
# @Description:
from typing import Optional

from domain.message import Attachment
from storage.storage_file import FileStorage, get_file_storage


class FileService:
    def __init__(self, storage: Optional[FileStorage] = None) -> None:
        self._storage = storage or get_file_storage()

    def save_image_file(
        self,
        user_id: int,
        session_id: str,
        file_bytes: bytes,
        file_name: str,
        mime_type: Optional[str] = None,
    ) -> Attachment:
        if not file_bytes:
            raise ValueError("文件内容为空")

        if mime_type is not None and not mime_type.startswith("image/"):
            raise ValueError(f"不支持的文件类型: {mime_type}")

        return self._storage.save_file(
            user_id=user_id,
            session_id=session_id,
            file_bytes=file_bytes,
            file_name=file_name,
            mime_type=mime_type,
        )
