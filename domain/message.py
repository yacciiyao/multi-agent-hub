# -*- coding: utf-8 -*-
# @File: message.py
# @Author: yaccii
# @Time: 2025-11-08 16:47
# @Description:
import time
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from domain.enums import Role, AttachmentType


class RagSource(BaseModel):
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None
    meta: Dict[str, str] = Field(default_factory=dict)


class Attachment(BaseModel):
    """消息附件（目前主要用于图片，多模态扩展预留）。"""

    id: str
    type: AttachmentType = AttachmentType.image

    # 供前端渲染的访问地址（本地静态路径或 CDN URL）
    url: str

    # 附件的基础元信息
    mime_type: Optional[str] = None
    file_name: Optional[str] = None
    size_bytes: Optional[int] = None

    # 预留一些通用扩展信息，如宽高、hash 等
    meta: Dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    session_id: str
    role: Role
    content: str
    attachments: List[Attachment] = Field(default_factory=list)
    rag_enabled: bool = False
    stream_enabled: bool = False
    sources: List[RagSource] = Field(default_factory=list)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    is_deleted: bool = False

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "role": getattr(self.role, "value", str(self.role)),
            "content": self.content,
            "attachments": [
                (a.model_dump() if hasattr(a, "model_dump") else dict(a))
                for a in (self.attachments or [])
            ],
            "rag_enabled": bool(self.rag_enabled),
            "stream_enabled": bool(self.stream_enabled),
            "sources": [
                (s.model_dump() if hasattr(s, "model_dump") else dict(s))
                for s in (self.sources or [])
            ],
            "created_at": int(self.created_at or 0),
            "is_deleted": bool(self.is_deleted),
        }
