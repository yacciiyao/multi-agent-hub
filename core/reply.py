# -*- coding: utf-8 -*-
"""
@Author: yaccii
@Date: 2025-10-29 14:52
@Desc:
"""
from typing import Optional, Any, List

from pydantic import BaseModel


class Reply(BaseModel):
    user_id: int
    session_id: str
    text: Optional[str] = None
    sources: Optional[List[Any]] = []
    stream: Optional[Any] = None
