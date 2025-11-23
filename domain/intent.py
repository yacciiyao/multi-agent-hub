# -*- coding: utf-8 -*-
# @File: intent.py
# @Author: yaccii
# @Time: 2025-11-23 21:16
# @Description:
from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class IntentLabel(str, Enum):
    QNA = "qna"                         # 一般问答
    SUMMARIZE = "summarize"             # 摘要/要点提取
    TRANSLATE = "translate"             # 翻译
    CODE = "code"                       # 代码相关
    IMAGE_ANALYZE = "image_analyze"     # 看图识别/分析
    IMAGE_GENERATE = "image_generate"   # 生成图片
    RAG_QA = "rag_qa"                   # 需要检索增强的问答
    SMALL_TALK = "small_talk"           # 闲聊
    OTHER = "other"                     # 其他


class IntentResult(BaseModel):
    label: IntentLabel = IntentLabel.OTHER
    confidence: float = 0.6
    require_rag: bool = False
    require_image_generation: bool = False
    meta: Dict[str, Any] = Field(default_factory=dict)
    # 备注/解释（可用于调试或埋点）
    note: Optional[str] = None