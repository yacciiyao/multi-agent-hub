# -*- coding: utf-8 -*-
# @File: enums.py
# @Author: yaccii
# @Time: 2025-11-08 16:45
# @Description:
from enum import Enum


class Channel(str, Enum):
    WEB = "web"
    WECHAT = "wechat"
    DINGTALK = "dingtalk"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOLS = "tool"


class AttachmentType(str, Enum):
    """消息附件类型，目前只用 image，后续可以扩展 audio / file 等。"""
    image = "image"