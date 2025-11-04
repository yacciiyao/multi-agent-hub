# -*- coding: utf-8 -*-
"""
@Author: yaccii
@Date: 2025-10-29 14:52
@Desc:
"""
from typing import Dict, List, Dict as TDict, AsyncIterator, Union


class BaseBot:
    """所有模型的统一抽象接口"""

    family: str = "unknown"
    models_info: Dict[str, dict] = {}

    async def chat(
            self,
            messages: List[TDict[str, str]],
            stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """
        子类必须实现：
        - 非流式：返回完整字符串
        - 流式：返回 AsyncIterator[str]（可 async for）
        """
        raise NotImplementedError
