# -*- coding: utf-8 -*-
# @File: agent_runtime.py
# @Author: yaccii
# @Time: 2025-11-20 14:34
# @Description: 公共版本：只实现默认对话逻辑。私有分支在这个基础上扩展真实 Agent。
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from domain.message import Message, RagSource


class AgentRuntime:
    def __init__(self, agent_config, bot: Any, rag_service: Any, storage: Any):
        self._agent = agent_config
        self._bot = bot
        self._rag = rag_service
        self._storage = storage

    async def run(
        self,
        agent_key: str,
        session: Any,
        message: Message,
        bot: Any,
        context: List[Dict[str, str]],
        rag_sources: Optional[List[RagSource]] = None,
        stream: bool = False,
    ) -> Tuple[str, List[RagSource]]:
        """
        执行一次完整的 Agent 调用（非流式）：

        参数：
        - agent_key: 当前使用的 agent 标识（暂未在逻辑中使用，为后续扩展预留）
        - session: 当前会话对象（如需基于会话做记忆，可从这里取信息）
        - message: 当前用户消息（包含 content / attachments 等）
        - bot: 当前绑定的 Bot 实例（由上游根据 session.bot_name 获取）
        - context: 上游拼好的对话上下文（system + 历史 + RAG 提示）
        - rag_sources: 本轮检索到的 RAG 结果列表，可用于返回给前端
        - stream: 是否流式；当前只支持非流式，若为 True 会直接报错

        返回：
        - reply_text: 模型回复的最终文本
        - final_sources: 用于返回给前端的 RAG 结果列表
        """
        if stream:
            raise RuntimeError("AgentRuntime.run: stream=True 暂未支持，请在 MessageService 中处理流式逻辑。")

        attachments = getattr(message, "attachments", None) or []

        allow_image = bool(getattr(bot, "allow_image", False))

        has_mm_chat = hasattr(bot, "chat_with_attachments")

        use_vision = bool(attachments) and allow_image and has_mm_chat

        if use_vision:
            reply = await bot.chat_with_attachments(
                messages=context,
                attachments=attachments,
                stream=False,
            )
        else:
            reply = await bot.chat(context, stream=False)

        reply_text = str(reply or "").strip()
        final_sources: List[RagSource] = rag_sources or []

        return reply_text, final_sources