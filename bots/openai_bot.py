# -*- coding: utf-8 -*-
"""
@Author: yaccii
@Date: 2025-10-29 14:52
@Desc: OpenAI 聊天接口
"""
from typing import Optional, List, Dict, AsyncIterator, Union
from openai import AsyncOpenAI
from infrastructure.config_manager import conf
from infrastructure.logger import logger
from bots.base_bot import BaseBot


class OpenAIBot(BaseBot):
    family = "openai"
    models_info = {
        "gpt-3.5-turbo": {"desc": "经典稳定版，适合常规任务"},
        "gpt-4o-mini": {"desc": "轻量快速版 GPT-4"},
        "gpt-4o": {"desc": "旗舰多模态模型"},
    }

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        super().__init__()
        cfg = conf().as_dict()

        api_key = cfg.get("openai_api_key")
        if not api_key:
            raise RuntimeError("OpenAI API key not found.")

        base_url = cfg.get("openai_base_url", "https://api.openai.com/v1")
        self.model_name = model_name or cfg.get("openai_default_model", "gpt-3.5-turbo")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _chat_non_stream(self, messages: List[Dict[str, str]]) -> str:
        """非流式调用"""

        resp = await self.client.responses.create(
            model=self.model_name,
            input=messages,
        )

        text = getattr(resp, "output_text", None)
        if text is not None:
            return text

        try:
            chunks = []
            for out in getattr(resp, "output", []) or []:
                if getattr(out, "type", "") == "output_text":
                    chunks.append(getattr(out, "content", "") or "")
            return "".join(chunks) if chunks else ""
        except Exception:
            return ""

    async def _chat_stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """
        流式调用 Responses API,
        将 token 拼接成完整片段后再 yield, 避免逐字输出导致前端格式混乱。
        """
        logger.info(f"[OpenAIBot] 调用模型 {self.model_name}, stream=True")

        stream = self.client.responses.stream(
            model=self.model_name,
            input=messages,
        )

        async def generator() -> AsyncIterator[str]:
            buffer = ""
            async with stream as s:
                async for event in s:
                    et = getattr(event, "type", "")
                    if et == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if not delta:
                            continue

                        buffer += delta

                        # 每当遇到换行或标点时，输出一次
                        if any(buffer.endswith(x) for x in [".", "!", "?", "\n", "。", "！", "？"]):
                            yield buffer
                            buffer = ""

                    elif et == "response.error":
                        err = getattr(event, "error", None)
                        msg = getattr(err, "message", "") if err else "unknown error"
                        logger.error(f"[OpenAIBot] stream error: {msg}")
                        yield f"[ERROR] {msg}"
                        return

                # 输出残余内容（最后一段）
                if buffer.strip():
                    yield buffer.strip()

            # 正常结束
            yield "[DONE]"

        return generator()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        logger.info(f"[OpenAIBot] 调用模型 {self.model_name}, stream={stream}")
        if stream:
            return await self._chat_stream(messages)
        return await self._chat_non_stream(messages)

    async def healthcheck(self) -> bool:
        try:
            _ = await self.client.models.list()
            return True
        except Exception as e:
            logger.warning(f"[OpenAIBot] healthcheck 失败: {e}")
            return False
