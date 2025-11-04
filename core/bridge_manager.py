# -*- coding: utf-8 -*-
"""
@Author: yaccii
@Date: 2025-10-29 14:51
@Desc: 模型桥接与对话调度
"""
import traceback
from typing import AsyncIterator

from core.dialogue_service import dialog_service
from core.message import Message
from core.reply import Reply
from core.model_registry import get_model_registry
from core.rag_engine import rag_engine
from infrastructure.logger import logger


class BridgeManager:
    """请求调度中心：统一调度普通对话与知识库问答"""

    def __init__(self):
        self.models = get_model_registry()

    async def handle_message(
            self,
            query: str,
            session_id: str,
            user_id: int,
            use_kg: bool = False,
            source: str = "web",
            stream: bool = False,
    ) -> Reply:
        logger.info(
            f"[BridgeManager] query={query[:40]}, session={session_id}, use_kg={use_kg}"
        )

        session = dialog_service.get_session(user_id, session_id)
        bot = self.models.get_model(session.model_name)

        history = [{"role": m.role, "content": m.content} for m in session.messages]
        messages = history + [{"role": "user", "content": query}]

        try:
            dialog_service.append_message(
                user_id=user_id,
                session_id=session_id,
                message=Message(
                    role="user", content=query, use_kg=use_kg, source=source
                ),
            )

            # 知识库模式
            if use_kg:
                rag_result = rag_engine.query(
                    query=query, namespace=session.namespace, model_name=session.model_name
                )
                answer = rag_result.get("answer", "未获取到答案。")
                sources = rag_result.get("sources", [])

                dialog_service.append_message(
                    user_id=user_id,
                    session_id=session_id,
                    message=Message(
                        role="assistant",
                        content=answer,
                        use_kg=use_kg,
                        source=source,
                    ),
                )

                # 自动命名（仅首次）
                if not session.session_name:
                    try:
                        title_prompt = f"请用10个字以内总结以下对话的主题（不要带问号和标点）：{query}"
                        title = await bot.chat(
                            [{"role": "user", "content": title_prompt}], stream=False
                        )
                        title = (title or "").strip()
                        if title:
                            dialog_service.rename_session(
                                user_id=user_id,
                                session_id=session_id,
                                session_name=title,
                            )
                    except Exception as e:
                        logger.warning(f"[Bridge] 自动命名失败: {e}")

                return Reply(
                    user_id=user_id,
                    session_id=session_id,
                    text=answer,
                    sources=sources,
                )

            # 2) 普通模型对话
            if stream:
                # 返回一个异步生成器；同时在结束时把完整回答落库、自动命名
                async def stream_response() -> AsyncIterator[str]:
                    buffer: list[str] = []
                    try:
                        async for chunk in (await bot.chat(messages, stream=True)):
                            buffer.append(chunk)
                            yield chunk
                    except Exception as e:
                        logger.error(f"[Bridge] 流式异常: {e}\n{traceback.format_exc()}")
                        # yield 一段错误提示，避免前端卡死
                        yield "\n[流式传输中断]"
                    finally:
                        full = "".join(buffer)

                        dialog_service.append_message(
                            user_id=user_id,
                            session_id=session_id,
                            message=Message(
                                role="assistant",
                                content=full,
                                use_kg=use_kg,
                                source=source,
                            ),
                        )

                        if not session.session_name:
                            try:
                                title_prompt = f"请用10个字以内总结以下对话的主题（不要带问号和标点）：{query}"
                                title = await bot.chat(
                                    [{"role": "user", "content": title_prompt}],
                                    stream=False,
                                )
                                title = (title or "").strip()
                                if title:
                                    dialog_service.rename_session(
                                        user_id=user_id,
                                        session_id=session_id,
                                        session_name=title,
                                    )
                            except Exception as e:
                                logger.warning(f"[Bridge] 自动命名失败: {e}")

                return Reply(
                    user_id=user_id,
                    session_id=session_id,
                    text=None,
                    sources=[],
                    stream=stream_response(),
                )

            # 非流式
            answer = await bot.chat(messages, stream=False)
            sources = []

            dialog_service.append_message(
                user_id=user_id,
                session_id=session_id,
                message=Message(
                    role="assistant", content=answer, use_kg=use_kg, source=source
                ),
            )

            if not session.session_name:
                try:
                    title_prompt = f"请用10个字以内总结以下对话的主题（不要带问号和标点）：{query}"
                    title = await bot.chat(
                        [{"role": "user", "content": title_prompt}], stream=False
                    )
                    title = (title or "").strip()
                    if title:
                        dialog_service.rename_session(
                            user_id=user_id,
                            session_id=session_id,
                            session_name=title,
                        )
                except Exception as e:
                    logger.warning(f"[Bridge] 自动命名失败: {e}")

            return Reply(
                user_id=user_id,
                session_id=session_id,
                text=answer,
                sources=sources,
            )

        except Exception as e:
            logger.error(f"[Bridge] 异常: {e}\n{traceback.format_exc()}")
            return Reply(
                user_id=user_id,
                session_id=session_id,
                text=f"查询失败: {str(e)}",
                sources=[],
            )


bridge = BridgeManager()
