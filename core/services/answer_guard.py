# -*- coding: utf-8 -*-
# @File: answer_guard.py
# @Author: yaccii
# @Time: 2025-11-23 22:15
# @Description: RAG 一致性自检（轻量二次模型调用，仅返回 ok / reason）
from __future__ import annotations
import json
from typing import List, Tuple, Any


class AnswerGuard:
    def __init__(self, max_snippets: int = 5, max_len: int = 300):
        self._max_snippets = max_snippets
        self._max_len = max_len

    @staticmethod
    def _safe_json(text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {}
        # 尝试截取首尾花括号
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        try:
            return json.loads(text)
        except Exception:
            return {}

    async def check(self, bot: Any, answer: str, snippets: List[str]) -> Tuple[bool, str]:
        if not answer or not snippets:
            return True, ""

        # 构造简要证据
        ss = []
        for s in snippets[: self._max_snippets]:
            t = (s or "").strip().replace("\n", " ")
            if len(t) > self._max_len:
                t = t[: self._max_len] + "…"
            if t:
                ss.append(t)

        if not ss:
            return True, ""

        sys = (
            "你是严谨的核对员。请基于提供的片段，判断答案是否严格建立在片段内容之上；"
            "如果超出、捏造或与片段矛盾，则视为不通过。"
            "仅输出 JSON，不要多余文字。JSON 结构："
            '{"ok": true|false, "reason": "用中文简要说明，不超过40字"}.'
        )
        usr = "【答案】\n" + answer + "\n\n【片段】\n" + "\n".join([f"{i+1}. {x}" for i, x in enumerate(ss)])

        try:
            resp = await bot.chat(
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
                stream=False,
            )
            data = self._safe_json(str(resp or ""))
            ok = bool(data.get("ok", False))
            reason = str(data.get("reason", "")).strip()
            # 若 ok 字段缺失，默认放行
            return (ok if isinstance(ok, bool) else True), reason
        except Exception:
            return True, ""
