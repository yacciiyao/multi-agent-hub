# -*- coding: utf-8 -*-
# @File: intent_service.py
# @Author: yaccii
# @Time: 2025-11-23 21:17
# @Description: 用户意图识别服务（启发式 + 可选 LLM 判别）
import asyncio
import re
from typing import Optional, List

from domain.enums import AttachmentType
from domain.intent import IntentResult, IntentLabel
from domain.message import Attachment
from infrastructure.config_manager import config
from infrastructure.mlogger import mlogger


class IntentService:
    """
    轻量级意图识别：
    1) 先用启发式规则，低成本、稳定。
    2) 若配置 intent.use_llm=True，则回退小模型做 JSON 判别（保守解析，避免影响主流程）。
    """

    def __init__(self) -> None:
        _cfg = config.as_dict() or {}
        _intent_cfg = _cfg.get("intent", {}) or {}
        self._enabled: bool = bool(_intent_cfg.get("enabled", True))
        self._use_llm: bool = bool(_intent_cfg.get("use_llm", False))
        self._llm_timeout: float = float(_intent_cfg.get("llm_timeout_sec", 3.0))

    async def detect(
        self,
        content: Optional[str],
        attachments: Optional[List[Attachment]] = None,
        bot: Optional[object] = None,
    ) -> IntentResult:
        if not self._enabled:
            return IntentResult(label=IntentLabel.OTHER, confidence=0.0, note="intent disabled")

        text = (content or "").strip()
        atts = attachments or []

        # ---------- 1) 启发式 ----------
        result = self._heuristic(text, atts)
        if result.label != IntentLabel.OTHER or not self._use_llm or not bot:
            return result

        # ---------- 2) 可选 LLM 判别（JSON 输出） ----------
        try:
            sys = {
                "role": "system",
                "content": (
                    "你是一个用户意图分类器。请仅用 JSON 返回，字段："
                    "label(枚举: qna/summarize/translate/code/image_analyze/image_generate/rag_qa/small_talk/other),"
                    "require_rag(bool),require_image_generation(bool)。不要输出多余文本。"
                ),
            }
            usr = {
                "role": "user",
                "content": f"内容: {text}\n有附件(图片)吗: {any(a.type == AttachmentType.image for a in atts)}",
            }
            raw = await asyncio.wait_for(bot.chat([sys, usr], stream=False), timeout=self._llm_timeout)
            label, req_rag, req_img = self._parse_llm_json(raw or "")
            if label:
                return IntentResult(
                    label=label,
                    confidence=max(0.6, result.confidence),  # 提升一点置信
                    require_rag=req_rag,
                    require_image_generation=req_img,
                    note="llm-override"
                )
        except Exception as e:
            mlogger.warning(self.__class__.__name__, "llm_intent", msg=str(e))

        return result

    # ---------------- internal utils ----------------

    @staticmethod
    def _contains_any(t: str, keys: List[str]) -> bool:
        return any(k in t for k in keys)

    def _heuristic(self, text: str, atts: List[Attachment]) -> IntentResult:
        t = text.lower()
        t_cn = text  # 中文命中直接用原文判断

        has_img = any((a and getattr(a, "type", None) == AttachmentType.image) for a in atts)

        # 代码：包含代码块或典型关键字
        if "```" in text or re.search(r"\b(class|def|import|public|function|SELECT|INSERT|npm|pip)\b", text, re.I):
            return IntentResult(label=IntentLabel.CODE, confidence=0.9, note="rule: code")

        # 翻译
        if self._contains_any(t_cn, ["翻译", "英译中", "中译英"]) or "translate" in t:
            return IntentResult(label=IntentLabel.TRANSLATE, confidence=0.8, note="rule: translate")

        # 摘要
        if self._contains_any(t_cn, ["总结", "概括", "要点", "提炼", "摘要"]) or "summarize" in t:
            return IntentResult(label=IntentLabel.SUMMARIZE, confidence=0.75, note="rule: summarize")

        # 图片生成
        if self._contains_any(
            t_cn,
            ["生成图片", "画一张", "画一个", "出一张图", "出图", "海报", "插画", "logo", "banner", "头像", "壁纸"]
        ) or "generate image" in t or "dall-e" in t or "stable diffusion" in t:
            return IntentResult(label=IntentLabel.IMAGE_GENERATE, confidence=0.85, require_image_generation=True, note="rule: image_generate")

        # 图片分析（带图或指明看图）
        if has_img and self._contains_any(t_cn, ["识别", "分析", "看图", "OCR", "提取文字", "这张图"]):
            return IntentResult(label=IntentLabel.IMAGE_ANALYZE, confidence=0.8, note="rule: image_analyze")
        if has_img and not self._contains_any(t_cn, ["生成", "画"]):
            return IntentResult(label=IntentLabel.IMAGE_ANALYZE, confidence=0.65, note="rule: image_analyze(has_img)")

        # 可能需要检索增强（长文本/提到文档知识库/政策规范）
        if len(text) >= 250 or self._contains_any(t_cn, ["根据文档", "资料", "知识库", "手册", "规范", "政策", "内规", "技术白皮书"]):
            return IntentResult(label=IntentLabel.RAG_QA, confidence=0.7, require_rag=True, note="rule: rag_qa")

        # 一般问答（含问号/疑问词）
        if "?" in text or self._contains_any(t_cn, ["是什么", "怎么", "如何", "为什么", "多少"]):
            return IntentResult(label=IntentLabel.QNA, confidence=0.6, note="rule: qna")

        # 闲聊
        if self._contains_any(t_cn, ["你好", "在吗", "聊聊", "讲个笑话", "你是谁"]):
            return IntentResult(label=IntentLabel.SMALL_TALK, confidence=0.55, note="rule: small_talk")

        return IntentResult(label=IntentLabel.OTHER, confidence=0.4, note="rule: fallback")

    @staticmethod
    def _parse_llm_json(s: str):
        """
        解析 LLM 返回的 JSON（容错：从文本中提取 JSON 块）。
        返回: (label: IntentLabel|None, require_rag: bool, require_image_generation: bool)
        """
        try:
            import json
            # 提取第一个 {...} JSON
            m = re.search(r'\{[\s\S]*\}', s)
            if not m:
                return None, False, False
            obj = json.loads(m.group(0))
            label_raw = (obj.get("label") or "").strip().lower()
            mapping = {v.value: v for v in IntentLabel}
            label = mapping.get(label_raw, None)
            req_rag = bool(obj.get("require_rag", False))
            req_img = bool(obj.get("require_image_generation", False))
            return label, req_rag, req_img
        except Exception:
            return None, False, False