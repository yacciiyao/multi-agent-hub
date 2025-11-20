# -*- coding: utf-8 -*-
# @File: agent_service.py
# @Author: yaccii
# @Time: 2025-11-20 15:46
# @Description: Agent domain service，封装对 AgentRegistry 的访问
from __future__ import annotations

from typing import List, Optional, Dict, Any

from domain.agent import AgentConfig
from infrastructure.agent_registry import list_agents, get_agent


class AgentService:
    async def list_agents(self) -> List[Dict[str, Any]]:
        agents: List[AgentConfig] = list_agents()
        data: List[Dict[str, Any]] = []

        for a in agents:
            data.append(
                {
                    "key": a.key,
                    "name": a.name,
                    "description": a.description,
                    "icon": a.icon,
                    # 模态能力：比如 ["text"] 或 ["text", "image"]
                    "supports_modalities": a.supports_modalgitities,
                    "task_shortcuts": [
                        {
                            "id": t.id,
                            "title": t.title,
                            "subtitle": t.subtitle,
                        }
                        for t in a.task_shortcuts
                    ],
                }
            )

        return data

    async def get_agent_detail(self, agent_key: str) -> Optional[Dict[str, Any]]:
        agent: Optional[AgentConfig] = get_agent(agent_key)
        if not agent:
            return None

        return {
            "key": agent.key,
            "name": agent.name,
            "description": agent.description,
            "icon": agent.icon,
            "bot_name": agent.bot_name,
            "allowed_models": agent.allowed_models,
            "supports_modalities": agent.supports_modalities,
            "enable_rag": agent.enable_rag,
            "rag_top_k": agent.rag_top_k,
            "system_prompt": agent.system_prompt,
            "task_shortcuts": [
                {
                    "id": t.id,
                    "title": t.title,
                    "subtitle": t.subtitle,
                    "prompt_template": t.prompt_template,
                }
                for t in agent.task_shortcuts
            ],
        }