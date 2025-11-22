# -*- coding: utf-8 -*-
# @File: agent_router.py
# @Author: yaccii
# @Time: 2025-11-20 15:46
# @Description:
from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from core.agent_service import AgentService
from infrastructure.response import success, failure


def get_agent_service() -> AgentService:
    return AgentService()


class AgentDetailQuery(BaseModel):
    key: str = Field(..., description="Agent key")


router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/list", summary="获取 Agent 列表")
async def list_agents():
    svc = get_agent_service()
    try:
        data = await svc.list_agents()
        return success(data=data)
    except Exception as e:
        return failure(message=str(e))


@router.get("/detail", summary="获取单个 Agent 详情")
async def get_agent_detail(key: str = Query(..., description="Agent key")):
    svc = get_agent_service()
    try:
        detail = await svc.get_agent_detail(agent_key=key)
        if not detail:
            return failure(message="agent not found")
        return success(data=detail)
    except Exception as e:
        return failure(message=str(e))