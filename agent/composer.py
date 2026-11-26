"""LEGACY：旧版模板/LLM 措辞层，仅被 legacy_rule_agent 使用。

当前 react_agent 直接让模型生成回复。

注意：这里只负责“怎么说”，不负责“下一步做什么”。
问题里不会传入任何私密核验细节，避免模型说漏嘴。
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .llm import llm

SLOT_HINTS = {
    "category": "物品类型，例如耳机、水杯、校园卡、鼠标",
    "location": "校园地点，楼栋 + 楼层/房间，例如图书馆三楼、纪忠楼 2 楼",
    "time": "大概时间，例如昨天下午、今天上午",
    "distinguishing": "颜色、品牌，或独特标记，例如挂件、贴纸、刻字",
    "marks": "独特标记，例如挂件、贴纸、刻字、划痕",
    "secret_detail": "只有主人知道的私密细节，例如盒内贴纸、刻字、耳机颜色、磨损；不能透露拾得者登记了什么",
    "custody": "物品现在保管在哪里，例如图书馆一楼服务台",
    "role": "用户是失主还是拾得者",
}

SLOT_LABELS = {
    "category": "物品类型",
    "location": "地点",
    "time": "时间",
    "distinguishing": "区分特征",
    "marks": "独特标记",
    "secret_detail": "核验细节",
    "custody": "保管地点",
}

QUESTION_SYSTEM = """你是校园失物招领助手“小寻”。根据对话状态，用自然、口语化的中文问一个问题。

要求：
1. 只问“要追问的字段”，一次只问一件事。
2. 不要用表单语气，不要一次列很多选项。
3. 不要编造用户没说过的事实。
4. 绝对不要透露拾得者登记的私密核验细节（如内部贴纸、刻字、划痕、具体保管点）。
5. 控制在 40 字以内，中文要通顺，避免“在……的”这类病句，像朋友帮忙一样自然。
6. 如果给了“必须带上的信息”，要自然地带进去。

只输出 JSON：{"question": "你要问的话"}
"""


def _public_draft(draft: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in (draft or {}).items()
        if k not in ("secret_detail", "secrets", "notes")
    }


def phrase_question(
    session: Dict[str, Any],
    slot: str,
    base_question: str,
    extra: str = "",
    must_include: str = "",
) -> str:
    if not llm.available or slot == "role":
        return base_question

    history = session.get("history") or []
    last_user = ""
    last_assistant = ""
    for item in reversed(history):
        if item.get("role") == "user" and not last_user:
            last_user = item.get("content", "")
        elif item.get("role") == "assistant" and not last_assistant:
            last_assistant = item.get("content", "")

    payload = json.dumps(
        {
            "身份": "失主" if session.get("role") == "lost" else "拾得者",
            "已知信息": _public_draft(session.get("draft") or {}),
            "要追问的字段": SLOT_LABELS.get(slot, slot),
            "字段提示": SLOT_HINTS.get(slot, ""),
            "上一轮助手": last_assistant[-300:],
            "用户刚说": last_user,
            "必须带上的信息": must_include,
            "模板参考（可以改写，不要照抄）": base_question,
            "额外上下文": extra,
        },
        ensure_ascii=False,
    )
    result = llm.chat_json(QUESTION_SYSTEM, payload, temperature=0.55, timeout=20)
    question = ""
    if result and isinstance(result.get("question"), str):
        question = result["question"].strip()
    question = " ".join(question.split())
    if not question:
        return base_question
    if len(question) > 90:
        question = question[:88]
    question = question.rstrip("，,。；;！!?？ ")
    question += "？"
    if "```" in question or "json" in question.lower():
        return base_question
    if must_include and must_include.rstrip("。") not in question:
        question = f"{must_include}{question}"
    return question
