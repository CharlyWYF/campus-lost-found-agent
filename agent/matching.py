"""检索 + LLM 重排：代码负责召回，模型负责判断“像不像”。"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from . import db, matcher
from .llm import LLMError, llm

RERANK_SYSTEM = """你是校园失物招领的匹配专家。给定失主的描述和若干条拾得记录，判断每一条与失主描述是否指向同一件物品。

评分要点：
1. 物品类别、颜色、品牌是否一致；
2. 地点、时间是否接近；
3. 独特标记（挂件、贴纸、刻字、划痕、内部特征）是否一致；
4. 不要因为“同类别 + 同颜色”就给高分；独特标记冲突要显著降分；
5. 信息不足时给中间分，不要猜；
6. reason 里不要复述候选的具体独特标记（如挂件颜色、图案、刻字内容），只写“独特标记一致/冲突/缺失”。

只输出 JSON：
{"matches": [{"id": "记录ID", "score": 0.0, "reason": "一句话理由"}]}
score 范围 0–1，必须覆盖所有候选。
"""


def _query_from_draft(draft: Dict[str, Any]) -> Dict[str, Any]:
    parts = []
    for key in ("color", "brand", "category", "custom_name"):
        value = draft.get(key)
        if value:
            parts.append(str(value))
    if draft.get("location"):
        parts.append(f"地点：{draft['location']}")
    if draft.get("time"):
        parts.append(f"时间：{draft['time']}")
    marks = [str(m) for m in (draft.get("marks") or []) if m]
    if marks:
        parts.append("特征：" + "、".join(marks))
    return {
        "role": "lost",
        "category": draft.get("category"),
        "custom_name": draft.get("custom_name"),
        "color": draft.get("color"),
        "brand": draft.get("brand"),
        "location": draft.get("location"),
        "time_text": draft.get("time"),
        "marks": marks,
        "description": "；".join(parts),
    }


def _public_candidate(case: Dict[str, Any]) -> Dict[str, Any]:
    # 只给重排模型公开信息；绝不能带保管点、联系方式、私密细节
    parts = [
        case.get("title"), case.get("category"), case.get("color"),
        case.get("brand"), case.get("location"), case.get("time_text"),
        "、".join(case.get("marks") or []),
    ]
    return {
        "id": case.get("id"),
        "title": case.get("title"),
        "category": case.get("category"),
        "color": case.get("color"),
        "brand": case.get("brand"),
        "location": case.get("location"),
        "time_text": case.get("time_text"),
        "marks": case.get("marks") or [],
        "description": "；".join(str(p) for p in parts if p),
    }


def search(session: Dict[str, Any], top_k: int = 8) -> Dict[str, Any]:
    draft = session.get("draft") or {}
    cases = [
        c for c in db.list_cases(role="found")
        if c.get("status") in ("open", "matched", "verifying")
    ]
    if not cases:
        return {"ok": True, "count": 0, "recommendation": "none", "candidates": []}

    query = _query_from_draft(draft)
    raw = matcher.rank(query, cases, top_k=top_k)
    if not raw:
        return {"ok": True, "count": 0, "recommendation": "none", "candidates": []}

    payload = json.dumps(
        {
            "失主描述": {
                "类别": draft.get("category") or draft.get("custom_name"),
                "颜色": draft.get("color"),
                "品牌": draft.get("brand"),
                "地点": draft.get("location"),
                "时间": draft.get("time"),
                "独特标记": draft.get("marks") or [],
            },
            "候选记录": [
                {
                    **_public_candidate(item["candidate"]),
                    "初筛分": item.get("score"),
                }
                for item in raw
            ],
        },
        ensure_ascii=False,
    )
    result = llm.chat_json_strict(RERANK_SYSTEM, payload, temperature=0.0, timeout=30)
    matches = result.get("matches")
    if not isinstance(matches, list):
        raise LLMError("匹配重排返回格式异常", str(result)[:400])

    llm_scores: Dict[str, Dict[str, Any]] = {}
    for item in matches:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("id") or "")
        if not case_id:
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        llm_scores[case_id] = {
            "score": max(0.0, min(1.0, score)),
            "reason": str(item.get("reason") or "")[:160],
        }

    ranked: List[Dict[str, Any]] = []
    for item in raw:
        case = _public_candidate(item["candidate"])
        case_id = case["id"]
        model = llm_scores.get(case_id)
        retrieval = float(item.get("score") or 0.0)
        if model:
            final = round(0.35 * retrieval + 0.65 * model["score"], 4)
            model_reason = model["reason"]
            model_score = model["score"]
        else:
            final = round(retrieval, 4)
            model_reason = "模型未单独评分"
            model_score = None
        # 理由只保留召回层的通用依据；不把模型看到的候选具体标记透给用户
        reasons = list(item.get("reasons") or [])
        if model_score is not None:
            if model_score >= 0.7:
                reasons.insert(0, "语义匹配度高")
            elif model_score < 0.4:
                reasons.insert(0, "语义匹配度较低")
        ranked.append(
            {
                "candidate": case,
                "retrieval_score": round(retrieval, 4),
                "llm_score": model_score,
                "score": final,
                "reasons": reasons[:4],
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    margin = 0.0
    if len(ranked) > 1:
        margin = ranked[0]["score"] - ranked[1]["score"]
    elif ranked:
        margin = 1.0

    top = ranked[0]["score"] if ranked else 0.0
    if top >= 0.76 and margin >= 0.08:
        recommendation = "strong"
    elif top >= 0.55:
        recommendation = "possible"
    else:
        recommendation = "weak"

    return {
        "ok": True,
        "count": len(ranked),
        "recommendation": recommendation,
        "margin": round(margin, 4),
        "query": query,
        "candidates": [
            {
                "id": item["candidate"].get("id"),
                "title": item["candidate"].get("title"),
                "category": item["candidate"].get("category"),
                "color": item["candidate"].get("color"),
                "location": item["candidate"].get("location"),
                "time_text": item["candidate"].get("time_text"),
                "has_marks": bool(item["candidate"].get("marks")),
                "score": item["score"],
                "retrieval_score": item["retrieval_score"],
                "llm_score": item["llm_score"],
                "reasons": item["reasons"],
            }
            for item in ranked[:5]
        ],
    }
