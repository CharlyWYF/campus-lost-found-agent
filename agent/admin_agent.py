"""管理员 Agent：自然语言查询、统计、修改、审核与核销。

和面向学生的 react_agent 分开：
- 独立工具集和系统提示词；
- 写操作必须“预览 → 等待一轮用户确认 → 执行”；
- 所有查看/修改都写审计事件；
- 列表默认脱敏，详情才展开私密信息。
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

from . import config, db, matcher
from .llm import LLMError, llm

MAX_STEPS = 6
MAX_HISTORY = 14
SHOW_REASONING = os.environ.get("LF_SHOW_REASONING", "0") == "1"

CASE_STATUSES = ["open", "matched", "verifying", "claimed", "returned", "escalated"]
CASE_FIELDS = [
    "status", "title", "category", "color", "brand", "location",
    "time_text", "description", "marks", "contact", "public_location",
]

STATUS_CN = {
    "open": "待处理",
    "matched": "已匹配",
    "verifying": "核验中",
    "claimed": "已认领",
    "returned": "已归还",
    "escalated": "已转人工",
}
ROLE_CN = {"lost": "失物", "found": "拾得"}
FIELD_CN = {
    "status": "状态", "title": "标题", "category": "类别", "color": "颜色",
    "brand": "品牌", "location": "地点", "time_text": "时间",
    "description": "描述", "marks": "公开标记", "contact": "联系方式",
    "public_location": "保管点",
}


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "admin_search_cases",
            "description": (
                "按关键词、角色、状态、类别、地点、日期范围查询案件，返回脱敏列表和案件 ID。"
                "查询列表时优先用这个工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "标题/描述/地点/颜色/品牌/标记中的关键词"},
                    "role": {"type": "string", "enum": ["lost", "found"]},
                    "status": {"type": "string", "enum": CASE_STATUSES},
                    "category": {"type": "string"},
                    "location": {"type": "string"},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                    "limit": {"type": "integer", "description": "默认 20，最多 100"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "admin_get_case",
            "description": (
                "查看单个案件的完整详情：公开信息、核验细节、联系方式、保管点、认领记录和审计事件。"
                "会记录管理员查看日志。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"case_id": {"type": "string"}},
                "required": ["case_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "admin_search_claims",
            "description": "查询认领记录，可按状态、案件、认领码、日期范围筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["accepted", "rejected", "redeemed"]},
                    "case_id": {"type": "string"},
                    "code": {"type": "string"},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                    "limit": {"type": "integer", "description": "默认 20，最多 100"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "admin_get_claim",
            "description": "查看单条认领记录的完整详情，包括每个核验问题的回答和得分。",
            "parameters": {
                "type": "object",
                "properties": {"claim_id": {"type": "string"}},
                "required": ["claim_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "admin_stats",
            "description": "查看总体统计：案件状态、失物/拾得数量、认领通过/拒绝/核销、热门类别和地点、超 7 天未处理数。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "admin_recent_events",
            "description": "查看审计日志，可按案件或事件类型筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "event_type": {"type": "string"},
                    "limit": {"type": "integer", "description": "默认 30，最多 200"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "admin_preview_case_update",
            "description": (
                "预览案件修改，不落库。必须先把 before→after 展示给管理员，等管理员明确确认后，"
                "下一轮再调用 admin_apply_pending_action。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "fields": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": CASE_STATUSES},
                            "title": {"type": "string"},
                            "category": {"type": "string"},
                            "color": {"type": "string"},
                            "brand": {"type": "string"},
                            "location": {"type": "string"},
                            "time_text": {"type": "string"},
                            "description": {"type": "string"},
                            "marks": {"type": "array", "items": {"type": "string"}},
                            "contact": {"type": "string"},
                            "public_location": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                    "reason": {"type": "string", "description": "修改原因，会写入审计日志"},
                },
                "required": ["case_id", "fields", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "admin_preview_claim_review",
            "description": (
                "预览认领审核：approve 把认领改为通过并把物品标为已认领；reject 改为拒绝并重新打开案件。"
                "必须先预览并等待管理员确认。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["approve", "reject"]},
                    "reason": {"type": "string"},
                },
                "required": ["claim_id", "decision", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "admin_preview_redeem",
            "description": "预览认领码核销：把认领记录标记为 redeemed，把物品标记为已归还。必须先预览并等待确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["code", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "admin_apply_pending_action",
            "description": "执行已经预览过、且管理员已明确确认的变更。没有 pending action 或未经过一轮确认会报错。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "admin_cancel_pending_action",
            "description": "取消当前待确认的变更。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


# ------------------------------------------------------------------ 会话

class AdminStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create(self, name: str) -> Dict[str, Any]:
        with self._lock:
            session = {
                "id": "admin-" + uuid.uuid4().hex[:10],
                "name": name,
                "is_admin": True,
                "pending_action": None,
                "last_case_ids": [],
                "last_claim_ids": [],
                "history": [],
                "turn": 0,
            }
            self._data[session["id"]] = session
            return session

    def get(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        return self._data.get(session_id)

    def reset(self) -> None:
        with self._lock:
            self._data.clear()


store = AdminStore()


# ------------------------------------------------------------------ 展示辅助

def _case_line(case: Dict[str, Any]) -> str:
    return (
        f"{case.get('id')} | {ROLE_CN.get(case.get('role'), case.get('role'))} | "
        f"{STATUS_CN.get(case.get('status'), case.get('status'))} | "
        f"{case.get('title') or '未命名'} | {case.get('location') or '-'} | "
        f"{case.get('time_text') or '-'} | 认领 {case.get('claim_count', 0)} 次"
    )


def _case_detail(case: Dict[str, Any], claims: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": case.get("id"),
        "role": case.get("role"),
        "status": case.get("status"),
        "title": case.get("title"),
        "category": case.get("category"),
        "color": case.get("color"),
        "brand": case.get("brand"),
        "location": case.get("location"),
        "time_text": case.get("time_text"),
        "description": case.get("description"),
        "marks": case.get("marks") or [],
        "secrets": case.get("secrets") or [],
        "contact": case.get("contact"),
        "public_location": case.get("public_location"),
        "created_at": case.get("created_at"),
        "claims": [
            {
                "id": c.get("id"),
                "status": c.get("status"),
                "score": c.get("score"),
                "code": c.get("code"),
                "created_at": c.get("created_at"),
                "answers": c.get("answers") or [],
            }
            for c in claims
        ],
        "events": [
            {
                "id": e.get("id"),
                "type": e.get("type"),
                "detail": e.get("detail"),
                "created_at": e.get("created_at"),
            }
            for e in events[:10]
        ],
    }


def _claim_line(claim: Dict[str, Any]) -> str:
    return (
        f"{claim.get('id')} | 案件 {claim.get('case_id')} | "
        f"{claim.get('case_title') or '-'} | {claim.get('status')} | "
        f"得分 {claim.get('score')} | 码 {claim.get('code') or '-'} | {claim.get('created_at')}"
    )


# ------------------------------------------------------------------ 工具实现

def _ok(message: str, **data) -> Dict[str, Any]:
    return {"ok": True, "message": message, **data}


def _fail(error: str, message: str, **data) -> Dict[str, Any]:
    return {"ok": False, "error": error, "message": message, **data}


def _tool_search_cases(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    cases = db.search_cases_admin(
        keyword=str(args.get("keyword") or ""),
        role=str(args.get("role") or ""),
        status=str(args.get("status") or ""),
        category=str(args.get("category") or ""),
        location=str(args.get("location") or ""),
        date_from=str(args.get("date_from") or ""),
        date_to=str(args.get("date_to") or ""),
        limit=int(args.get("limit") or 20),
    )
    session["last_case_ids"] = [c["id"] for c in cases[:20]]
    public = [
        {
            "id": c.get("id"),
            "role": c.get("role"),
            "status": c.get("status"),
            "title": c.get("title"),
            "category": c.get("category"),
            "color": c.get("color"),
            "location": c.get("location"),
            "time_text": c.get("time_text"),
            "claim_count": c.get("claim_count", 0),
            "created_at": c.get("created_at"),
        }
        for c in cases
    ]
    return _ok(
        f"找到 {len(cases)} 条案件",
        count=len(cases),
        cases=public,
        display="\n".join(_case_line(c) for c in cases) or "（无结果）",
    )


def _tool_get_case(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    case_id = str(args.get("case_id") or "").strip()
    case = db.get_case(case_id)
    if not case:
        return _fail("not_found", f"没有找到案件 {case_id}")
    session["last_case_ids"] = [case_id]
    claims = db.claims_for_case(case_id)
    events = db.events_for_case(case_id)
    db.add_event(session["id"], case_id, "admin_view_case", f"管理员 {session.get('name')} 查看案件详情")
    detail = _case_detail(case, claims, events)
    return _ok(f"案件 {case_id} 详情", case=detail, display=json.dumps(detail, ensure_ascii=False, indent=2))


def _tool_search_claims(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    claims = db.search_claims_admin(
        status=str(args.get("status") or ""),
        case_id=str(args.get("case_id") or ""),
        code=str(args.get("code") or ""),
        date_from=str(args.get("date_from") or ""),
        date_to=str(args.get("date_to") or ""),
        limit=int(args.get("limit") or 20),
    )
    session["last_claim_ids"] = [c["id"] for c in claims[:20]]
    public = [
        {
            "id": c.get("id"),
            "case_id": c.get("case_id"),
            "case_title": c.get("case_title"),
            "case_role": c.get("case_role"),
            "case_status": c.get("case_status"),
            "status": c.get("status"),
            "score": c.get("score"),
            "code": c.get("code"),
            "created_at": c.get("created_at"),
        }
        for c in claims
    ]
    return _ok(
        f"找到 {len(claims)} 条认领记录",
        count=len(claims),
        claims=public,
        display="\n".join(_claim_line(c) for c in claims) or "（无结果）",
    )


def _tool_get_claim(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    claim_id = str(args.get("claim_id") or "").strip()
    claim = db.get_claim(claim_id)
    if not claim:
        return _fail("not_found", f"没有找到认领记录 {claim_id}")
    session["last_claim_ids"] = [claim_id]
    db.add_event(session["id"], claim.get("case_id"), "admin_view_claim", f"管理员 {session.get('name')} 查看认领详情")
    return _ok(
        f"认领 {claim_id} 详情",
        claim=claim,
        display=json.dumps(claim, ensure_ascii=False, indent=2),
    )


def _tool_stats(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    stats = db.admin_stats()
    display = (
        f"案件总数 {stats['total_cases']}，未结 {stats['open_cases']}，超 7 天 {stats['overdue_7d']}\n"
        f"状态：{stats['by_status']}\n"
        f"认领：{stats['claims']}\n"
        f"热门类别：{stats['top_categories']}\n"
        f"热门地点：{stats['top_locations']}"
    )
    return _ok("统计完成", stats=stats, display=display)


def _tool_events(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    events = db.search_events_admin(
        case_id=str(args.get("case_id") or ""),
        event_type=str(args.get("event_type") or ""),
        limit=int(args.get("limit") or 30),
    )
    return _ok(
        f"最近 {len(events)} 条审计事件",
        count=len(events),
        events=events,
        display="\n".join(
            f"{e.get('created_at')} | {e.get('type')} | {e.get('case_id') or '-'} | {e.get('detail')}"
            for e in events
        ) or "（无记录）",
    )


def _preview_text_case(before: Dict[str, Any], fields: Dict[str, Any]) -> str:
    lines = [f"将修改案件 {before.get('id')}（{before.get('title')}）："]
    for key, value in fields.items():
        old = before.get(key)
        if key == "marks":
            old = old or []
        if key == "status":
            old_display = STATUS_CN.get(old, old)
            new_display = STATUS_CN.get(value, value)
        else:
            old_display = old if old not in (None, "") else "-"
            new_display = value if value not in (None, "") else "-"
        lines.append(f"- {FIELD_CN.get(key, key)}：{old_display} → {new_display}")
    return "\n".join(lines)


def _tool_preview_case_update(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    case_id = str(args.get("case_id") or "").strip()
    case = db.get_case(case_id)
    if not case:
        return _fail("not_found", f"没有找到案件 {case_id}")
    raw_fields = args.get("fields") or {}
    if not isinstance(raw_fields, dict):
        return _fail("invalid_fields", "fields 必须是对象")
    fields: Dict[str, Any] = {}
    for key, value in raw_fields.items():
        if key not in CASE_FIELDS:
            return _fail("invalid_field", f"不允许修改字段：{key}")
        if key == "status" and value not in CASE_STATUSES:
            return _fail("invalid_status", f"不支持的状态：{value}")
        if key == "marks":
            if not isinstance(value, list):
                return _fail("invalid_marks", "marks 必须是数组")
            value = [str(v).strip() for v in value if str(v).strip()]
        if key in ("title", "description", "contact", "public_location", "location", "time_text", "category", "color", "brand"):
            value = str(value).strip()
        if case.get(key) == value:
            continue
        fields[key] = value
    if not fields:
        return _fail("no_change", "没有实际变化")
    reason = str(args.get("reason") or "管理员修改").strip()
    preview = _preview_text_case(case, fields)
    session["pending_action"] = {
        "type": "case_update",
        "turn": session["turn"],
        "case_id": case_id,
        "fields": fields,
        "reason": reason,
        "preview": preview,
    }
    return _ok(
        "已生成修改预览，等待管理员确认",
        requires_confirmation=True,
        preview=preview,
        pending=True,
    )


def _tool_preview_claim_review(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    claim_id = str(args.get("claim_id") or "").strip()
    claim = db.get_claim(claim_id)
    if not claim:
        return _fail("not_found", f"没有找到认领记录 {claim_id}")
    if claim.get("status") == "redeemed":
        return _fail("already_redeemed", "这条认领已经核销，不能再改状态")
    decision = args.get("decision")
    if decision not in ("approve", "reject"):
        return _fail("invalid_decision", "decision 只能是 approve 或 reject")
    reason = str(args.get("reason") or "管理员复核").strip()
    case = db.get_case(claim.get("case_id"))
    target_case_status = ""
    if decision == "approve":
        target_case_status = "claimed" if (case or {}).get("role") == "found" else "returned"
    else:
        target_case_status = "open"
    claim_status_cn = {"accepted": "已通过", "rejected": "已拒绝", "redeemed": "已核销"}
    preview = (
        f"将认领 {claim_id}（案件 {claim.get('case_id')}）状态 "
        f"{claim_status_cn.get(claim.get('status'), claim.get('status'))} → "
        f"{'已通过' if decision == 'approve' else '已拒绝'}；"
        f"案件状态 → {STATUS_CN.get(target_case_status, target_case_status)}。原因：{reason}"
    )
    session["pending_action"] = {
        "type": "claim_review",
        "turn": session["turn"],
        "claim_id": claim_id,
        "decision": decision,
        "case_id": claim.get("case_id"),
        "target_case_status": target_case_status,
        "reason": reason,
        "preview": preview,
    }
    return _ok("已生成认领审核预览，等待管理员确认", requires_confirmation=True, preview=preview, pending=True)


def _tool_preview_redeem(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    code = str(args.get("code") or "").strip().upper()
    claim = db.get_claim_by_code(code)
    if not claim:
        return _fail("not_found", f"没有找到认领码 {code}")
    if claim.get("status") == "redeemed":
        return _fail("already_redeemed", "这个认领码已经核销过了")
    if claim.get("status") != "accepted":
        return _fail("not_accepted", "只有已通过的认领才能核销")
    reason = str(args.get("reason") or "服务台核销").strip()
    preview = (
        f"将核销认领码 {code}（案件 {claim.get('case_id')}，{claim.get('case_title')}）："
        f"认领 → 已核销，案件 → 已归还。原因：{reason}"
    )
    session["pending_action"] = {
        "type": "redeem",
        "turn": session["turn"],
        "claim_id": claim.get("id"),
        "code": code,
        "case_id": claim.get("case_id"),
        "reason": reason,
        "preview": preview,
    }
    return _ok("已生成核销预览，等待管理员确认", requires_confirmation=True, preview=preview, pending=True)


def _tool_apply(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    pending = session.get("pending_action")
    if not pending:
        return _fail("no_pending", "当前没有待确认的变更")
    if session.get("turn", 0) <= pending.get("turn", 0):
        return _fail(
            "confirmation_required",
            "变更还停留在同一轮对话，必须先展示给管理员并等待明确确认",
        )
    kind = pending["type"]
    if kind == "case_update":
        before = db.get_case(pending["case_id"])
        fields = dict(pending["fields"])
        if "time_text" in fields:
            dt, _ = matcher.parse_time_text(fields["time_text"])
            if dt:
                fields["time_iso"] = dt.isoformat(timespec="seconds")
        after = db.update_case(pending["case_id"], **fields)
        db.add_event(
            session["id"],
            pending["case_id"],
            "admin_case_updated",
            json.dumps(
                {
                    "admin": session.get("name"),
                    "reason": pending.get("reason"),
                    "before": {k: before.get(k) for k in fields} if before else {},
                    "after": {k: (after or {}).get(k) for k in fields},
                },
                ensure_ascii=False,
            ),
        )
    elif kind == "claim_review":
        claim = db.update_claim_status(
            pending["claim_id"],
            "accepted" if pending["decision"] == "approve" else "rejected",
        )
        if pending.get("case_id"):
            db.update_case(pending["case_id"], status=pending["target_case_status"])
        db.add_event(
            session["id"],
            pending.get("case_id"),
            "admin_claim_reviewed",
            json.dumps(
                {
                    "admin": session.get("name"),
                    "claim_id": pending["claim_id"],
                    "decision": pending["decision"],
                    "reason": pending.get("reason"),
                },
                ensure_ascii=False,
            ),
        )
        after = claim
    elif kind == "redeem":
        claim = db.update_claim_status(pending["claim_id"], "redeemed")
        if pending.get("case_id"):
            db.update_case(pending["case_id"], status="returned")
        db.add_event(
            session["id"],
            pending.get("case_id"),
            "admin_claim_redeemed",
            json.dumps(
                {
                    "admin": session.get("name"),
                    "claim_id": pending["claim_id"],
                    "code": pending.get("code"),
                    "reason": pending.get("reason"),
                },
                ensure_ascii=False,
            ),
        )
        after = claim
    else:
        return _fail("invalid_pending", "未知的待确认操作")

    session["pending_action"] = None
    return _ok(
        "变更已执行",
        applied=True,
        type=kind,
        result=after,
        preview=pending.get("preview"),
    )


def _tool_cancel(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    pending = session.get("pending_action")
    if not pending:
        return _fail("no_pending", "当前没有待确认的变更")
    session["pending_action"] = None
    return _ok("已取消待确认变更")


def _execute_tool(session: Dict[str, Any], name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    dispatch = {
        "admin_search_cases": _tool_search_cases,
        "admin_get_case": _tool_get_case,
        "admin_search_claims": _tool_search_claims,
        "admin_get_claim": _tool_get_claim,
        "admin_stats": _tool_stats,
        "admin_recent_events": _tool_events,
        "admin_preview_case_update": _tool_preview_case_update,
        "admin_preview_claim_review": _tool_preview_claim_review,
        "admin_preview_redeem": _tool_preview_redeem,
        "admin_apply_pending_action": _tool_apply,
        "admin_cancel_pending_action": _tool_cancel,
    }
    func = dispatch.get(name)
    if not func:
        return _fail("unknown_tool", f"未知工具：{name}")
    return func(session, args)


# ------------------------------------------------------------------ 提示词与主循环

def _state(session: Dict[str, Any]) -> Dict[str, Any]:
    pending = session.get("pending_action")
    return {
        "管理员": session.get("name"),
        "今天": config.now().strftime("%Y-%m-%d"),
        "待确认操作": {
            "type": pending.get("type"),
            "preview": pending.get("preview"),
        } if pending else None,
        "最近案件ID": (session.get("last_case_ids") or [])[:8],
        "最近认领ID": (session.get("last_claim_ids") or [])[:8],
    }


SYSTEM_PROMPT = """你是「小寻·管理台」，校园失物招领后台的管理员 Agent。你面对的是服务台老师，不是学生。

职责：
1. 用自然语言帮管理员查案件、查认领、查统计、查审计日志；
2. 帮管理员修改案件信息、审核认领、核销认领码；
3. 所有回答必须基于工具结果，不要编造。

工具使用规则：
1. 查询列表优先用 admin_search_cases / admin_search_claims；看单条详情用 admin_get_case / admin_get_claim。
2. 修改必须两步：先调用 admin_preview_* 生成 before→after 预览，把预览展示给管理员并询问是否确认；管理员明确说“确认/执行”后，下一轮才调用 admin_apply_pending_action。
3. 绝对不要在同一轮里预览后立刻执行；也不要绕过预览直接修改。
4. 不要删除数据，只允许状态流转：open / matched / verifying / claimed / returned / escalated。
5. 列表默认脱敏；只有管理员明确问某个案件时，才展开联系方式、核验细节。
6. 不要承诺通知用户；系统没有通知能力。
7. 超过 5 条结果时，先给摘要，再列关键几条。
8. 用简洁自然的中文，不要向管理员展示工具名、JSON 字段名。

当前状态（JSON）：
{state}
"""


def _system_prompt(session: Dict[str, Any]) -> str:
    return SYSTEM_PROMPT.replace(
        "{state}", json.dumps(_state(session), ensure_ascii=False, indent=2)
    )


def _build_messages(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(session)}
    ]
    for item in (session.get("history") or [])[-MAX_HISTORY:]:
        messages.append({"role": item["role"], "content": item["content"]})
    return messages


def _trace(name: str, args: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    if name == "admin_search_cases":
        detail = f"查询案件：{result.get('count', 0)} 条"
    elif name == "admin_search_claims":
        detail = f"查询认领：{result.get('count', 0)} 条"
    elif name == "admin_get_case":
        detail = f"查看案件：{args.get('case_id')}"
    elif name == "admin_get_claim":
        detail = f"查看认领：{args.get('claim_id')}"
    elif name == "admin_stats":
        detail = "查看统计"
    elif name == "admin_recent_events":
        detail = f"查看审计：{result.get('count', 0)} 条"
    elif name.startswith("admin_preview_"):
        detail = f"生成预览：{result.get('preview', '')[:160]}"
    elif name == "admin_apply_pending_action":
        detail = "执行变更" if result.get("ok") else f"执行失败：{result.get('message')}"
    elif name == "admin_cancel_pending_action":
        detail = "取消变更"
    else:
        detail = result.get("message") or ""
    if not result.get("ok") and name not in ("admin_apply_pending_action",):
        detail = f"失败：{result.get('message')}"
    return {"type": "tool", "title": name, "detail": detail, "data": {"args": args, "ok": result.get("ok")}}


def _public_session(session: Dict[str, Any]) -> Dict[str, Any]:
    pending = session.get("pending_action")
    return {
        "id": session["id"],
        "name": session.get("name"),
        "is_admin": True,
        "turn": session.get("turn", 0),
        "pending": bool(pending),
        "pending_preview": (pending or {}).get("preview"),
        "last_case_ids": session.get("last_case_ids") or [],
        "last_claim_ids": session.get("last_claim_ids") or [],
    }


def _error(session: Dict[str, Any], trace: List[Dict[str, Any]], exc: Exception) -> Dict[str, Any]:
    detail = getattr(exc, "detail", "") or str(exc)
    trace.append({"type": "error", "title": "模型调用失败", "detail": detail[:500]})
    return {
        "reply": "抱歉，模型服务暂时不可用，我这边没法继续处理。请稍后重试。",
        "trace": trace,
        "session": _public_session(session),
        "error": {"code": "LLM_UNAVAILABLE", "message": str(exc), "detail": detail[:500]},
    }


def handle(session_id: Optional[str], text: str) -> Dict[str, Any]:
    session = store.get(session_id)
    if not session or not session.get("is_admin"):
        return {
            "reply": "未授权：请先通过管理员登录。",
            "trace": [],
            "session": {"is_admin": False},
            "error": {"code": "UNAUTHORIZED", "message": "管理员会话不存在"},
        }
    session["turn"] += 1
    text = (text or "").strip()
    session["history"].append({"role": "user", "content": text})
    trace: List[Dict[str, Any]] = []

    if not llm.available:
        return _error(session, trace, LLMError("模型服务不可用或已熔断", llm.last_error))

    messages = _build_messages(session)
    for _ in range(MAX_STEPS):
        try:
            message = llm.chat_message(messages, tools=TOOLS, temperature=0.2)
        except LLMError as exc:
            return _error(session, trace, exc)

        reasoning = message.get("reasoning_content") or ""
        if SHOW_REASONING and reasoning and not any(t.get("type") == "llm" for t in trace):
            trace.append(
                {
                    "type": "llm",
                    "title": "LLM 推理（节选）",
                    "detail": " ".join(str(reasoning).split())[:320],
                }
            )

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result = _execute_tool(session, name, args)
                trace.append(_trace(name, args, result))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            continue

        content = (message.get("content") or "").strip()
        if not content:
            return _error(session, trace, LLMError("模型返回了空回复"))
        session["history"].append({"role": "assistant", "content": content})
        return {
            "reply": content,
            "trace": trace,
            "session": _public_session(session),
            "error": None,
        }

    return _error(session, trace, LLMError("工具调用步数超出上限"))
