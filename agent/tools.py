"""Agent 可调用的工具：登记、检索、更新、核验与交接。"""
from __future__ import annotations

import random
import string
from typing import Any, Dict, List, Optional

from . import config, db, matcher, policy


def title_for(draft: Dict[str, Any]) -> str:
    color = str(draft.get("color") or "").strip()
    brand = str(draft.get("brand") or "").strip()
    category = str(draft.get("category") or "").strip()
    custom = str(draft.get("custom_name") or "").strip()
    if custom:
        category = custom
    if brand:
        value = " ".join(x for x in (color, brand, category) if x)
    else:
        value = f"{color}{category}".strip()
    return value or "未命名物品"


def description_for(draft: Dict[str, Any], role: str) -> str:
    parts = [title_for(draft)]
    if draft.get("location"):
        parts.append(f"地点：{draft['location']}")
    if draft.get("time"):
        parts.append(f"时间：{draft['time']}")
    marks = [m for m in (draft.get("marks") or []) if m]
    if marks:
        parts.append("公开特征：" + "、".join(marks[:4]))
    if role == "found" and draft.get("custody"):
        parts.append(f"保管地点：{draft['custody']}")
    notes = [str(n).strip() for n in (draft.get("notes") or []) if str(n).strip()]
    if notes:
        parts.append("补充：" + "；".join(notes[-3:]))
    return "；".join(parts)


def _time_iso(draft: Dict[str, Any]) -> str:
    dt, _ = matcher.parse_time_text(draft.get("time"))
    if dt:
        return dt.isoformat(timespec="seconds")
    return config.now().isoformat(timespec="seconds")


def create_lost(session: Dict[str, Any]) -> Dict[str, Any]:
    draft = session["draft"]
    case = db.create_case(
        {
            "role": "lost",
            "status": "open",
            "category": draft.get("category"),
            "title": title_for(draft),
            "description": description_for(draft, "lost"),
            "color": draft.get("color"),
            "brand": draft.get("brand"),
            "location": draft.get("location"),
            "time_text": draft.get("time"),
            "time_iso": _time_iso(draft),
            "marks": draft.get("marks") or [],
            "secrets": [],
            "contact": draft.get("contact") or "",
            "public_location": "",
            "session_id": session["id"],
        }
    )
    session["case_id"] = case["id"]
    session["case"] = case
    session["dirty"] = True
    db.add_event(session["id"], case["id"], "case_created", f"失主登记：{case['title']}")
    return case


def create_found(session: Dict[str, Any]) -> Dict[str, Any]:
    draft = session["draft"]
    secret = draft.get("secret_detail")
    if isinstance(secret, str):
        secret = policy.build_secret(secret, draft.get("category"))
    secrets = [secret] if isinstance(secret, dict) else []
    case = db.create_case(
        {
            "role": "found",
            "status": "open",
            "category": draft.get("category"),
            "title": title_for(draft),
            "description": description_for(draft, "found"),
            "color": draft.get("color"),
            "brand": draft.get("brand"),
            "location": draft.get("location"),
            "time_text": draft.get("time"),
            "time_iso": _time_iso(draft),
            "marks": draft.get("marks") or [],
            "secrets": secrets,
            "contact": draft.get("contact") or "",
            "public_location": draft.get("custody") or "校园失物招领服务台",
            "session_id": session["id"],
        }
    )
    session["case_id"] = case["id"]
    session["case"] = case
    session["dirty"] = False
    db.add_event(session["id"], case["id"], "case_created", f"拾得登记：{case['title']}")
    return case


def update_lost(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not session.get("case_id"):
        return None
    draft = session["draft"]
    case = db.update_case(
        session["case_id"],
        category=draft.get("category"),
        title=title_for(draft),
        description=description_for(draft, "lost"),
        color=draft.get("color"),
        brand=draft.get("brand"),
        location=draft.get("location"),
        time_text=draft.get("time"),
        time_iso=_time_iso(draft),
        marks=draft.get("marks") or [],
    )
    if case:
        session["case"] = case
    return case


def update_found(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not session.get("case_id"):
        return None
    draft = session["draft"]
    fields: Dict[str, Any] = {
        "category": draft.get("category"),
        "title": title_for(draft),
        "description": description_for(draft, "found"),
        "color": draft.get("color"),
        "brand": draft.get("brand"),
        "location": draft.get("location"),
        "time_text": draft.get("time"),
        "time_iso": _time_iso(draft),
        "marks": draft.get("marks") or [],
        "public_location": draft.get("custody") or "校园失物招领服务台",
    }
    secret = draft.get("secret_detail")
    if isinstance(secret, str):
        secret = policy.build_secret(secret, draft.get("category"))
    if isinstance(secret, dict):
        fields["secrets"] = [secret]
    case = db.update_case(session["case_id"], **fields)
    if case:
        session["case"] = case
    return case


def search_found(session: Dict[str, Any], top_k: int = 6) -> List[Dict[str, Any]]:
    candidates = [
        c for c in db.list_cases(role="found")
        if c.get("status") in ("open", "matched", "verifying")
    ]
    results = matcher.rank(session.get("case") or session["draft"], candidates, top_k=top_k)
    margin = 0.0
    if len(results) > 1:
        margin = max(0.0, results[0]["score"] - results[1]["score"])
    elif results:
        margin = 1.0
    for item in results:
        item["margin"] = round(margin, 4)
        item["candidate"] = db.public_case(item["candidate"])
    return results


def mark_verifying(session: Dict[str, Any]) -> None:
    verification = session.get("verification") or {}
    found_id = verification.get("case_id")
    if found_id:
        db.update_case(found_id, status="verifying")
    if session.get("case_id"):
        db.update_case(session["case_id"], status="matched")


def finalize_claim(
    session: Dict[str, Any],
    passed: bool,
    answers: List[Dict[str, Any]],
    ratio: float,
) -> Dict[str, Any]:
    verification = session.get("verification") or {}
    case = verification.get("case") or {}
    case_id = verification.get("case_id") or case.get("id")
    code = None
    if passed:
        code = "LF-" + "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=4))
    claim = db.create_claim(
        case_id=case_id,
        session_id=session["id"],
        answers=answers,
        score=ratio,
        status="accepted" if passed else "rejected",
        code=code,
    )
    if passed:
        db.update_case(case_id, status="claimed")
        if session.get("case_id"):
            db.update_case(session["case_id"], status="returned")
        db.add_event(session["id"], case_id, "claim_accepted", f"认领通过，认领码 {code}")
    else:
        db.update_case(case_id, status="open")
        if session.get("case_id"):
            db.update_case(session["case_id"], status="open")
        db.add_event(session["id"], case_id, "claim_rejected", f"核验未通过，得分 {ratio:.2f}")
    session["phase"] = "done" if passed else "open"
    session["verification_result"] = {
        "passed": passed,
        "claim": claim,
        "case": case,
    }
    return claim


def candidate_card(item: Dict[str, Any]) -> Dict[str, Any]:
    case = item["candidate"]
    return {
        "type": "candidate",
        "id": case.get("id"),
        "title": case.get("title") or "未命名物品",
        "subtitle": " · ".join(
            str(x) for x in [case.get("location"), case.get("time_text")] if x
        ),
        "score": item.get("score", 0),
        "reasons": item.get("reasons", []),
    }


def ticket_card(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    result = session.get("verification_result") or {}
    claim = result.get("claim") or {}
    if not result.get("passed"):
        return None
    case = result.get("case") or {}
    return {
        "type": "ticket",
        "code": claim.get("code"),
        "title": f"认领凭证 · {case.get('title', '失物')}",
        "location": case.get("public_location") or "校园失物招领服务台",
        "contact": case.get("contact") or "",
        "case_id": case.get("id"),
    }
