"""LLM 自主规划 + 原生 tool calling 的失物招领 Agent。

设计原则：
- 语言理解、下一步动作、追问措辞、候选重排、语义核验全部交给 LLM；
- 代码只提供工具、状态和不可绕过的风控（未核验通过不能出认领码）；
- API 失败不静默降级，直接返回明确错误。
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from . import db, matcher, matching, nlu, policy, tools
from .llm import LLMError, llm

MAX_STEPS = 6
MAX_HISTORY = 12
# 默认不把模型原始推理放进用户可见轨迹：推理可能复述用户刚说的核验细节
SHOW_REASONING = os.environ.get("LF_SHOW_REASONING", "0") == "1"


# ------------------------------------------------------------------ 工具定义

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_role",
            "description": "设置当前会话身份。只有用户明确说“我丢了/我的东西不见了”或“我捡到/我找到一个东西”时才调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["lost", "found"]},
                },
                "required": ["role"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_draft",
            "description": (
                "把用户这句话里明确提到的物品信息合并进当前草稿。"
                "只写用户说过的，禁止猜测；用户纠正时覆盖旧值。"
                "如果用户没有回答上一个问题，而是补充了别的内容，按实际内容填写，不要硬填。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "物品类别，如耳机盒/耳机/手机/水杯/校园卡/鼠标/键盘；词表外填其他",
                    },
                    "custom_name": {
                        "type": "string",
                        "description": "词表外的具体物品名，如小提琴、吉他",
                    },
                    "color": {"type": "string"},
                    "brand": {"type": "string"},
                    "location": {
                        "type": "string",
                        "description": "校园地点，保留楼栋/楼层/房间，如纪忠楼 2 楼",
                    },
                    "time": {"type": "string", "description": "口语时间原文，如昨天下午"},
                    "marks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "独特标记，如黄色小鸭子挂件、蓝色皮卡丘贴纸",
                    },
                    "custody": {
                        "type": "string",
                        "description": "拾得者把物品放在哪里，如图书馆一楼服务台",
                    },
                    "contact": {"type": "string"},
                    "secret_detail": {
                        "type": "string",
                        "description": (
                            "只有主人知道的私密细节，仅拾得登记使用；不要放进公开描述。"
                            "只要拾得者提供了一个可用于核对的具体细节（贴纸/划痕/刻字/内部颜色/数量/品牌等），"
                            "就放这里，并且不要再要求第二个细节。"
                        ),
                    },
                    "no_secret": {
                        "type": "boolean",
                        "description": "拾得者确实无法提供任何核验细节时设为 true",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_lost_case",
            "description": "创建寻物案件。至少需要类别、地点、时间；缺字段会返回 missing。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_found_case",
            "description": "创建拾得案件。至少需要类别、地点、时间、一个只有主人知道的细节、保管地点。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_found_cases",
            "description": (
                "用当前草稿检索拾得记录，返回候选、匹配分和 recommendation。"
                "recommendation=strong 可直接核验；possible 先追问区分特征；weak 告知暂无匹配。"
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_verification",
            "description": (
                "对一条候选开始归属核验。返回第一个只有主人能回答的问题，不返回答案。"
                "只能对 search_found_cases 返回的候选调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string", "description": "候选 ID；不填则用最高分候选"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_verification_answer",
            "description": "提交用户对当前核验问题的回答。返回下一题，或 passed / failed。",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "用户的原话，不要改写"},
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_claim",
            "description": "核验 passed 后生成认领码和领取地点。未通过调用会报错。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_staff",
            "description": "转人工复核。核验失败、记录缺少核验细节、或用户要求人工时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "转人工的原因，简短中文"},
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]


# ------------------------------------------------------------------ 会话

class SessionStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get_or_create(self, session_id: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            if session_id and session_id in self._data:
                return self._data[session_id]
            new_id = session_id or uuid.uuid4().hex[:12]
            session = {
                "id": new_id,
                "role": None,
                "phase": "intake",
                "draft": {},
                "case_id": None,
                "case": None,
                "candidates": [],
                "last_search": None,
                "verification": None,
                "verification_result": None,
                "claim": None,
                "last_case": None,
                "last_question": "",
                "history": [],
                "turn": 0,
            }
            self._data[new_id] = session
            return session

    def reset(self) -> None:
        with self._lock:
            self._data.clear()


store = SessionStore()


# ------------------------------------------------------------------ 状态与提示词

def _public_draft(draft: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in (draft or {}).items()
        if k not in ("secret_detail", "secrets", "notes")
    }


def _missing(session: Dict[str, Any]) -> List[str]:
    draft = session.get("draft") or {}
    role = session.get("role")
    if role == "lost":
        missing = [k for k in ("category", "location", "time") if not draft.get(k)]
        if not any(draft.get(k) for k in ("color", "brand", "marks")):
            missing.append("distinguishing")
        return missing
    if role == "found":
        missing = [k for k in ("category", "location", "time") if not draft.get(k)]
        if not draft.get("secret_detail") and not draft.get("no_secret"):
            missing.append("secret_detail")
        if not draft.get("custody"):
            missing.append("custody")
        return missing
    return ["role"]


def _public_candidates(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": c.get("id"),
            "title": c.get("title"),
            "location": c.get("location"),
            "time_text": c.get("time_text"),
            "has_marks": bool(c.get("has_marks") or c.get("marks")),
            "score": c.get("score"),
            "reasons": c.get("reasons") or [],
        }
        for c in (session.get("candidates") or [])[:3]
    ]


def _state(session: Dict[str, Any]) -> Dict[str, Any]:
    verification = session.get("verification") or {}
    questions = verification.get("questions") or []
    state = {
        "身份": session.get("role") or "未确定",
        "阶段": session.get("phase") or "intake",
        "草稿": _public_draft(session.get("draft") or {}),
        "还缺": _missing(session),
        "案件ID": session.get("case_id"),
        "候选": _public_candidates(session),
        "推荐": (session.get("last_search") or {}).get("recommendation"),
        "上一轮问题": session.get("last_question") or "",
    }
    if verification:
        state["核验"] = {
            "case_id": verification.get("case_id"),
            "index": verification.get("index", 0),
            "total": len(questions),
            "status": verification.get("status"),
        }
    if session.get("claim"):
        state["认领凭证"] = {
            "code": session["claim"].get("code"),
            "location": session["claim"].get("location"),
        }
    return state


SYSTEM_PROMPT = """你是「小寻」，校园失物招领 Agent。你像一位耐心、机灵的校园服务台老师，不是表单，也不是搜索框。

工作原则：
1. 每收到用户消息，先调用 update_draft 记录所有明确信息；用户没说的不要猜。
2. 如果用户没有回答你上一个问题，而是补充了别的信息，按实际内容更新草稿，不要硬塞进上一个字段。
3. 信息不足时，只问一个最关键的问题。一次只问一件事，最多两句话。
4. 失主流程：update_draft → 信息够了先 create_lost_case（若你忘了，search 会自动补建）→ search_found_cases → 根据 recommendation 决定追问、核验或告知暂无匹配。
   - strong：可以直接 start_verification；
   - possible：先追问一个最能区分的特征，再搜一次；
   - weak/none：不要硬匹配，告诉用户目前没有足够吻合的记录，建议补充线索。
5. 拾得流程：update_draft → 追问一个只有主人知道的私密细节 → 问清保管地点 → create_found_case。
   私密细节只放进 secret_detail，绝对不要放进 marks；marks 只放拾得者主动愿意公开的特征。
   只要用户给了一个可用于核对的具体细节（贴纸、划痕、刻字、内部颜色、数量、品牌等），就足够建一个核验点，直接 create_found_case，不要再追问第二个细节。
6. 核验流程：start_verification → 用户回答后 submit_verification_answer → 按返回的 next_question 继续问；passed 后必须调用 finalize_claim。
7. 安全边界：未 finalize_claim 成功前，绝不能透露物品内部细节、私密标记、保管点、拾得者联系方式。不要向用户展示工具名、JSON、字段名。
7.1 核验阶段：如果用户在回答核验问题，只调用 submit_verification_answer；如果用户其实在补充公开特征（品牌/颜色/地点/公开挂件），先 update_draft，然后重新问当前核验问题，不要把公开特征提交成答案。不要修改核验细节。
7.2 即使核验通过，也不要复述私密细节；只给认领码和领取地点。
8. 核验失败或记录缺少核验细节时，调用 escalate_to_staff，不要和用户争辩。
9. 工具返回 ok=false 时，按 missing 字段继续问，或如实告诉用户；不要编造工具结果。
10. 最终回答用自然中文，简洁、有同理心。不要用“请填写/请提供字段”这种表单词。
11. 内部推理也尽量用简短中文，不要长篇英文独白。
12. 不要承诺系统没有的能力：不要说“会通知你/会联系你/会帮你留意”；转人工只说“已转服务台人工复核，可携带学生证前往服务台”。
13. 涉及日期、时间、地点、案件编号、认领码，必须逐字使用工具返回值；不要自己换算日期或编造。
14. 互不依赖的工具可以在同一次回复里一起调用（如 set_role + update_draft、create_lost_case + search_found_cases），减少来回。

当前会话状态（JSON）：
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


# ------------------------------------------------------------------ 工具执行

def _ok(message: str, **data) -> Dict[str, Any]:
    return {"ok": True, "message": message, **data}


def _fail(error: str, message: str, **data) -> Dict[str, Any]:
    return {"ok": False, "error": error, "message": message, **data}


def _tool_set_role(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    role = args.get("role")
    if role not in ("lost", "found"):
        return _fail("invalid_role", "身份只能是 lost 或 found")
    current = session.get("role")
    if current and current != role and session.get("case_id"):
        return _fail("role_locked", "当前会话已经登记过案件，若要切换身份请开新会话")
    session["role"] = role
    return _ok("身份已记录", role=role)


def _is_private_mark(mark: str, secret: str) -> bool:
    """公开标记和私密细节高度重叠时，按私密处理，避免提前泄露。"""
    if not mark or not secret:
        return False
    a, b = matcher.compact(mark), matcher.compact(secret)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    ga, gb = nlu.char_bigrams(a), nlu.char_bigrams(b)
    if not ga or not gb:
        return False
    return len(ga & gb) / len(ga | gb) >= 0.42


def _clean_marks(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return nlu._dedupe_marks([nlu._clean_mark(str(v)) for v in value if str(v).strip()])


def _tool_update_draft(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    verifying = session.get("phase") == "verifying"
    if verifying and args.get("secret_detail"):
        return _fail(
            "verification_in_progress",
            "核验阶段不要修改核验细节；如果用户在补充公开特征，只更新公开字段",
        )
    draft = session.setdefault("draft", {})
    role = session.get("role")
    changed = []
    for key in ("color", "brand", "location", "time", "custody", "contact", "custom_name"):
        if key not in args:
            continue
        value = args.get(key)
        if not isinstance(value, (str, int, float)):
            continue
        value = str(value).strip()
        if not value:
            continue
        if key == "time" and len(value) > 30:
            # 明显的“答非所问”：不要把一段细节描述硬塞进时间
            if role == "found" and any(w in value for w in ["贴", "刻", "划痕", "内部", "里面", "背面", "底部"]):
                draft["secret_detail"] = value
                changed.append("secret_detail")
            continue
        draft[key] = value
        changed.append(key)

    if "category" in args:
        raw = str(args.get("category") or "").strip()
        if raw:
            category = matcher.category_of(raw, fallback="其他")
            draft["category"] = category
            changed.append("category")
            if category == "其他" and raw != "其他" and not draft.get("custom_name"):
                draft["custom_name"] = raw
                changed.append("custom_name")

    if "marks" in args:
        marks = _clean_marks(args.get("marks"))
        secret_text = str(args.get("secret_detail") or draft.get("secret_detail") or "")
        if verifying:
            # 核验阶段：用户可能补充的是公开特征，不能把它当成核验答案
            questions = (session.get("verification") or {}).get("questions") or []
            secret_text += " " + " ".join(
                str(q.get("answer") or "") + " " + " ".join(str(k) for k in (q.get("keywords") or []))
                for q in questions
            )
        if secret_text and (session.get("role") == "found" or verifying):
            marks = [m for m in marks if not _is_private_mark(m, secret_text)]
        if marks:
            existing = list(draft.get("marks") or [])
            for mark in marks:
                if mark not in existing:
                    existing.append(mark)
            draft["marks"] = nlu._dedupe_marks(existing)
            changed.append("marks")

    if "secret_detail" in args:
        value = str(args.get("secret_detail") or "").strip()
        if value:
            # 不依赖 set_role 与 update_draft 的调用顺序
            draft["secret_detail"] = value
            # 如果之前误把同一个细节放进了公开 marks，这里要撤回
            existing = list(draft.get("marks") or [])
            draft["marks"] = [m for m in existing if not _is_private_mark(m, value)]
            changed.append("secret_detail")

    if args.get("no_secret") is True:
        draft["no_secret"] = True
        changed.append("no_secret")

    note = ""
    if verifying:
        note = "核验阶段：如果用户没有回答当前核验问题，请问回当前问题，不要把公开特征提交成答案。"
    return _ok(
        "草稿已更新" if changed else "没有新的公开信息",
        updated=sorted(set(changed)),
        draft=_public_draft(draft),
        missing=_missing(session),
        note=note,
    )


def _tool_create_lost(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    if session.get("role") != "lost":
        return _fail("wrong_role", "当前不是失主身份，不能创建寻物案件")
    if session.get("case_id"):
        case = db.get_case(session["case_id"])
        return _ok("寻物案件已存在", case=_case_brief(case), case_id=session["case_id"])
    draft = session.get("draft") or {}
    missing = [k for k in ("category", "location", "time") if not draft.get(k)]
    if missing:
        return _fail("missing_fields", "还缺少必要信息", missing=missing)
    case = tools.create_lost(session)
    session["last_case"] = case
    return _ok("寻物案件已创建", case=_case_brief(case), case_id=case["id"])


def _tool_create_found(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    if session.get("role") != "found":
        return _fail("wrong_role", "当前不是拾得者身份，不能创建拾得案件")
    if session.get("case_id"):
        case = db.get_case(session["case_id"])
        return _ok("拾得案件已存在", case=_case_brief(case), case_id=session["case_id"])
    draft = session.get("draft") or {}
    missing = [k for k in ("category", "location", "time") if not draft.get(k)]
    if not draft.get("secret_detail") and not draft.get("no_secret"):
        missing.append("secret_detail")
    if not draft.get("custody"):
        missing.append("custody")
    if missing:
        return _fail("missing_fields", "还缺少必要信息", missing=missing)
    # 创建前最后一道隐私闸：和核验细节重叠的公开标记一律撤回
    secret = str(draft.get("secret_detail") or "")
    if secret:
        draft["marks"] = [m for m in (draft.get("marks") or []) if not _is_private_mark(m, secret)]
    case = tools.create_found(session)
    session["last_case"] = case
    return _ok(
        "拾得案件已创建，核验细节已隐藏保存",
        case=_case_brief(case),
        case_id=case["id"],
    )


def _tool_search(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    if session.get("role") != "lost":
        return _fail("wrong_role", "只有失主需要检索拾得记录")
    # 检索前补建寻物案件，保证管理台能审计；信息不够时不强建
    if not session.get("case_id"):
        draft = session.get("draft") or {}
        if all(draft.get(k) for k in ("category", "location", "time")):
            case = tools.create_lost(session)
            session["last_case"] = case
    result = matching.search(session)
    session["last_search"] = result
    session["candidates"] = result.get("candidates") or []
    if session.get("phase") not in ("verifying", "done"):
        session["phase"] = "searching"
    return _ok(
        f"检索到 {result.get('count', 0)} 条候选",
        recommendation=result.get("recommendation"),
        count=result.get("count", 0),
        margin=result.get("margin", 0),
        candidates=result.get("candidates") or [],
        case_id=session.get("case_id"),
    )


def _tool_start_verification(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    if session.get("role") != "lost":
        return _fail("wrong_role", "只有失主需要做归属核验")
    candidates = session.get("candidates") or []
    if not candidates:
        return _fail("no_candidates", "还没有候选记录，请先检索")
    case_id = str(args.get("case_id") or "").strip()
    if not case_id:
        case_id = str(candidates[0].get("id"))
    if case_id not in {str(c.get("id")) for c in candidates}:
        return _fail("invalid_case", "只能对检索结果中的候选发起核验")
    case = db.get_case(case_id)
    if not case:
        return _fail("case_not_found", "这条记录不存在")
    if case.get("status") in ("claimed", "returned"):
        return _fail("already_claimed", "这条记录已经完成认领")
    verification = policy.start_verification(session, case)
    if not verification:
        return _fail(
            "no_secret",
            "这条记录缺少可供核验的私密细节，建议转人工复核",
        )
    tools.mark_verifying(session)
    session["phase"] = "verifying"
    question = policy.current_question(session) or {}
    session["last_question"] = str(question.get("q") or "")
    return _ok(
        "已开始核验",
        case_id=case_id,
        question=session["last_question"],
        index=verification.get("index", 0) + 1,
        total=len(verification.get("questions") or []),
    )


def _tool_submit_answer(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    verification = session.get("verification") or {}
    if not verification:
        return _fail("no_verification", "当前没有进行中的核验")
    if verification.get("status") in ("passed", "failed", "escalated"):
        return _fail("verification_closed", "核验已经结束")
    answer = str(args.get("answer") or "").strip()
    if not answer:
        return _fail("empty_answer", "回答不能为空")
    result = policy.evaluate_answer(session, answer, require_llm=True)
    status = result.get("status")
    if status == "next_question":
        nxt = result.get("next") or {}
        session["last_question"] = str(nxt.get("q") or "")
        return _ok(
            "继续下一个核验点",
            status=status,
            question=session["last_question"],
            index=verification.get("index", 0) + 1,
            total=len(verification.get("questions") or []),
            score=result.get("score", 0),
        )
    verification["status"] = status
    session["last_question"] = ""
    if status == "passed":
        return _ok(
            "核验通过，可以生成认领凭证",
            status=status,
            ratio=result.get("ratio", 0),
        )
    # 失败也落一条 rejected 认领记录，方便管理台审计
    tools.finalize_claim(
        session,
        False,
        verification.get("answers") or [],
        float(result.get("ratio") or 0),
    )
    return _ok(
        "核验未通过",
        status="failed",
        ratio=result.get("ratio", 0),
        note="细节对不上，不能直接放行；应转人工复核。",
    )


def _tool_finalize_claim(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    if session.get("claim"):
        return _ok("认领凭证已生成", **session["claim"])
    verification = session.get("verification") or {}
    if verification.get("status") != "passed":
        return _fail("not_verified", "核验尚未通过，不能生成认领码")
    claim = tools.finalize_claim(
        session,
        True,
        verification.get("answers") or [],
        float(verification.get("ratio") or 0),
    )
    ticket = tools.ticket_card(session) or {}
    session["claim"] = ticket
    return _ok(
        "认领凭证已生成",
        code=claim.get("code"),
        location=ticket.get("location"),
        contact=ticket.get("contact"),
        ticket=ticket,
    )


def _tool_escalate(session: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    reason = str(args.get("reason") or "需要人工复核")[:200]
    db.add_event(session.get("id"), session.get("case_id"), "manual_review", reason)
    verification = session.get("verification")
    if verification:
        verification["status"] = "escalated"
    session["phase"] = "open"
    session["last_question"] = ""
    return _ok("已转人工复核", reason=reason)


def _case_brief(case: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not case:
        return {}
    return {
        "id": case.get("id"),
        "title": case.get("title"),
        "location": case.get("location"),
        "time_text": case.get("time_text"),
    }


def _execute_tool(session: Dict[str, Any], name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    dispatch = {
        "set_role": _tool_set_role,
        "update_draft": _tool_update_draft,
        "create_lost_case": _tool_create_lost,
        "create_found_case": _tool_create_found,
        "search_found_cases": _tool_search,
        "start_verification": _tool_start_verification,
        "submit_verification_answer": _tool_submit_answer,
        "finalize_claim": _tool_finalize_claim,
        "escalate_to_staff": _tool_escalate,
    }
    func = dispatch.get(name)
    if not func:
        return _fail("unknown_tool", f"未知工具：{name}")
    return func(session, args)


# ------------------------------------------------------------------ 轨迹与响应

def _safe_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in args.items() if k not in ("secret_detail",)}


def _reasoning_sensitive(session: Dict[str, Any], text: str) -> bool:
    """推理里如果出现核验细节，就不往用户可见的轨迹里放。"""
    if not text:
        return False
    parts: List[str] = []
    draft = session.get("draft") or {}
    if draft.get("secret_detail"):
        parts.append(str(draft["secret_detail"]))
    for q in (session.get("verification") or {}).get("questions") or []:
        if q.get("answer"):
            parts.append(str(q["answer"]))
        parts.extend(str(k) for k in (q.get("keywords") or []))
    for part in parts:
        for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{3,}", part):
            if token and token in text:
                return True
    return False


def _tool_trace(name: str, args: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    if name == "update_draft":
        public = {k: v for k, v in args.items() if k not in ("secret_detail", "no_secret")}
        detail = "记录：" + "、".join(f"{k}={v}" for k, v in public.items() if v not in (None, "", []))
        if args.get("secret_detail"):
            detail += "、secret_detail=（已隐藏）"
    elif name == "search_found_cases":
        top = (result.get("candidates") or [{}])[0]
        detail = (
            f"{result.get('count', 0)} 条候选，推荐 {result.get('recommendation')}；"
            f"Top1 {top.get('title')} {top.get('score')}"
        )
    elif name == "start_verification":
        detail = f"开始核验：第 {result.get('index')}/{result.get('total')} 问"
    elif name == "submit_verification_answer":
        detail = f"核验结果：{result.get('status')}；得分 {result.get('score', result.get('ratio', 0))}"
    elif name == "finalize_claim":
        detail = f"生成认领码：{result.get('code')}"
    elif name == "escalate_to_staff":
        detail = f"转人工：{result.get('reason')}"
    elif name == "set_role":
        detail = f"身份：{result.get('role')}"
    elif name == "create_lost_case":
        detail = f"创建寻物案件：{(result.get('case') or {}).get('id')}"
    elif name == "create_found_case":
        detail = f"创建拾得案件：{(result.get('case') or {}).get('id')}"
    else:
        detail = result.get("message") or ""
    if not result.get("ok"):
        detail = f"失败：{result.get('message')}；缺 {result.get('missing')}"
    return {
        "type": "tool",
        "title": name,
        "detail": detail,
        "data": {"args": _safe_args(args), "ok": result.get("ok")},
    }


def _public_session(session: Dict[str, Any]) -> Dict[str, Any]:
    verification = session.get("verification") or {}
    questions = verification.get("questions") or []
    return {
        "id": session["id"],
        "role": session.get("role"),
        "phase": session.get("phase"),
        "draft": _public_draft(session.get("draft") or {}),
        "case_id": session.get("case_id"),
        "turn": session.get("turn", 0),
        "candidates": _public_candidates(session),
        "recommendation": (session.get("last_search") or {}).get("recommendation"),
        "verification": {
            "case_id": verification.get("case_id"),
            "index": verification.get("index", 0),
            "total": len(questions),
            "status": verification.get("status"),
        } if verification else None,
        "claim": session.get("claim"),
    }


def _cards(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    if session.get("claim"):
        return [session["claim"]]
    cards: List[Dict[str, Any]] = []
    for item in (session.get("candidates") or [])[:3]:
        cards.append(
            {
                "type": "candidate",
                "id": item.get("id"),
                "title": item.get("title"),
                "subtitle": " · ".join(
                    str(x) for x in (item.get("location"), item.get("time_text")) if x
                ),
                "score": item.get("score", 0),
                "reasons": item.get("reasons") or [],
            }
        )
    if not cards and session.get("last_case"):
        case = session["last_case"]
        cards.append(
            {
                "type": "case_created",
                "id": case.get("id"),
                "title": case.get("title"),
                "subtitle": "拾得登记" if case.get("role") == "found" else "寻物登记",
                "status": case.get("status"),
            }
        )
    return cards


def _error_response(
    session: Dict[str, Any],
    trace: List[Dict[str, Any]],
    exc: Exception,
) -> Dict[str, Any]:
    detail = getattr(exc, "detail", "") or str(exc)
    trace.append(
        {
            "type": "error",
            "title": "模型调用失败",
            "detail": detail[:500],
        }
    )
    return {
        "reply": "抱歉，模型服务暂时不可用，我这边没法继续处理。请稍后重试。",
        "trace": trace,
        "session": _public_session(session),
        "cards": [],
        "quick_replies": [],
        "error": {
            "code": "LLM_UNAVAILABLE",
            "message": str(exc),
            "detail": detail[:500],
        },
    }


# ------------------------------------------------------------------ 主循环

def handle(session_id: Optional[str], text: str) -> Dict[str, Any]:
    session = store.get_or_create(session_id)
    session["turn"] += 1
    text = (text or "").strip()
    session["history"].append({"role": "user", "content": text})
    trace: List[Dict[str, Any]] = []

    if not llm.available:
        return _error_response(session, trace, LLMError("模型服务不可用或已熔断", llm.last_error))

    messages = _build_messages(session)
    for step in range(MAX_STEPS):
        try:
            message = llm.chat_message(messages, tools=TOOLS, temperature=0.25)
        except LLMError as exc:
            return _error_response(session, trace, exc)

        reasoning = message.get("reasoning_content") or ""
        if (
            SHOW_REASONING
            and reasoning
            and not any(item.get("type") == "llm" for item in trace)
        ):
            detail = " ".join(str(reasoning).split())[:320]
            if _reasoning_sensitive(session, detail):
                detail = "[已隐藏：推理中包含核验细节]"
            trace.append(
                {
                    "type": "llm",
                    "title": "LLM 推理（节选）",
                    "detail": detail,
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
                fn = (call.get("function") or {})
                name = fn.get("name") or ""
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                try:
                    result = _execute_tool(session, name, args)
                except LLMError as exc:
                    return _error_response(session, trace, exc)
                trace.append(_tool_trace(name, args, result))
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
            return _error_response(
                session, trace, LLMError("模型返回了空回复", str(message)[:300])
            )
        if session.get("claim") and session["claim"].get("code") not in content:
            content += f"\n\n认领码：{session['claim'].get('code')}"
        session["history"].append({"role": "assistant", "content": content})
        return {
            "reply": content,
            "trace": trace,
            "session": _public_session(session),
            "cards": _cards(session),
            "quick_replies": [],
            "error": None,
        }

    return _error_response(
        session,
        trace,
        LLMError("工具调用步数超出上限", f"steps={MAX_STEPS}"),
    )
