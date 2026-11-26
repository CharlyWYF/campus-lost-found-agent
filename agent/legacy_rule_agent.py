"""LEGACY：旧的规则状态机 Agent，仅保留作对比实验，不参与当前运行路径。

当前入口是 agent/react_agent.py（LLM 自主规划 + 原生 tool calling）。
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from . import composer, db, matcher, nlu, policy, tools
from .llm import llm

INTENT_LABELS = {
    "report_lost": "失物登记",
    "report_found": "拾得登记",
    "search": "查询",
    "claim": "认领",
    "confirm": "确认",
    "other": "其他",
}

SLOT_LABELS = {
    "category": "类别",
    "color": "颜色",
    "brand": "品牌",
    "location": "地点",
    "time": "时间",
    "marks": "标记",
    "custody": "保管点",
    "contact": "联系方式",
    "secret_detail": "核验细节",
    "distinguishing": "区分特征",
}

UNKNOWN_WORDS = ["不记得", "不知道", "忘了", "没注意", "不清楚", "记不清", "没印象"]


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
                "asked": [],
                "ask_counts": {},
                "unknown_slots": [],
                "awaiting": None,
                "candidates": [],
                "verification": None,
                "verification_result": None,
                "history": [],
                "turn": 0,
                "dirty": False,
            }
            self._data[new_id] = session
            return session

    def reset(self) -> None:
        with self._lock:
            self._data.clear()


store = SessionStore()


def _trace(trace: List[Dict[str, Any]], kind: str, title: str, detail: str, **data) -> None:
    item = {"type": kind, "title": title, "detail": detail}
    if data:
        item["data"] = data
    trace.append(item)


def _replace_trace(trace: List[Dict[str, Any]], kind: str, detail: str) -> None:
    for item in reversed(trace):
        if item.get("type") == kind:
            item["detail"] = detail
            return


def _looks_like_detail(text: str) -> bool:
    if any(w in text for w in ["补充", "还有", "细节", "另外", "键", "刻", "贴", "划痕", "图案", "颜色", "品牌", "型号", "材质", "尺寸"]):
        return True
    return len(text) <= 12 and "?" not in text and "？" not in text


def _is_public_update(text: str, slots: Dict[str, Any]) -> bool:
    """核验阶段用户补充公开属性，不应被当成核验回答打分。"""
    if ("品牌" in text or "型号" in text or "牌子" in text) and slots.get("brand"):
        return True
    if ("地点" in text or "位置" in text) and slots.get("location"):
        return True
    if "时间" in text and slots.get("time"):
        return True
    return False


AWAIT_ALLOWED = {
    # 用户经常在回答一个问题时顺带补充别的公开信息，这些都应该保留
    "category": {"category", "location", "color", "brand", "marks"},
    "location": {"location", "category", "color", "brand", "marks"},
    "time": {"time", "category", "location", "color", "brand", "marks"},
    "custody": {"custody"},
    "marks": {"marks", "color", "brand"},
    "distinguishing": {"color", "brand", "marks"},
}


def _slot_summary(slots: Dict[str, Any]) -> str:
    parts = []
    for key, value in slots.items():
        if key in ("secret_detail", "secrets"):
            continue
        if not value:
            continue
        label = SLOT_LABELS.get(key, key)
        if isinstance(value, list):
            value = "、".join(str(v) for v in value)
        parts.append(f"{label}={value}")
    return "；".join(parts)


def _is_unknown(text: str) -> bool:
    return any(word in text for word in UNKNOWN_WORDS)


def _is_negative(text: str) -> bool:
    t = text.strip()
    return any(word in t for word in ["没有", "没", "无", "并无"]) and len(t) <= 20


def _is_claim_phrase(text: str) -> bool:
    t = text.strip()
    return len(t) <= 12 and any(word in t for word in ["是我的", "就是我的", "我来认领", "认领"])


def _has_secret_marker(text: str) -> bool:
    return any(
        word in text
        for word in ["贴", "刻", "划痕", "磨损", "内部", "里面", "内侧", "背面", "底部", "锁屏", "壁纸", "图案", "挂件", "贴纸", "写着"]
    )


def _looks_like_time(text: str) -> bool:
    if nlu.extract_time(text):
        return True
    return bool(__import__("re").search(r"\d|上午|下午|中午|晚上|凌晨|昨天|今天|前天|周|点", text))


def _looks_like_location(text: str) -> bool:
    if matcher.location_of(text)["building"]:
        return True
    return bool(__import__("re").search(r"\d{3}|楼|馆|室|食堂|操场|宿舍", text))


def _looks_like_category(text: str) -> bool:
    return bool(matcher.category_of(text, fallback=""))


def _public_session(session: Dict[str, Any]) -> Dict[str, Any]:
    verification = session.get("verification") or {}
    questions = verification.get("questions") or []
    return {
        "id": session["id"],
        "role": session.get("role"),
        "phase": session.get("phase"),
        "draft": {
            k: v for k, v in (session.get("draft") or {}).items()
            if k not in ("secret_detail", "secrets")
        },
        "case_id": session.get("case_id"),
        "awaiting": session.get("awaiting"),
        "turn": session.get("turn", 0),
        "candidates": [
            {
                "id": c["candidate"].get("id"),
                "title": c["candidate"].get("title"),
                "score": c.get("score"),
                "reasons": c.get("reasons", []),
            }
            for c in (session.get("candidates") or [])[:3]
        ],
        "verification": {
            "case_id": verification.get("case_id"),
            "index": verification.get("index", 0),
            "total": len(questions),
            "status": verification.get("status"),
        } if verification else None,
    }


def _quick_replies(session: Dict[str, Any], kind: str) -> List[str]:
    if kind == "ask_role":
        return ["我丢东西了", "我捡到东西了"]
    if kind == "ask_marks":
        return ["有，挂着一个黄色小鸭子挂件", "没有特别的标记", "不记得了"]
    if kind == "ask_location":
        return ["图书馆三楼", "教学楼 204", "一食堂"]
    if kind == "ask_time":
        return ["昨天下午", "今天上午", "记不清了"]
    if kind == "no_match":
        return ["我再补充一些细节", "帮我挂寻物启事"]
    if kind == "passed":
        return ["查看认领凭证", "重新开始"]
    if kind == "failed":
        return ["我再想想", "怎么联系服务台"]
    if kind == "done_found":
        return ["还有东西要补充", "重新开始"]
    return []


def _handle_awaiting(
    session: Dict[str, Any],
    text: str,
    slots: Dict[str, Any],
    trace: List[Dict[str, Any]],
) -> None:
    """把用户对上一个追问的回答落到 draft 里。"""
    awaiting = session.get("awaiting")
    if not awaiting or awaiting in ("verify", "role"):
        return
    if _is_claim_phrase(text):
        # “是我的”是认领意愿，不是物品特征，不能写进 marks/color
        return

    if awaiting == "secret_detail":
        if _is_unknown(text):
            session["draft"]["no_secret"] = True
            session["draft"].pop("secret_detail", None)
            _trace(trace, "observe", "核验细节缺失", "用户不记得只有主人知道的细节，改为人工核验")
            session["awaiting"] = None
            session["dirty"] = True
            return
        # “品牌是 acer”属于公开属性，不应该被藏进核验细节里
        public = nlu.rule_extract(text)["slots"]
        secret_markers = ["贴", "刻", "划痕", "磨损", "内部", "里面", "内侧", "背面", "底部", "锁屏", "壁纸", "图案", "挂件", "贴纸", "写"]
        looks_public = (
            bool(public.get("brand"))
            or "品牌" in text or "牌子" in text or "型号" in text
        ) and not any(marker in text for marker in secret_markers)
        if looks_public:
            for key in ("brand", "color"):
                if public.get(key):
                    slots[key] = public[key]
            if public.get("brand"):
                session["notice"] = f"品牌 {public['brand']} 我记下了。"
            else:
                session["notice"] = "这个信息我记下了。"
            _trace(trace, "observe", "补充公开属性", "这属于公开信息，已补进物品档案；还需要一个不公开的核验细节")
            return
        session["draft"]["secret_detail"] = text
        _trace(trace, "observe", "补充核验细节", "已记录一个不公开的核验点")
        session["awaiting"] = None
        session["dirty"] = True
        return

    if slots.get(awaiting):
        session["awaiting"] = None
        return

    if _is_unknown(text):
        if awaiting == "marks":
            session["draft"]["marks"] = []
        else:
            if awaiting not in session["unknown_slots"]:
                session["unknown_slots"].append(awaiting)
        session["awaiting"] = None
        return

    if awaiting in ("time", "location", "category") and not slots.get(awaiting):
        # 答非所问：不要把“底部贴纸”硬塞进“时间”，也不要把“图书馆三楼”塞进“类别”
        if _has_secret_marker(text) and session.get("role") == "found":
            session["draft"]["secret_detail"] = text
            session["notice"] = "这个细节我记下了。"
            session["_secret_routed"] = True
            _trace(
                trace,
                "observe",
                "语义归位",
                f"这更像核验细节，先记下；{SLOT_LABELS.get(awaiting, awaiting)}还没确认",
            )
            return
        if awaiting == "time" and not _looks_like_time(text):
            return
        if awaiting == "location" and not _looks_like_location(text):
            return
        if awaiting == "category" and not _looks_like_category(text):
            return

    if _is_negative(text) and awaiting == "marks":
        session["draft"]["marks"] = []
        session["awaiting"] = None
        return

    if len(text) <= 40:
        value: Any = text.strip()
        if awaiting == "marks":
            marks = nlu.extract_marks(text)
            value = marks or [text.strip()]
            slots["marks"] = value
        elif awaiting == "category":
            # 用户可能先回答了地点；不要把“在艺术楼”当成物品类别
            if matcher.location_of(text)["building"] and not matcher.category_of(text, fallback=""):
                slots["location"] = nlu.extract_location(text) or text.strip()
                return
            category = matcher.category_of(text, fallback="其他")
            slots["category"] = category
            if category == "其他":
                session["draft"]["custom_name"] = text.strip()
            else:
                session["draft"].pop("custom_name", None)
        elif awaiting == "distinguishing":
            if matcher.color_of(text):
                slots["color"] = matcher.COLOR_LABELS.get(matcher.color_of(text), matcher.color_of(text))
            if nlu.extract_marks(text):
                slots["marks"] = nlu.extract_marks(text)
            if nlu.extract_brand(text):
                slots["brand"] = nlu.extract_brand(text)
        elif awaiting == "location":
            slots["location"] = nlu.extract_location(text) or text.strip()
        elif awaiting == "time":
            slots["time"] = nlu.extract_time(text) or text.strip()
        elif awaiting == "custody":
            slots["custody"] = text.strip()
        session["awaiting"] = None


def handle(session_id: Optional[str], text: str) -> Dict[str, Any]:
    session = store.get_or_create(session_id)
    text = (text or "").strip()
    if not text:
        return {
            "reply": "我在。你可以直接描述丢了什么，或者捡到了什么。",
            "trace": [],
            "session": _public_session(session),
            "cards": [],
            "quick_replies": ["我丢东西了", "我捡到东西了"],
        }

    session["turn"] += 1
    session["history"].append({"role": "user", "content": text})
    trace: List[Dict[str, Any]] = []

    understanding = nlu.understand(
        text,
        session.get("draft") or {},
        session["history"][:-1],
        session.get("awaiting") or "",
    )
    intent = understanding["intent"]
    slots = dict(understanding.get("slots") or {})
    source_label = {"llm+rule": "LLM+规则", "llm": "LLM", "rule": "规则"}.get(
        understanding.get("source"), understanding.get("source")
    )
    _trace(
        trace,
        "perceive",
        "理解",
        f"意图：{INTENT_LABELS.get(intent, intent)}；抽取：{_slot_summary(slots) or '暂无新字段'}（{source_label}）",
        intent=intent,
        slots=slots,
    )

    awaiting_before = session.get("awaiting")
    phase_before = session.get("phase")
    preface = ""

    # 核验阶段：如果用户补充“品牌/型号”等公开属性，应该重新检索，而不是判为答错
    if phase_before == "verify" and _is_public_update(text, slots):
        session["draft"] = nlu.merge_slots(session.get("draft") or {}, slots)
        if session.get("case_id") and session.get("role") == "lost":
            tools.update_lost(session)
        session["verification"] = None
        session["verification_result"] = None
        session["phase"] = "search"
        session["dirty"] = True
        session["candidates"] = []
        session["awaiting"] = None
        preface = "这条公开信息我补进寻物记录了，重新匹配一下。\n"
        _trace(trace, "observe", "补充公开属性", "核验阶段收到品牌/型号补充，更新档案后重新检索")
    else:
        _handle_awaiting(session, text, slots, trace)

    # 这句话已被判定为“核验细节”，不要让它里面的“蓝色”覆盖物品本身的颜色
    secret_routed = bool(session.pop("_secret_routed", False))
    if secret_routed:
        slots = {k: v for k, v in slots.items() if k == "secret_detail"}

    # 用户说“不记得”，不要把同一句话里抽出的“什么贴纸”当成新标记
    if awaiting_before and _is_unknown(text):
        slots.pop(awaiting_before, None)
        if awaiting_before in ("marks", "distinguishing"):
            for key in ("marks", "color", "brand"):
                slots.pop(key, None)

    # 核验阶段的回答只用于打分，不能被当成新的登记字段
    if phase_before == "verify" and not preface:
        slots = {}
        _replace_trace(trace, "perceive", "核验回答，仅用于归属判断，不更新登记字段")
    elif awaiting_before in AWAIT_ALLOWED:
        allowed = AWAIT_ALLOWED[awaiting_before]
        slots = {k: v for k, v in slots.items() if k in allowed}
    elif awaiting_before == "secret_detail":
        if session.get("awaiting") == "secret_detail":
            # 用户回答的是品牌/型号等公开信息，保留并继续追问核验细节
            slots = {k: v for k, v in slots.items() if k in ("brand", "color")}
        else:
            slots = {k: v for k, v in slots.items() if k == "secret_detail"}
    # “只有主人知道的细节”里可能夹带颜色等信息，不能覆盖公开字段；
    # 但如果用户回答的是品牌/型号，仍然要保留为公开字段。
    if awaiting_before == "secret_detail" and session.get("awaiting") != "secret_detail":
        slots = {k: v for k, v in slots.items() if k == "secret_detail"}
        _replace_trace(trace, "perceive", "已收到核验细节，转为私密字段保存")

    # 身份判定：只在还没有案件时允许切换
    if not session.get("role"):
        if intent == "report_found" or "捡" in text or "拾" in text:
            session["role"] = "found"
        elif intent == "report_lost" or "丢" in text or "遗失" in text:
            session["role"] = "lost"
        elif intent == "claim":
            session["role"] = "lost"
        else:
            session["awaiting"] = "role"
            reply = (
                "你好，我是校园失物招领助手「小寻」。"
                "你是丢了东西，还是捡到了东西？直接用自己的话告诉我就行。"
            )
            session["history"].append({"role": "assistant", "content": reply})
            return {
                "reply": reply,
                "trace": trace,
                "session": _public_session(session),
                "cards": [],
                "quick_replies": _quick_replies(session, "ask_role"),
            }
    elif session.get("role") == "lost" and intent == "report_found" and not session.get("case_id"):
        session["role"] = "found"
    elif session.get("role") == "found" and intent == "report_lost" and not session.get("case_id"):
        session["role"] = "lost"

    # 合并槽位
    before = dict(session.get("draft") or {})
    session["draft"] = nlu.merge_slots(before, slots)
    changed = session["draft"] != before
    if (
        phase_before != "verify"
        and not secret_routed
        and not changed
        and session.get("role")
        and _looks_like_detail(text)
        and not _is_claim_phrase(text)
        and not _is_unknown(text)
    ):
        notes = list(session["draft"].get("notes") or [])
        if text not in notes:
            notes.append(text)
            session["draft"]["notes"] = notes[-5:]
            changed = True
    if changed:
        session["dirty"] = True
        if session.get("case_id") and session.get("role") == "lost":
            tools.update_lost(session)

    # 用户明确说“是我的”：用现有信息重新检索，进入核验
    if intent in ("claim", "confirm") and session.get("role") == "lost" and session.get("phase") != "verify":
        if session.get("case_id"):
            session["dirty"] = True

    # 核验中用户说“是我的/对”，不要当成答案打分，先把问题再说一遍
    if phase_before == "verify" and intent in ("claim", "confirm") and len(text) <= 12:
        question = policy.current_question(session) or {}
        reply = (
            "我理解，但为了对失主负责，还是要先核对一个只有主人知道的细节："
            + str(question.get("q") or "能再描述一下这件物品吗？")
        )
        _trace(trace, "plan", "澄清", "用户表达了认领意愿，但核验问题还没回答，先不进入打分")
        session["history"].append({"role": "assistant", "content": reply})
        return {
            "reply": reply,
            "trace": trace,
            "session": _public_session(session),
            "cards": [],
            "quick_replies": [],
        }

    reply = ""
    cards: List[Dict[str, Any]] = []
    quick: List[str] = []
    action_kind = ""

    for _ in range(5):
        action = policy.next_action(session)
        kind = action["type"]

        if kind == "ask_role":
            action_kind = "ask_role"
            session["awaiting"] = "role"
            reply = "你是丢了东西，还是捡到了东西？直接告诉我就行。"
            _trace(trace, "plan", "澄清身份", "先确定用户是失主还是拾得者，后续流程不同")
            break

        if kind == "ask":
            slot = action["slots"][0]
            action_kind = f"ask_{slot}"
            session["awaiting"] = slot
            if slot not in session["asked"]:
                session["asked"].append(slot)
            session["ask_counts"][slot] = session["ask_counts"].get(slot, 0) + 1
            base_question = action.get("question", "")
            notice = action.get("notice", "")
            reply = composer.phrase_question(
                session,
                slot,
                base_question,
                extra=action.get("extra", ""),
                must_include=notice,
            )
            engine = "LLM 措辞" if llm.available else "模板"
            _trace(
                trace,
                "plan",
                "主动追问",
                f"{action.get('reason', '信息不足')} → 追问「{SLOT_LABELS.get(slot, slot)}」；{engine}",
                slot=slot,
            )
            if llm.available:
                _trace(trace, "llm", "LLM 措辞", f"生成追问：{reply[:70]}")
            if action.get("results"):
                cards = [tools.candidate_card(x) for x in action["results"][:2]]
            quick = _quick_replies(session, f"ask_{slot}")
            break

        if kind == "create_lost":
            case = tools.create_lost(session)
            _trace(
                trace,
                "tool",
                "create_lost_case",
                f"已创建寻物案件 {case['id']}：{case['title']}",
                case_id=case["id"],
            )
            continue

        if kind == "create_found":
            case = tools.create_found(session)
            _trace(
                trace,
                "tool",
                "create_found_case",
                f"已创建拾得案件 {case['id']}：{case['title']}",
                case_id=case["id"],
            )
            action_kind = "done_found"
            if case.get("secrets"):
                privacy = (
                    "你补充的核验细节只有我看得到，不会展示给认领人；"
                    "失主来描述时，我会用它来核对。"
                )
            else:
                privacy = "这条记录暂时没有可核验的私密细节，认领时会转服务台人工核对。"
            reply = (
                f"登记完成 ✅ 案件编号 {case['id']}。\n"
                f"我记下了：{case['title']}，{case.get('location') or '地点待补充'}，"
                f"{case.get('time_text') or '时间待补充'}。\n" + privacy
            )
            cards = [{
                "type": "case_created",
                "id": case["id"],
                "title": case["title"],
                "subtitle": "拾得登记 · 等待失主描述",
                "status": "open",
            }]
            quick = _quick_replies(session, "done_found")
            break

        if kind == "update_found":
            tools.update_found(session)
            action_kind = "done_found"
            reply = "已经更新到这条拾得记录里了。还有其他要补充的吗？"
            quick = _quick_replies(session, "done_found")
            break

        if kind == "search":
            if not session.get("case_id"):
                reply = "我先帮你把信息整理成一条寻物记录。"
                continue
            session["search_count"] = session.get("search_count", 0) + 1
            results = tools.search_found(session)
            session["dirty"] = False
            session["candidates"] = results
            _trace(
                trace,
                "tool",
                "search_cases",
                f"检索到 {len(results)} 条反向记录；Top1 "
                f"{results[0]['score']:.2f}（{results[0]['candidate'].get('title')}）"
                if results else "未检索到反向记录",
                count=len(results),
            )
            after = policy.after_search(session, results)

            if after["type"] == "no_match":
                action_kind = "no_match"
                session["phase"] = "open"
                _trace(
                    trace,
                    "reflect",
                    "反思",
                    f"{after.get('reason')} → 不强行匹配，转为挂起并邀请补充线索",
                )
                attempt = session.get("search_count", 1)
                lead = "我用补充后的信息又查了一遍" if attempt > 1 else "我按现有信息查了一遍"
                reply = (
                    f"{lead}，暂时没有足够吻合的记录。"
                    "已经帮你把这条寻物信息挂出去了。\n"
                    "你可以再补充颜色、品牌、挂件/贴纸/刻字，或者更准确的时间地点，我再查一次。"
                )
                quick = _quick_replies(session, "no_match")
                break

            if after["type"] == "ask":
                slot = after["slots"][0]
                action_kind = f"ask_{slot}"
                session["awaiting"] = slot
                if slot not in session["asked"]:
                    session["asked"].append(slot)
                session["ask_counts"][slot] = session["ask_counts"].get(slot, 0) + 1
                base_question = after.get("question", "")
                notice = after.get("notice", "")
                reply = composer.phrase_question(
                    session,
                    slot,
                    base_question,
                    extra=after.get("extra", ""),
                    must_include=notice,
                )
                engine = "LLM 措辞" if llm.available else "模板"
                _trace(
                    trace,
                    "reflect",
                    "反思",
                    f"{after.get('reason')} → 再问一个能提高区分度的字段；{engine}",
                )
                if llm.available:
                    _trace(trace, "llm", "LLM 措辞", f"生成追问：{reply[:70]}")
                cards = [tools.candidate_card(x) for x in results[:2]]
                quick = _quick_replies(session, f"ask_{slot}")
                break

            if after["type"] == "verify":
                candidate = db.get_case(after["case_id"])
                if not candidate:
                    reply = "这条记录暂时不可用，我转给服务台核查一下。"
                    session["phase"] = "open"
                    break
                verification = policy.start_verification(session, candidate)
                if not verification:
                    reply = (
                        "我找到一条相似记录，但它缺少可供核验的私密细节。"
                        "为了稳妥，我先转给服务台人工核对，不直接开放认领。"
                    )
                    db.add_event(session["id"], candidate["id"], "manual_review", "缺少核验细节")
                    session["phase"] = "open"
                    break
                tools.mark_verifying(session)
                first = policy.current_question(session)
                action_kind = "verify"
                session["awaiting"] = "verify"
                _trace(
                    trace,
                    "reflect",
                    "反思",
                    f"{after.get('reason')} → 进入归属核验，只问不公开的细节",
                )
                reply = (
                    f"我找到一条高度吻合的记录：{candidate.get('title')}，"
                    f"{candidate.get('location') or '地点未知'}，"
                    f"{candidate.get('time_text') or '时间未知'}。\n"
                    "先不急着确认——为了对失主负责，我需要核对几个只有真正主人知道的细节。\n\n"
                    f"第一个问题：{first.get('q') if first else '能再描述一下这件物品吗？'}"
                )
                cards = [tools.candidate_card(results[0])]
                quick = []
                break

        if kind == "verify_answer":
            result = policy.evaluate_answer(session, text)
            if result["status"] == "next_question":
                nxt = result.get("next") or {}
                action_kind = "verify"
                llm_part = (
                    f" / LLM {result.get('llm_score'):.2f}"
                    if result.get("llm_score") is not None else ""
                )
                _trace(
                    trace,
                    "observe",
                    "核验",
                    f"第 {session['verification']['index']} 个核验点：关键词 "
                    f"{result.get('keyword_score', 0):.2f}{llm_part} → 综合 "
                    f"{result.get('score', 0):.2f}；继续下一个",
                )
                reply = f"我再确认一个细节：{nxt.get('q', '还有什么只有主人知道的特征？')}"
                break
            passed = result["status"] == "passed"
            claim = tools.finalize_claim(
                session,
                passed,
                session["verification"].get("answers") or [],
                result.get("ratio", 0),
            )
            _trace(
                trace,
                "reflect",
                "核验结论",
                f"综合得分 {result.get('ratio', 0):.2f} → {'通过' if passed else '不通过'}",
            )
            if passed:
                action_kind = "passed"
                ticket = tools.ticket_card(session)
                cards = [ticket] if ticket else []
                reply = (
                    "核对通过 ✅ 这几个细节都能对上。\n"
                    f"认领码：{claim.get('code')}\n"
                    f"请携带学生证到 {ticket.get('location') if ticket else '服务台'} 领取。"
                )
                if ticket and ticket.get("contact"):
                    reply += f"\n服务台联系方式：{ticket['contact']}"
                quick = _quick_replies(session, "passed")
            else:
                action_kind = "failed"
                reply = (
                    "这几个细节和登记信息对不上，我不能直接把物品交给你。\n"
                    "为了稳妥，我已经把这起认领转给服务台人工复核。"
                    "如果你能补充更多细节，也可以继续说明。"
                )
                quick = _quick_replies(session, "failed")
            break

        if kind == "done":
            action_kind = "done_found"
            reply = "这条记录已经登记好了。如果还有补充，直接告诉我。"
            quick = _quick_replies(session, "done_found")
            break

        if kind == "open_wait":
            action_kind = "no_match"
            reply = "我还在等更多线索。你可以继续补充物品特征，或者更准确的时间地点。"
            quick = _quick_replies(session, "no_match")
            break

        # 兜底
        reply = "我先帮你整理一下现有信息。"
        break

    if preface:
        reply = preface + reply
    session["history"].append({"role": "assistant", "content": reply})
    return {
        "reply": reply,
        "trace": trace,
        "session": _public_session(session),
        "cards": cards,
        "quick_replies": quick,
    }
