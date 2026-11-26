"""核验机制与旧状态机。

当前 Agent 的流程由 react_agent 的 LLM 规划；这里保留 start_verification /
evaluate_answer 等不可绕过的核验工具，以及旧状态机用于对比。

这样设计的原因：
- 失物招领是有明确阶段和风险边界的任务，流程不能交给模型自由发挥；
- 但“追问什么、怎么问、如何判断描述是否吻合”又需要语言能力；
- 所以采用“LLM 理解 + 规则策略 + 工具执行”的神经符号结构。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import nlu
from .llm import LLMError, llm

STRONG_SCORE = 0.74
POSSIBLE_SCORE = 0.56
MAX_ASKS_PER_SLOT = 2

SLOT_LABELS = {
    "category": "物品类型",
    "location": "地点",
    "time": "时间",
    "color": "颜色",
    "brand": "品牌",
    "marks": "独特标记",
    "secret_detail": "核验细节",
    "custody": "保管地点",
    "contact": "联系方式",
    "distinguishing": "区分特征",
}

QUESTIONS = {
    "category_lost": "你丢的是哪一类东西？比如耳机、水杯、校园卡。",
    "category_found": "你捡到的是哪一类东西？比如耳机、水杯、校园卡。",
    "location": "大概在校园的哪个位置？比如图书馆三楼、教学楼 204。",
    "time": "大概是什么时候？昨天下午、今天上午都可以。",
    "distinguishing": "它有什么颜色、品牌，或者特别好认的标记吗？比如挂件、贴纸、刻字。",
    "marks": "上面有没有挂件、贴纸、刻字之类特别好认的特征？这能帮我排除几个相似的。",
    "secret": "为了之后核对失主，能不能补充一个只有主人知道的细节？比如盒内的贴纸、刻字，或者耳机本身的颜色。",
    "custody": "物品现在放在哪里？比如图书馆一楼服务台。",
    "role": "可以先告诉我：你是丢了东西，还是捡到了东西？",
}


def _missing_for_lost(draft: Dict[str, Any]) -> List[str]:
    if not draft.get("category"):
        return ["category"]
    if not draft.get("location"):
        return ["location"]
    if not draft.get("time"):
        return ["time"]
    if not any(draft.get(k) for k in ("color", "brand", "marks")):
        return ["distinguishing"]
    return []


def _missing_for_found(draft: Dict[str, Any]) -> List[str]:
    if not draft.get("category"):
        return ["category"]
    if not draft.get("location"):
        return ["location"]
    if not draft.get("time"):
        return ["time"]
    if not draft.get("secret_detail") and not draft.get("secrets") and not draft.get("no_secret"):
        return ["secret_detail"]
    if not draft.get("custody"):
        return ["custody"]
    return []


def _ask_action(
    session: Dict[str, Any],
    slot: str,
    extra: str = "",
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    role = session.get("role") or "lost"
    notice = session.pop("notice", "")
    if slot == "category":
        question = QUESTIONS[f"category_{role}"]
    elif slot == "distinguishing":
        question = QUESTIONS["distinguishing"]
    elif slot == "secret_detail":
        question = secret_question_for((session.get("draft") or {}).get("category"))
    else:
        question = QUESTIONS.get(slot, "能再补充一点信息吗？")
    if extra:
        question = extra + question
    if notice:
        question = f"{notice}{question}"
    return {
        "type": "ask",
        "slots": [slot],
        "question": question,
        "notice": notice,
        "reason": reason or f"缺少{SLOT_LABELS.get(slot, slot)}",
    }


def next_action(session: Dict[str, Any]) -> Dict[str, Any]:
    """返回当前应执行的动作。"""
    if session.get("phase") == "verify":
        return {"type": "verify_answer"}
    if session.get("phase") == "done":
        return {"type": "done"}

    role = session.get("role")
    if not role:
        return {"type": "ask_role"}

    draft = session.get("draft") or {}
    asked = session.get("asked") or []
    unknown = set(session.get("unknown_slots") or [])

    if role == "lost":
        if not session.get("case_id"):
            missing = _missing_for_lost(draft)
            if missing:
                slot = missing[0]
                if slot in unknown:
                    # 用户明确说不记得，换一个能推进的字段；实在没有就开始搜
                    alternatives = [s for s in ("location", "time", "marks", "color") if s not in unknown and not draft.get(s)]
                    if alternatives:
                        slot = "distinguishing" if "marks" in alternatives else alternatives[0]
                    else:
                        return {"type": "create_lost"}
                return _ask_action(session, slot)
            return {"type": "create_lost"}
        if session.get("dirty"):
            return {"type": "search"}
        if session.get("phase") == "open":
            return {"type": "open_wait"}
        return {"type": "search"}

    if role == "found":
        if not session.get("case_id"):
            missing = _missing_for_found(draft)
            if missing:
                slot = missing[0]
                return _ask_action(session, slot)
            return {"type": "create_found"}
        if session.get("dirty"):
            return {"type": "update_found"}
        return {"type": "done"}

    return {"type": "ask_role"}


SLOT_PICK_SYSTEM = """你是校园失物招领 Agent 的追问策略模块。候选记录彼此很像，请选一个最能区分它们、同时用户最容易回答的字段。
只输出 JSON：{"slot": "字段名", "reason": "一句话理由"}
字段名只能从“可选字段”里选。
"""


def choose_slot(
    session: Dict[str, Any],
    results: List[Dict[str, Any]],
    options: List[str],
) -> Dict[str, Any]:
    """让 LLM 在多个可追问字段中选一个；失败时回退到规则顺序。"""
    if not options:
        return {"slot": "", "reason": ""}
    if len(options) == 1 or not llm.available:
        return {"slot": options[0], "reason": "", "source": "fallback"}
    candidates = []
    for item in results[:3]:
        case = item.get("candidate") or {}
        candidates.append(
            {
                "标题": case.get("title"),
                "地点": case.get("location"),
                "时间": case.get("time_text"),
                "公开特征": case.get("marks") or [],
                "匹配分": item.get("score"),
            }
        )
    payload = json.dumps(
        {
            "已知信息": {k: v for k, v in (session.get("draft") or {}).items() if k not in ("secret_detail", "secrets")},
            "候选": candidates,
            "可选字段": options,
            "目标": "选一个最能区分候选、且用户容易回答的字段，不要选用户已经说过的",
        },
        ensure_ascii=False,
    )
    result = llm.chat_json(SLOT_PICK_SYSTEM, payload, temperature=0.2, timeout=20)
    if result and result.get("slot") in options:
        return {
            "slot": result["slot"],
            "reason": str(result.get("reason") or "")[:80],
            "source": "llm",
        }
    return {"slot": options[0], "reason": "", "source": "fallback"}


def after_search(session: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """拿到检索结果后的二次决策。"""
    if not results:
        return {"type": "no_match", "reason": "没有任何反向角色的记录"}

    top = results[0]
    score = top["score"]
    margin = top.get("margin", 0.0)
    draft = session.get("draft") or {}
    asked = session.get("asked") or []
    unknown = set(session.get("unknown_slots") or [])

    if score < POSSIBLE_SCORE:
        return {
            "type": "no_match",
            "reason": f"最高匹配分 {score:.2f} 低于阈值 {POSSIBLE_SCORE}",
            "results": results,
        }

    if score >= STRONG_SCORE:
        # 高分但仍有多个相似项时，先问一个能拉开差距的独特标记
        if (
            not draft.get("marks")
            and "marks" not in asked
            and "marks" not in unknown
            and margin < 0.25
        ):
            return _ask_action(
                session,
                "marks",
                extra=f"我找到 {len(results)} 条相似记录，先确认一个细节：",
            )
        return {
            "type": "verify",
            "case_id": top["candidate"]["id"],
            "score": score,
            "results": results,
            "reason": f"最高匹配分 {score:.2f}，超过直接核验阈值 {STRONG_SCORE}",
        }

    # 中等分数：让 LLM 在多个可追问字段里挑一个最能区分候选的
    options = [
        slot for slot in ("marks", "color", "time", "location")
        if not draft.get(slot) and slot not in asked and slot not in unknown
    ]
    picked = choose_slot(session, results, options)
    if picked["slot"]:
        reason = picked.get("reason") or "补充一个能提高区分度的字段"
        who = "LLM" if picked.get("source") == "llm" else "规则"
        return _ask_action(
            session,
            picked["slot"],
            extra=f"我找到 {len(results)} 条可能相关的记录，但还不够确定。",
            reason=f"{who} 选择追问「{SLOT_LABELS.get(picked['slot'], picked['slot'])}」：{reason}",
        )
    return {
        "type": "verify",
        "case_id": top["candidate"]["id"],
        "score": score,
        "results": results,
        "reason": f"最高匹配分 {score:.2f}，进入谨慎核验",
    }


def secret_question_for(category: Optional[str]) -> str:
    mapping = {
        "耳机盒": "耳机盒内部或表面有没有贴纸、刻字，或者明显的磨损？",
        "耳机": "耳机线、插头或耳机本体有什么特别之处？",
        "手机": "手机的锁屏壁纸或手机壳有什么特征？",
        "校园卡": "卡面上的姓名或学号后四位是什么？",
        "身份证": "证件上的姓名和出生年份后两位是什么？",
        "银行卡": "卡面是哪家银行、尾号是多少？",
        "水杯": "杯底、杯盖或杯身有没有特殊标记？",
        "书包": "包里有什么东西，或者拉链上有什么挂件？",
        "钥匙": "钥匙扣是什么样式？一共几把钥匙？",
        "雨伞": "伞柄、伞面图案有什么特征？",
        "笔记本": "封面颜色和里面记的课程是什么？",
        "计算器": "品牌是什么？背面有没有写名字？",
        "鼠标": "鼠标底部、接收器或表面有没有贴纸、刻字，或者明显磨损？",
        "键盘": "键帽、背面或线材上有没有特殊标记？",
        "平板": "平板背面或保护壳上有什么特征？锁屏壁纸是什么？",
        "相机": "相机机身、镜头或背带上有什么特殊标记？",
    }
    return mapping.get(category or "", "这件物品有没有只有主人知道的细节？比如贴纸、刻字、内部特征或磨损。")


def build_secret(text: str, category: Optional[str]) -> Dict[str, Any]:
    return {
        "q": secret_question_for(category),
        "keywords": nlu.secret_keywords(text),
        "answer": text,
        "weight": 2,
    }


def start_verification(session: Dict[str, Any], case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    secrets = [s for s in (case.get("secrets") or []) if isinstance(s, dict)]
    if not secrets:
        return None
    secrets = sorted(secrets, key=lambda s: float(s.get("weight", 1)), reverse=True)[:3]
    total = sum(float(s.get("weight", 1)) for s in secrets) or 1.0
    session["verification"] = {
        "case_id": case["id"],
        "questions": secrets,
        "index": 0,
        "answers": [],
        "weight_total": total,
        "weight_hit": 0.0,
        "status": "asking",
        "case": case,
    }
    session["phase"] = "verify"
    return session["verification"]


def current_question(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    verification = session.get("verification") or {}
    questions = verification.get("questions") or []
    index = verification.get("index", 0)
    if index < len(questions):
        return questions[index]
    return None


def evaluate_answer(
    session: Dict[str, Any],
    answer: str,
    require_llm: bool = False,
) -> Dict[str, Any]:
    """评估当前回答，返回下一步：next_question / passed / failed。"""
    verification = session.get("verification") or {}
    questions = verification.get("questions") or []
    index = verification.get("index", 0)
    if index >= len(questions):
        return {"status": "failed", "reason": "没有更多问题"}

    secret = questions[index]
    keyword_score = nlu.answer_score(answer, secret)
    llm_score = nlu.judge_answer(answer, secret)
    if llm_score is None and require_llm:
        raise LLMError("核验模型不可用，无法完成归属判断")
    if require_llm:
        # 新架构：归属判断以 LLM 语义核验为准；关键词只记录为审计信号
        score = round(llm_score, 4)
    elif llm_score is None:
        score = keyword_score
    else:
        # 旧规则版：关键词硬证据 + LLM 同义转述兜底
        score = max(keyword_score, round(llm_score * 0.9, 4))
    weight = float(secret.get("weight", 1))
    verification.setdefault("answers", []).append(
        {
            "q": secret.get("q"),
            "answer": answer,
            "score": round(score, 4),
            "keyword_score": round(keyword_score, 4),
            "llm_score": None if llm_score is None else round(llm_score, 4),
            "weight": weight,
        }
    )
    verification["weight_hit"] = verification.get("weight_hit", 0.0) + score * weight
    verification["index"] = index + 1
    ratio = verification["weight_hit"] / max(0.001, verification.get("weight_total", 1.0))
    verification["ratio"] = round(ratio, 4)

    remaining = len(questions) - verification["index"]
    answered = verification["index"]
    strong_hit = any(
        float(item.get("weight", 1)) >= 2 and item.get("score", 0) >= 0.5
        for item in verification["answers"]
    )

    # 连续两个问题都对不上，尽早止损，避免在和冒领者对话里泄露更多信息
    if answered >= 2 and ratio < 0.16:
        return {
            "status": "failed",
            "reason": "连续两个核验点不匹配",
            "ratio": ratio,
            "score": score,
            "keyword_score": keyword_score,
            "llm_score": llm_score,
        }

    if remaining > 0:
        return {
            "status": "next_question",
            "ratio": ratio,
            "score": score,
            "keyword_score": keyword_score,
            "llm_score": llm_score,
            "next": current_question(session),
        }

    passed = ratio >= 0.55 and strong_hit
    return {
        "status": "passed" if passed else "failed",
        "ratio": ratio,
        "score": score,
        "keyword_score": keyword_score,
        "llm_score": llm_score,
        "reason": "核验细节达到阈值" if passed else "核验细节未达到阈值",
    }
