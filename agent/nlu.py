"""语言工具：语义核验、标记清洗。

注意：rule_extract/understand 是旧规则链路，当前 Agent 不使用；
当前运行路径是 react_agent + 原生 tool calling。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from . import matcher
from .llm import llm

CATEGORY_CHOICES = [
    "耳机盒", "耳机", "笔记本电脑", "手机", "鼠标", "键盘", "平板", "相机",
    "充电宝", "数据线", "校园卡", "身份证", "银行卡", "书包", "钱包", "水杯",
    "雨伞", "钥匙", "眼镜", "外套", "围巾", "帽子", "书", "笔记本", "计算器",
    "手表", "U盘", "化妆品", "其他",
]

BRANDS = [
    "AirPods", "Apple", "苹果", "华为", "小米", "荣耀", "三星", "OPPO", "vivo",
    "膳魔师", "象印", "乐扣", "耐克", "阿迪达斯", "无印良品", "卡西欧", "Casio",
    "罗技", "联想", "戴尔", "惠普", "索尼", "Kindle", "Acer", "宏碁",
    "雷蛇", "双飞燕", "雷柏", "微软",
]

MARK_SUFFIXES = [
    "挂件", "贴纸", "刻字", "划痕", "磨损", "图案", "标签", "名字",
    "流苏", "挂饰", "钥匙扣", "娃娃", "徽章", "保护壳", "卡套", "瓶盖",
    "贴画", "印记",
]
MARK_OBJECTS = [
    "鸭子", "小鸭", "皮卡丘", "熊猫", "青蛙", "兔子", "小猫", "小狗",
    "星星", "爱心", "校徽", "卡通", "字母",
]
SECRET_LEXICON = [
    "皮卡丘", "鸭子", "熊猫", "青蛙", "兔子", "猫", "狗", "星星", "爱心",
    "贴纸", "贴画", "挂件", "刻字", "划痕", "磨损", "刮痕", "图案", "字母",
    "名字", "标签", "流苏", "校徽", "卡通", "锁屏", "壁纸", "照片",
    "白色", "黑色", "蓝色", "红色", "绿色", "黄色", "灰色", "粉色",
    "透明", "硅胶", "金属", "羊毛", "针织", "高数", "高等数学",
    "卡西欧", "casio", "三把", "3把",
]

INTENT_WORDS = {
    "report_found": [
        "捡到", "捡了", "拾到", "拾了", "捡到一个", "发现一个", "上交",
        "找到一个", "找到个", "找到一只", "找到一件", "发现个", "发现了一个",
    ],
    "report_lost": ["丢了", "丢失", "找不到", "不见了", "落下了", "遗失"],
    "claim": ["是我的", "我来认领", "认领", "还给我", "那是我的", "我的东西"],
    "search": ["帮我找", "有没有人", "查一下", "查询", "找一下", "搜一下"],
    "confirm": ["是的", "对的", "没错", "嗯", "是它", "就是它", "就是这个", "是我丢的"],
}


def _dedupe_marks(marks: List[str]) -> List[str]:
    """去掉被更长标记包含的短标记，例如“鸭子”被“黄色小鸭子挂件”覆盖。"""
    cleaned: List[str] = []
    for mark in sorted({m.strip() for m in marks if m and m.strip()}, key=len, reverse=True):
        if any(mark in longer for longer in cleaned):
            continue
        cleaned.append(mark)
    return cleaned[:5]


def _looks_like_claim_or_confirm(text: str) -> bool:
    t = matcher.normalize(text)
    if any(w in t for w in ["是我的", "就是我的", "我来认领", "认领", "还给我", "我的东西", "是我丢的"]):
        return True
    if len(t) <= 12 and any(w in t for w in ["是的", "对的", "没错", "嗯", "是它", "就是它", "就是这个"]):
        return True
    if t in {"对", "是", "嗯", "好"}:
        return True
    return False


def _first_match(text: str, words: List[str]) -> Optional[str]:
    for word in words:
        if word in text:
            return word
    return None


def detect_intent(text: str) -> str:
    t = matcher.normalize(text)
    if any(w in t for w in INTENT_WORDS["report_found"]):
        return "report_found"
    if any(w in t for w in INTENT_WORDS["report_lost"]):
        return "report_lost"
    if any(w in t for w in INTENT_WORDS["claim"]):
        return "claim"
    if any(w in t for w in INTENT_WORDS["search"]):
        return "search"
    if any(w in t for w in INTENT_WORDS["confirm"]) and len(t) <= 12 and "不" not in t:
        return "confirm"
    if any(w in t for w in ["你好", "在吗", "怎么用", "你是谁", "帮助"]):
        return "other"
    return "other"


def extract_brand(text: str) -> str:
    t = text.lower()
    for brand in sorted(BRANDS, key=len, reverse=True):
        if brand.lower() in t:
            return brand
    # 词表外的品牌：品牌/牌子/型号是 xxx
    m = re.search(r"(?:品牌|牌子|型号)\s*(?:是|为|:|：)?\s*([A-Za-z0-9][A-Za-z0-9 \-]{1,20}|[\u4e00-\u9fff]{2,10})", text)
    if m:
        value = m.group(1).strip()
        value = re.split(r"[，,。；;\s]", value)[0]
        if value:
            return value
    return ""


def extract_location(text: str) -> str:
    """取句子中第一个地点短语，避免把“交到服务台”误当成丢失地点。"""
    loc = matcher.location_of(text)
    building = loc.get("building") or ""
    if not building:
        return ""
    idx = text.find(building)
    if idx < 0:
        return building
    phrase = text[idx:idx + 18]
    phrase = re.split(r"捡到|捡了|丢了|丢失|交到|放在|送到|发现|已经|，|。|；|;", phrase)[0]
    phrase = phrase.strip()
    if len(phrase) > 18:
        phrase = phrase[:18]
    return phrase


def extract_time(text: str) -> str:
    day = r"(?:大前天|前天|昨天|今天|上周[一二三四五六日天]?|周[一二三四五六日天]|星期[一二三四五六日天])"
    period = r"(?:早上|早晨|上午|中午|下午|傍晚|晚上|凌晨)"
    clock = r"(?:\d{1,2}\s*[:：]\s*\d{1,2}|\d{1,2}\s*点\s*(?:半|\d{1,2})?|[一二三四五六七八九十]+\s*点\s*(?:半)?)"
    patterns = [
        rf"(?:{day})?\s*(?:{period})?\s*{clock}",
        rf"(?:{day})\s*(?:{period})?",
        rf"(?:{day})",
        rf"(?:{period})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m and m.group(0).strip():
            return m.group(0).strip()
    return ""


def _clean_mark(value: str) -> str:
    value = value.strip()
    # 去掉“盒盖/杯底/里/内侧”等位置前缀
    value = re.sub(
        r"^(?:盒盖|盒子|杯底|杯盖|鼠标|键盘|物品|内部|内侧|里面|底部|背面|表面|上面|下面|外面|里|内|上|下|面|的|还)+",
        "",
        value,
    )
    # 去掉“挂着/贴了张/有”等动词和量词
    value = re.sub(
        r"^(?:挂了|挂着一个|挂着一张|挂着|挂了个|贴了张|贴了一张|贴了|贴着|刻了|刻着|写了|写着|印了|印着|带着|带了|有|是|个|一张|一个|了)+",
        "",
        value,
    )
    if "有" in value and len(value) > 4:
        value = value.rsplit("有", 1)[-1]
    value = re.sub(r"^[道个条张]+", "", value)
    return value.strip()


def extract_marks(text: str) -> List[str]:
    marks: List[str] = []
    for suffix in MARK_SUFFIXES:
        for m in re.finditer(r"([\u4e00-\u9fff]{0,8}" + re.escape(suffix) + r")", text):
            value = _clean_mark(m.group(1).strip())
            if 2 <= len(value) <= 12 and value not in marks:
                marks.append(value)
    for obj in MARK_OBJECTS:
        if obj in text and not any(obj in mark for mark in marks):
            marks.append(obj)
    return _dedupe_marks(marks)


def extract_contact(text: str) -> str:
    patterns = [
        r"1[3-9]\d{9}",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"(?:微信|wechat|qq)\s*[:：]?\s*[A-Za-z0-9_-]{5,20}",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return ""


def extract_custody(text: str) -> str:
    m = re.search(r"(?:交到|放在|送到|存到|已交到|已放在)([^，。；;]{2,30})", text)
    if not m:
        return ""
    value = m.group(1).strip()
    value = re.sub(r"(?:了|的|啦|呢|。|！|!)$", "", value).strip()
    return value


def _correction_clause(text: str) -> str:
    m = re.search(r"不是[^，,。；;]{1,12}[，,]\s*(?:是|而是)?\s*(.+)", text)
    return m.group(1).strip() if m else text


def rule_extract(text: str) -> Dict[str, Any]:
    source = _correction_clause(text)
    category = matcher.category_of(source, fallback="")
    if category == "其他":
        category = ""
    slots: Dict[str, Any] = {}
    if category:
        slots["category"] = category
    color = matcher.color_of(source)
    if color:
        slots["color"] = matcher.COLOR_LABELS.get(color, color)
    brand = extract_brand(source)
    if brand:
        slots["brand"] = brand
    location = extract_location(text)
    if location:
        slots["location"] = location
    time_text = extract_time(text)
    if time_text:
        slots["time"] = time_text
    marks = extract_marks(text)
    if marks:
        slots["marks"] = marks
    contact = extract_contact(text)
    if contact:
        slots["contact"] = contact
    custody = extract_custody(text)
    if custody:
        slots["custody"] = custody
    return {
        "intent": detect_intent(text),
        "slots": slots,
        "confidence": 0.62,
        "source": "rule",
    }


SYSTEM_PROMPT = (
    "你是校园失物招领 Agent 的语言理解模块。你的任务只是从用户最新一句话里抽取信息，不做回答、不编造事实。\n\n"
    "输出严格 JSON：\n"
    "{\n"
    '  "intent": "report_lost | report_found | search | claim | confirm | other",\n'
    '  "slots": {\n'
    '    "category": "物品类别，尽量归一化",\n'
    '    "color": "颜色",\n'
    '    "brand": "品牌或型号",\n'
    '    "location": "校园地点，保留楼栋/楼层/区域",\n'
    '    "time": "口语时间原文",\n'
    '    "marks": ["独特标记，如挂件、贴纸、刻字、划痕"],\n'
    '    "contact": "联系方式",\n'
    '    "custody": "捡到后放在哪里",\n'
    '    "secret_detail": "只有失主才知道的细节；没有就省略"\n'
    "  },\n"
    '  "confidence": 0.0\n'
    "}\n\n"
    "规则：\n"
    "1. 只输出用户明确说到的字段；没提到就省略，禁止猜测。\n"
    "2. 用户在纠正时，输出纠正后的新值。\n"
    "3. category 只能从这些值里选：" + "、".join(CATEGORY_CHOICES) + "。\n"
    "4. marks 必须是数组；把“黄色小鸭子挂件”这类完整短语放进去。\n"
    "5. secret_detail 只在用户描述“内部、背面、底部、锁屏、刻字、贴纸”等非公开细节时填写。\n"
    "6. 如果“正在追问”的字段和用户回答的内容不匹配，不要硬填；按用户实际说的内容抽取，并把不匹配的字段留空。\n"
)


def llm_extract(
    text: str,
    draft: Dict[str, Any],
    history: List[Dict[str, str]],
    awaiting: str = "",
) -> Optional[Dict[str, Any]]:
    if not llm.available:
        return None
    recent = history[-6:]
    prompt = json.dumps(
        {
            "已知信息": draft,
            "正在追问": awaiting,
            "最近对话": recent,
            "用户最新消息": text,
        },
        ensure_ascii=False,
    )
    result = llm.chat_json(SYSTEM_PROMPT, prompt, temperature=0.1)
    if not result:
        return None
    slots = result.get("slots") or {}
    if not isinstance(slots, dict):
        slots = {}
    clean: Dict[str, Any] = {}
    for key in ("category", "color", "brand", "location", "time", "contact", "custody", "secret_detail"):
        value = slots.get(key)
        if isinstance(value, str) and value.strip():
            clean[key] = value.strip()
    marks = slots.get("marks")
    if isinstance(marks, str):
        marks = [marks]
    if isinstance(marks, list):
        clean["marks"] = _dedupe_marks(
            [_clean_mark(str(m)) for m in marks if str(m).strip()]
        )
    intent = result.get("intent")
    if intent not in {"report_lost", "report_found", "search", "claim", "confirm", "other"}:
        intent = detect_intent(text)
    # LLM 容易把“耳机是白的”误判成确认/认领，用文本证据校正
    if intent in ("claim", "confirm") and not _looks_like_claim_or_confirm(text):
        intent = detect_intent(text)
    if intent == "confirm" and (
        len(text) > 12
        or any(w in text for w in ["贴", "刻", "划痕", "内部", "里面", "背面", "底部", "写着"])
    ):
        intent = "other"
    return {
        "intent": intent,
        "slots": clean,
        "confidence": float(result.get("confidence") or 0.8),
        "source": "llm",
    }


def understand(
    text: str,
    draft: Dict[str, Any],
    history: List[Dict[str, str]],
    awaiting: str = "",
) -> Dict[str, Any]:
    rule = rule_extract(text)
    if not llm.available:
        return rule
    model = llm_extract(text, draft, history, awaiting)
    if not model:
        return rule
    merged_slots = dict(rule["slots"])
    for key, value in model["slots"].items():
        if key == "marks":
            existing = list(merged_slots.get("marks") or [])
            for mark in value:
                if mark not in existing:
                    existing.append(mark)
            merged_slots["marks"] = _dedupe_marks(existing)
        else:
            merged_slots[key] = value
    return {
        "intent": model["intent"] if model["intent"] != "other" else rule["intent"],
        "slots": merged_slots,
        "confidence": max(rule["confidence"], model["confidence"]),
        "source": "llm+rule",
    }


def merge_slots(draft: Dict[str, Any], slots: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(draft)
    for key, value in slots.items():
        if value in (None, "", [], {}):
            continue
        if key == "marks":
            existing = list(merged.get("marks") or [])
            for mark in value:
                if mark and mark not in existing:
                    existing.append(mark)
            merged["marks"] = _dedupe_marks(existing)
        elif key == "secret_detail":
            merged["secret_detail"] = value
        else:
            merged[key] = value
    return merged


def secret_keywords(text: str) -> List[str]:
    t = matcher.normalize(text)
    hits = []
    for word in SECRET_LEXICON:
        if word.lower() in t and word not in hits:
            hits.append(word)
    return hits[:8]


def char_bigrams(text: str) -> set:
    t = matcher.compact(text)
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def answer_score(answer: str, secret: Dict[str, Any]) -> float:
    a = matcher.normalize(answer)
    if not a:
        return 0.0
    if any(w in a for w in ["不记得", "忘了", "不知道", "没注意", "不清楚", "没印象"]) and len(a) <= 14:
        return 0.0
    keywords = [str(k) for k in secret.get("keywords") or [] if str(k)]
    groups = secret.get("groups")
    if isinstance(groups, list) and groups:
        group_scores = []
        for group in groups:
            terms = [str(x) for x in group if str(x)] if isinstance(group, list) else [str(group)]
            group_scores.append(1.0 if any(matcher.normalize(t) in a for t in terms) else 0.0)
        coverage = sum(group_scores) / max(1, len(group_scores))
    else:
        hit = sum(1 for k in keywords if matcher.normalize(k) in a)
        coverage = hit / max(1, len(keywords)) if keywords else 0.0
    reference = secret.get("answer") or ""
    sim = 0.0
    if reference:
        ga, gb = char_bigrams(a), char_bigrams(reference)
        if ga and gb:
            sim = len(ga & gb) / len(ga | gb)
    # 一个关键词命中即可视为“说到了点上”，再结合文本相似度兜底
    return round(max(coverage, min(1.0, sim * 1.8)), 4)


JUDGE_SYSTEM = """你是校园失物招领的归属核验员。你会看到一条只有真正主人知道的登记细节，以及认领人的回答。
请严格判断回答是否与登记细节吻合，不要因为语气肯定、提到同类别或模糊沾边就给高分。
只输出 JSON：{"match": 0.0, "reason": "一句话理由"}
match 含义：0=完全不符/不记得，0.5=部分沾边，1=关键细节完全吻合。
"""


def judge_answer(answer: str, secret: Dict[str, Any]) -> Optional[float]:
    """用 LLM 做语义核验；失败时返回 None，由关键词分数兜底。"""
    if not llm.available or not answer.strip():
        return None
    a = matcher.normalize(answer)
    if any(w in a for w in ["不记得", "忘了", "不知道", "没注意", "不清楚", "没印象"]) and len(a) <= 14:
        return 0.0
    payload = json.dumps(
        {
            "核验问题": secret.get("q"),
            "登记关键词": secret.get("keywords") or [],
            "登记时的原始描述": secret.get("answer") or "",
            "认领人回答": answer,
        },
        ensure_ascii=False,
    )
    result = llm.chat_json(JUDGE_SYSTEM, payload, temperature=0.0, timeout=20)
    if not result:
        return None
    try:
        score = float(result.get("match"))
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))
