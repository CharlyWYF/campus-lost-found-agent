"""可解释的混合匹配：字符 n-gram 语义 + 结构化字段 + 独特标记。

不依赖外部向量库，保证课堂现场断网也能跑；同时每个分数都能说清来源。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import config

# ---------------------------------------------------------------- 类别与同义词

CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    ("耳机盒", ["耳机盒", "耳机仓", "充电盒", "充电仓", "耳机充电盒"]),
    ("耳机", ["耳机", "耳麦", "airpods", "蓝牙耳机", "有线耳机"]),
    ("笔记本电脑", ["笔记本电脑", "手提电脑", "macbook", "thinkpad"]),
    ("手机", ["手机", "iphone", "华为手机", "小米手机"]),
    ("鼠标", ["鼠标", "无线鼠标", "有线鼠标"]),
    ("键盘", ["键盘", "机械键盘"]),
    ("平板", ["平板", "ipad"]),
    ("相机", ["相机", "单反", "微单"]),
    ("充电宝", ["充电宝", "移动电源"]),
    ("数据线", ["数据线", "充电线"]),
    ("校园卡", ["校园卡", "学生卡", "一卡通", "饭卡"]),
    ("身份证", ["身份证"]),
    ("银行卡", ["银行卡", "信用卡", "储蓄卡"]),
    ("书包", ["书包", "双肩包", "背包"]),
    ("钱包", ["钱包", "卡包", "钱夹"]),
    ("水杯", ["水杯", "保温杯", "杯子", "水壶", "保温壶"]),
    ("雨伞", ["雨伞", "伞"]),
    ("钥匙", ["钥匙", "钥匙串"]),
    ("眼镜", ["眼镜", "近视镜", "墨镜"]),
    ("外套", ["外套", "夹克", "卫衣", "羽绒服"]),
    ("围巾", ["围巾", " scarf"]),
    ("帽子", ["帽子", "棒球帽", "鸭舌帽"]),
    ("书", ["教材", "课本", "书籍"]),
    ("笔记本", ["笔记本", "记事本", "本子"]),
    ("计算器", ["计算器"]),
    ("手表", ["手表", "手环"]),
    ("U盘", ["u盘", "优盘", "移动硬盘"]),
    ("化妆品", ["口红", "粉饼", "化妆品", "香水"]),
    ("其他", []),
]

CATEGORY_GROUP = {
    "耳机盒": "耳机", "耳机": "耳机",
    "笔记本电脑": "数码", "手机": "数码", "充电宝": "数码", "U盘": "数码",
    "计算器": "数码", "手表": "数码", "鼠标": "数码", "键盘": "数码",
    "平板": "数码", "相机": "数码", "数据线": "数码",
    "校园卡": "证件", "身份证": "证件", "银行卡": "证件",
    "书包": "包袋", "钱包": "包袋",
    "水杯": "生活", "雨伞": "生活", "钥匙": "生活", "眼镜": "生活",
    "外套": "衣物", "围巾": "衣物", "帽子": "衣物",
    "书": "学习", "笔记本": "学习",
    "化妆品": "生活", "其他": "其他",
}

COLOR_LABELS = {
    "黑": "黑色", "白": "白色", "蓝": "蓝色", "红": "红色", "粉": "粉色",
    "绿": "绿色", "黄": "黄色", "灰": "灰色", "棕": "棕色", "银": "银色",
    "透明": "透明",
}

COLOR_FAMILIES: Dict[str, List[str]] = {
    "黑": ["黑色", "黑", "深黑", "曜石黑", "碳黑"],
    "白": ["白色", "白", "米白", "奶白", "雪白"],
    "蓝": ["深蓝", "藏青", "天蓝", "雾霾蓝", "蓝色", "蓝"],
    "红": ["酒红", "玫红", "红色", "红"],
    "粉": ["粉色", "粉", "藕粉"],
    "绿": ["墨绿", "军绿", "绿色", "绿"],
    "黄": ["姜黄", "柠檬黄", "黄色", "黄"],
    "灰": ["银灰", "深灰", "灰色", "灰"],
    "棕": ["咖啡色", "卡其色", "棕色", "棕"],
    "银": ["银色", "银"],
    "透明": ["透明", "clear"],
}

BUILDINGS = [
    "图书馆", "教学楼", "实验楼", "体育馆", "一食堂", "二食堂", "三食堂",
    "食堂", "宿舍", "操场", "报告厅", "校门", "行政楼", "艺术楼",
]
AREAS = [
    "靠窗", "靠墙", "门口", "前台", "服务台", "值班室", "器材室", "座位",
    "自习区", "大厅", "走廊", "电梯", "楼梯", "看台", "球场", "餐桌上",
]

_CJK = re.compile(r"[^\u4e00-\u9fff]+")
_PUNCT = re.compile(r"[\s,，。；;：:、!！?？\"'“”‘’（）()\[\]【】<>《》\-—_/\\|]+")


def normalize(text: Any) -> str:
    if text is None:
        return ""
    return _PUNCT.sub(" ", str(text).lower()).strip()


def compact(text: Any) -> str:
    return _CJK.sub("", normalize(text))


def category_of(text: Any, fallback: str = "其他") -> str:
    t = normalize(text)
    if not t:
        return fallback
    # 先处理容易混淆的复合词
    if any(k in t for k in ["耳机盒", "耳机仓", "充电盒", "充电仓"]):
        return "耳机盒"
    if "airpods" in t and ("盒" in t or "仓" in t or "充电" in t):
        return "耳机盒"
    for category, words in CATEGORY_RULES:
        if category in {"耳机盒", "其他"}:
            continue
        if any(w and w in t for w in words):
            return category
    if "airpods" in t:
        return "耳机"
    return fallback


def color_of(text: Any) -> str:
    t = normalize(text)
    if not t:
        return ""
    candidates: List[Tuple[int, str, str]] = []
    for family, words in COLOR_FAMILIES.items():
        for word in words:
            if word and word in t:
                candidates.append((len(word), family, word))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


def _cn_number(value: str) -> Optional[int]:
    mapping = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if value.isdigit():
        return int(value)
    if value in mapping:
        return mapping[value]
    return None


def location_of(text: Any) -> Dict[str, Any]:
    t = normalize(text)
    building = next((b for b in sorted(BUILDINGS, key=len, reverse=True) if b in t), "")
    if not building:
        # 校园里楼栋名字很多，不能只靠枚举；用“XX楼/XX馆”兜底
        m = re.search(r"([\u4e00-\u9fff]{2,6}(?:楼|馆|堂|中心))", t)
        if m:
            candidate = m.group(1)
            if candidate not in {"一楼", "二楼", "三楼", "四楼", "五楼", "六楼", "七楼", "八楼", "九楼", "十楼"}:
                building = candidate
    floor = None
    m = re.search(r"([一二三四五六七八九十两\d]+)\s*楼", t)
    if m:
        floor = _cn_number(m.group(1))
    area = next((a for a in sorted(AREAS, key=len, reverse=True) if a in t), "")
    return {"building": building, "floor": floor, "area": area, "raw": t}


def parse_time_text(text: Any, now: Optional[datetime] = None) -> Tuple[Optional[datetime], bool]:
    """解析“昨天下午”“今天 16:30”等口语时间，返回 (时间, 是否精确到小时)。"""
    t = normalize(text)
    if not t:
        return None, False
    now = now or config.now()
    day_offset = 0
    day_found = False
    for word, offset in [("大前天", -3), ("前天", -2), ("昨天", -1), ("今天", 0),
                         ("今早", 0), ("昨晚", 0), ("刚才", 0), ("刚刚", 0)]:
        if word in t:
            day_offset = offset
            day_found = True
            break
    if not day_found and "上周" in t:
        day_offset = -7

    hour = None
    minute = 0
    precise = False
    m = re.search(r"(\d{1,2})\s*[:：]\s*(\d{1,2})", t)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        precise = True
    else:
        m = re.search(r"(\d{1,2})\s*点\s*(半|\d{1,2})?", t)
        if m:
            hour = int(m.group(1))
            if m.group(2) == "半":
                minute = 30
            elif m.group(2) and m.group(2).isdigit():
                minute = int(m.group(2))
            precise = True

    if hour is None:
        if "凌晨" in t:
            hour = 2
        elif "早上" in t or "早晨" in t or "上午" in t:
            hour = 9
        elif "中午" in t:
            hour = 12
        elif "下午" in t:
            hour = 15
        elif "傍晚" in t:
            hour = 18
        elif "晚上" in t or "今晚" in t:
            hour = 19
        else:
            hour = 12

    # 下午三点、晚上八点这类表达
    if precise and hour is not None and hour <= 12:
        if "下午" in t or "晚上" in t or "傍晚" in t:
            if hour < 12:
                hour += 12

    base = now + timedelta(days=day_offset)
    try:
        return base.replace(hour=hour, minute=minute, second=0, microsecond=0), precise
    except ValueError:
        return None, False


def _time_of(case: Dict[str, Any], now: Optional[datetime] = None) -> Tuple[Optional[datetime], bool]:
    iso = case.get("time_iso")
    if iso:
        try:
            return datetime.fromisoformat(iso), True
        except ValueError:
            pass
    return parse_time_text(case.get("time_text"), now)


def _jaccard(a: str, b: str) -> float:
    a, b = compact(a), compact(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    grams_a = {a[i:i + 2] for i in range(max(1, len(a) - 1))}
    grams_b = {b[i:i + 2] for i in range(max(1, len(b) - 1))}
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


# ---------------------------------------------------------------- 文本相似度

def tokenize(text: Any) -> List[str]:
    t = normalize(text)
    tokens: List[str] = []
    tokens.extend(re.findall(r"[a-z0-9]+", t))
    chinese = re.sub(r"[^\u4e00-\u9fff]", "", t)
    for n in (2, 3):
        if len(chinese) >= n:
            tokens.extend(chinese[i:i + n] for i in range(len(chinese) - n + 1))
    return tokens


class TfidfIndex:
    def __init__(self, docs: Iterable[str]):
        self.docs = list(docs)
        self.counts = [Counter(tokenize(d)) for d in self.docs]
        df: Counter = Counter()
        for counts in self.counts:
            df.update(counts.keys())
        n = max(1, len(self.docs))
        self.idf = {term: math.log((n + 1) / (freq + 1)) + 1 for term, freq in df.items()}
        self.vectors = [self._vector(counts) for counts in self.counts]

    def _vector(self, counts: Counter) -> Dict[str, float]:
        vec = {t: c * self.idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def sim(self, query: str, index: int) -> float:
        if index < 0 or index >= len(self.vectors):
            return 0.0
        q = self._vector(Counter(tokenize(query)))
        return sum(v * self.vectors[index].get(t, 0.0) for t, v in q.items())


def case_text(case: Dict[str, Any]) -> str:
    parts = [
        case.get("title"), case.get("description"), case.get("category"),
        case.get("color"), case.get("brand"), case.get("location"),
        " ".join(case.get("marks") or []),
    ]
    return " ".join(str(p) for p in parts if p)


# ---------------------------------------------------------------- 结构化打分

def category_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ca = a.get("category") or category_of(case_text(a))
    cb = b.get("category") or category_of(case_text(b))
    if ca == cb and ca not in ("", "其他"):
        return 1.0
    if ca == "其他" or cb == "其他" or not ca or not cb:
        return 0.3
    if CATEGORY_GROUP.get(ca) == CATEGORY_GROUP.get(cb):
        return 0.6
    return 0.05


def color_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ca = a.get("color") or color_of(case_text(a))
    cb = b.get("color") or color_of(case_text(b))
    ca = color_of(ca) or ca
    cb = color_of(cb) or cb
    if not ca and not cb:
        return 0.4
    if not ca or not cb:
        return 0.45
    if ca == cb:
        return 1.0
    if {ca, cb} <= {"黑", "灰", "银"}:
        return 0.45
    if {ca, cb} <= {"白", "银", "透明"}:
        return 0.4
    return 0.1


def location_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    la = location_of(a.get("location") or case_text(a))
    lb = location_of(b.get("location") or case_text(b))
    if not la["building"] and not lb["building"] and not la["area"] and not lb["area"]:
        return 0.3
    if la["building"] and lb["building"]:
        if la["building"] != lb["building"]:
            return 0.08
        if la["floor"] and lb["floor"]:
            if la["floor"] != lb["floor"]:
                return 0.55
            if la["area"] and lb["area"] and la["area"] == lb["area"]:
                return 1.0
            return 0.95
        return 0.82
    if la["area"] and lb["area"] and la["area"] == lb["area"]:
        return 0.78
    # 同一栋楼但一方没写楼，用字符重合兜底
    if _jaccard(la["raw"], lb["raw"]) > 0.4:
        return 0.6
    return 0.3


def time_score(a: Dict[str, Any], b: Dict[str, Any], now: Optional[datetime] = None) -> float:
    ta, pa = _time_of(a, now)
    tb, pb = _time_of(b, now)
    if ta and tb:
        diff_days = abs((ta.date() - tb.date()).days)
        hour_diff = abs((ta - tb).total_seconds()) / 3600.0
        if diff_days == 0:
            if pa and pb:
                return 1.0 if hour_diff <= 3 else (0.9 if hour_diff <= 6 else 0.8)
            return 0.9
        if diff_days == 1:
            return 0.78
        if diff_days == 2:
            return 0.6
        if diff_days <= 7:
            return 0.4
        return 0.2
    raw_a, raw_b = normalize(a.get("time_text")), normalize(b.get("time_text"))
    if raw_a and raw_b:
        if raw_a == raw_b:
            return 0.75
        if _jaccard(raw_a, raw_b) > 0.35:
            return 0.65
    return 0.35


def marks_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ma = [m for m in (a.get("marks") or []) if m]
    mb = [m for m in (b.get("marks") or []) if m]
    if not ma and not mb:
        return 0.5
    if not ma or not mb:
        return 0.42
    best = max(_jaccard(x, y) for x in ma for y in mb)
    if best >= 0.5:
        return 1.0
    if best >= 0.28:
        return 0.72
    return 0.18


BRAND_ALIASES = [
    ("apple", ["苹果", "apple", "airpods", "iphone", "macbook", "ipad"]),
    ("huawei", ["华为", "huawei", "荣耀", "honor"]),
    ("xiaomi", ["小米", "xiaomi", "红米", "redmi"]),
    ("lenovo", ["联想", "lenovo"]),
    ("dell", ["戴尔", "dell"]),
    ("hp", ["惠普", "hp"]),
    ("logitech", ["罗技", "logitech"]),
    ("casio", ["卡西欧", "casio"]),
]


def canonical_brand(value: Any) -> str:
    t = normalize(value)
    if not t:
        return ""
    for canonical, aliases in BRAND_ALIASES:
        if any(alias in t for alias in aliases):
            return canonical
    return t


def keyword_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    def keys(c: Dict[str, Any]) -> set:
        out = set()
        for field in ("category", "color"):
            value = normalize(c.get(field))
            if value:
                out.add(value)
        brand = canonical_brand(c.get("brand"))
        if brand:
            out.add(brand)
        loc = location_of(c.get("location"))
        for value in (loc["building"], str(loc["floor"]) if loc["floor"] else "", loc["area"]):
            if value:
                out.add(value)
        for mark in c.get("marks") or []:
            out.add(compact(mark))
        return {x for x in out if x}

    ka, kb = keys(a), keys(b)
    if not ka or not kb:
        return 0.3
    inter = ka & kb
    if not inter:
        return 0.08
    return len(inter) / max(1, min(len(ka), len(kb)))


# ---------------------------------------------------------------- 主入口

WEIGHTS = {
    "text": 0.38,
    "structured": 0.27,
    "marks": 0.15,
    "keyword": 0.20,
}
STRUCT_WEIGHTS = {"category": 0.40, "color": 0.20, "location": 0.25, "time": 0.15}


def score_pair(
    query: Dict[str, Any],
    candidate: Dict[str, Any],
    text_sim: float = 0.0,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    cat = category_score(query, candidate)
    col = color_score(query, candidate)
    loc = location_score(query, candidate)
    tim = time_score(query, candidate, now)
    structured = (
        cat * STRUCT_WEIGHTS["category"]
        + col * STRUCT_WEIGHTS["color"]
        + loc * STRUCT_WEIGHTS["location"]
        + tim * STRUCT_WEIGHTS["time"]
    )
    marks = marks_score(query, candidate)
    keyword = keyword_overlap(query, candidate)
    total = (
        text_sim * WEIGHTS["text"]
        + structured * WEIGHTS["structured"]
        + marks * WEIGHTS["marks"]
        + keyword * WEIGHTS["keyword"]
    )
    total = max(0.0, min(1.0, total))

    reasons: List[str] = []
    if cat >= 0.99:
        reasons.append(f"同类物品：{candidate.get('category') or '未分类'}")
    elif cat >= 0.55:
        reasons.append("物品类别相近")
    if col >= 0.99:
        reasons.append(f"颜色一致：{candidate.get('color') or ''}")
    if loc >= 0.9:
        reasons.append(f"地点吻合：{candidate.get('location') or ''}")
    elif loc >= 0.7:
        reasons.append("地点相近")
    if tim >= 0.75:
        reasons.append("时间接近")
    if marks >= 0.99:
        reasons.append("都有相同的独特标记")
    elif marks >= 0.7:
        reasons.append("独特标记相似")
    if not reasons and text_sim >= 0.3:
        reasons.append("描述内容相似")

    return {
        "score": round(total, 4),
        "reasons": reasons[:4],
        "breakdown": {
            "text": round(text_sim, 4),
            "category": round(cat, 4),
            "color": round(col, 4),
            "location": round(loc, 4),
            "time": round(tim, 4),
            "marks": round(marks, 4),
            "keyword": round(keyword, 4),
            "structured": round(structured, 4),
        },
        "candidate": candidate,
    }


def rank(
    query: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    docs = [case_text(c) for c in candidates]
    index = TfidfIndex(docs)
    qtext = case_text(query)
    scored: List[Dict[str, Any]] = []
    for i, candidate in enumerate(candidates):
        item = score_pair(query, candidate, index.sim(qtext, i), now)
        item["candidate"] = candidate
        scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def best_margin(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    if len(results) == 1:
        return 1.0
    return max(0.0, results[0]["score"] - results[1]["score"])
