"""演示数据集：故意包含相似干扰项，用来展示 Agent 的追问与核验。"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from . import config, db


def _at(offset_days: int, hour: int, minute: int) -> datetime:
    base = config.now() + timedelta(days=offset_days)
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _relative_time(dt: datetime) -> str:
    delta = (dt.date() - config.now().date()).days
    if delta == 0:
        label = "今天"
    elif delta == -1:
        label = "昨天"
    elif delta == -2:
        label = "前天"
    elif delta < 0:
        label = f"{-delta}天前"
    else:
        label = dt.strftime("%m月%d日")
    return f"{label} {dt.strftime('%H:%M')}"


def _found(
    title: str,
    description: str,
    category: str,
    color: str,
    brand: str,
    location: str,
    when: datetime,
    marks: List[str],
    secrets: List[Dict[str, Any]],
    contact: str,
    public_location: str,
    status: str = "open",
) -> Dict[str, Any]:
    return {
        "role": "found",
        "status": status,
        "title": title,
        "description": description,
        "category": category,
        "color": color,
        "brand": brand,
        "location": location,
        "time_text": _relative_time(when),
        "time_iso": _iso(when),
        "marks": marks,
        "secrets": secrets,
        "contact": contact,
        "public_location": public_location,
        "created_at": _iso(when + timedelta(minutes=25)),
    }


def _lost(
    title: str,
    description: str,
    category: str,
    color: str,
    brand: str,
    location: str,
    when: datetime,
    marks: List[str],
    status: str = "open",
) -> Dict[str, Any]:
    return {
        "role": "lost",
        "status": status,
        "title": title,
        "description": description,
        "category": category,
        "color": color,
        "brand": brand,
        "location": location,
        "time_text": _relative_time(when),
        "time_iso": _iso(when),
        "marks": marks,
        "secrets": [],
        "contact": "",
        "public_location": "",
        "created_at": _iso(when + timedelta(hours=2)),
    }


def build_cases() -> List[Dict[str, Any]]:
    y = -1
    cases = [
        _found(
            title="黑色 AirPods 耳机盒",
            description=(
                "在图书馆三楼靠窗座位捡到一个黑色耳机盒，外面挂着一个黄色小鸭子挂件。"
                "已经交到图书馆一楼服务台。"
            ),
            category="耳机盒",
            color="黑色",
            brand="Apple AirPods",
            location="图书馆三楼靠窗座位",
            when=_at(y, 16, 30),
            marks=["黄色小鸭子挂件"],
            secrets=[
                {
                    "q": "耳机盒内部有没有贴纸、图案或刻字？",
                    "keywords": ["皮卡丘", "蓝色", "贴纸"],
                    "groups": [["皮卡丘"], ["贴纸"]],
                    "weight": 2,
                    "hint": "盒盖内侧",
                },
                {
                    "q": "耳机本体是什么颜色？",
                    "keywords": ["白色"],
                    "groups": [["白色"]],
                    "weight": 1,
                    "hint": "耳机",
                },
                {
                    "q": "充电口附近有没有明显的划痕或磨损？",
                    "keywords": ["划痕", "磨损", "刮痕"],
                    "groups": [["划痕", "磨损", "刮痕"]],
                    "weight": 1,
                    "hint": "充电口",
                },
            ],
            contact="图书馆一楼服务台（08:00–22:00）",
            public_location="图书馆一楼服务台",
        ),
        _found(
            title="黑色有线耳机",
            description="在图书馆三楼自习区桌上捡到一副黑色有线耳机，耳机线是白色的。",
            category="耳机",
            color="黑色",
            brand="",
            location="图书馆三楼自习区",
            when=_at(y, 15, 10),
            marks=["白色耳机线"],
            secrets=[
                {"q": "耳机线是什么颜色？", "keywords": ["白色"], "weight": 2, "hint": "耳机线"},
                {"q": "耳机插头是圆孔还是 Type-C？", "keywords": ["圆孔", "3.5", "type-c", "typec"], "weight": 1, "hint": "插头"},
            ],
            contact="图书馆一楼服务台（08:00–22:00）",
            public_location="图书馆一楼服务台",
        ),
        _found(
            title="白色 AirPods 充电盒",
            description="在教学楼 204 捡到一个白色 AirPods 充电盒，外面套着透明保护壳。",
            category="耳机盒",
            color="白色",
            brand="Apple AirPods",
            location="教学楼 204",
            when=_at(y, 10, 20),
            marks=["透明保护壳"],
            secrets=[
                {"q": "保护壳是什么颜色或材质？", "keywords": ["透明", "硅胶"], "weight": 2, "hint": "保护壳"},
                {"q": "盒盖背面有没有贴什么东西？", "keywords": ["卡通", "贴纸", "贴画"], "weight": 1, "hint": "盒盖背面"},
            ],
            contact="教学楼一楼值班室",
            public_location="教学楼一楼值班室",
        ),
        _found(
            title="蓝色膳魔师保温杯",
            description="在图书馆二楼靠墙的桌子捡到一个蓝色保温杯，杯身贴着校徽贴纸。",
            category="水杯",
            color="蓝色",
            brand="膳魔师",
            location="图书馆二楼靠墙座位",
            when=_at(y, 12, 20),
            marks=["校徽贴纸"],
            secrets=[
                {"q": "杯底有没有什么痕迹或标记？", "keywords": ["划痕", "磨损", "刻字"], "weight": 2, "hint": "杯底"},
                {"q": "杯盖内侧是什么颜色？", "keywords": ["白色", "白"], "weight": 1, "hint": "杯盖内侧"},
            ],
            contact="图书馆一楼服务台（08:00–22:00）",
            public_location="图书馆一楼服务台",
        ),
        _found(
            title="校园卡（王同学）",
            description="在一食堂二楼餐桌上捡到一张校园卡，卡套是蓝色的。",
            category="校园卡",
            color="蓝色",
            brand="",
            location="一食堂二楼",
            when=_at(0, 12, 10),
            marks=["蓝色卡套"],
            secrets=[
                {"q": "卡面上的学号后四位是多少？", "keywords": ["6621"], "weight": 2, "hint": "学号"},
                {"q": "卡套是什么颜色？", "keywords": ["蓝色", "蓝"], "weight": 1, "hint": "卡套"},
            ],
            contact="一食堂一楼服务台",
            public_location="一食堂一楼服务台",
        ),
        _found(
            title="一串钥匙（熊猫挂件）",
            description="在体育馆羽毛球馆捡到一串钥匙，上面有个熊猫挂件。",
            category="钥匙",
            color="银色",
            brand="",
            location="体育馆羽毛球馆",
            when=_at(y, 19, 0),
            marks=["熊猫挂件"],
            secrets=[
                {"q": "钥匙扣上的挂件是什么动物？", "keywords": ["熊猫"], "weight": 2, "hint": "挂件"},
                {"q": "钥匙串上一共几把钥匙？", "keywords": ["三把", "3把", "3"], "weight": 1, "hint": "数量"},
            ],
            contact="体育馆器材室",
            public_location="体育馆器材室",
        ),
        _found(
            title="灰色格子雨伞",
            description="在教学楼一楼大厅捡到一把灰色格子雨伞，伞柄是黑色的。",
            category="雨伞",
            color="灰色",
            brand="",
            location="教学楼一楼大厅",
            when=_at(-2, 18, 0),
            marks=["格子图案"],
            secrets=[
                {"q": "伞柄是什么颜色？", "keywords": ["黑色", "黑"], "weight": 1, "hint": "伞柄"},
                {"q": "伞面是什么图案？", "keywords": ["格子", "格纹"], "weight": 1, "hint": "伞面"},
            ],
            contact="教学楼一楼值班室",
            public_location="教学楼一楼值班室",
        ),
        _found(
            title="深蓝色高数笔记本",
            description="在图书馆四楼自习区捡到一本深蓝色封面的笔记本，里面记满了高数公式。",
            category="笔记本",
            color="深蓝色",
            brand="",
            location="图书馆四楼自习区",
            when=_at(y, 20, 0),
            marks=["高数公式"],
            secrets=[
                {"q": "笔记本封面是什么颜色？", "keywords": ["深蓝", "蓝色", "蓝"], "weight": 1, "hint": "封面"},
                {"q": "里面记的是哪门课的内容？", "keywords": ["高数", "高等数学", "数学"], "weight": 2, "hint": "内容"},
            ],
            contact="图书馆一楼服务台（08:00–22:00）",
            public_location="图书馆一楼服务台",
        ),
        _found(
            title="卡西欧计算器",
            description="在实验楼 302 捡到一个卡西欧计算器，背面用马克笔写了名字。",
            category="计算器",
            color="黑色",
            brand="卡西欧",
            location="实验楼 302",
            when=_at(-3, 15, 0),
            marks=["背面写名字"],
            secrets=[
                {"q": "计算器是什么品牌？", "keywords": ["卡西欧", "casio"], "weight": 1, "hint": "品牌"},
                {"q": "背面写了什么？", "keywords": ["名字", "姓名", "字"], "weight": 2, "hint": "背面"},
            ],
            contact="实验楼 302 值班室",
            public_location="实验楼 302 值班室",
        ),
        _found(
            title="红色针织围巾",
            description="在二食堂门口捡到一条红色针织围巾，两端有流苏。",
            category="围巾",
            color="红色",
            brand="",
            location="二食堂门口",
            when=_at(y, 11, 30),
            marks=["流苏"],
            secrets=[
                {"q": "围巾是什么材质或织法？", "keywords": ["针织", "毛线", "羊毛"], "weight": 1, "hint": "材质"},
                {"q": "围巾两端有没有流苏？", "keywords": ["有", "流苏"], "weight": 1, "hint": "两端"},
            ],
            contact="二食堂一楼服务台",
            public_location="二食堂一楼服务台",
        ),
        _found(
            title="黑色双肩包",
            description="在图书馆二楼大厅捡到一个黑色双肩包，拉链上挂着一个绿色青蛙挂件。",
            category="书包",
            color="黑色",
            brand="",
            location="图书馆二楼大厅",
            when=_at(0, 9, 0),
            marks=["绿色青蛙挂件"],
            secrets=[
                {"q": "包里有没有电子产品？", "keywords": ["电脑", "笔记本", "平板"], "weight": 2, "hint": "包内"},
                {"q": "拉链上的挂件是什么动物？", "keywords": ["青蛙"], "weight": 1, "hint": "挂件"},
            ],
            contact="图书馆一楼服务台（08:00–22:00）",
            public_location="图书馆一楼服务台",
        ),
        _found(
            title="白色运动水壶",
            description="在操场看台捡到一个白色运动水壶，瓶盖是蓝色的。",
            category="水杯",
            color="白色",
            brand="",
            location="操场看台",
            when=_at(y, 17, 0),
            marks=["蓝色瓶盖"],
            secrets=[
                {"q": "瓶盖是什么颜色？", "keywords": ["蓝色", "蓝"], "weight": 1, "hint": "瓶盖"},
                {"q": "瓶身有没有贴纸或记号？", "keywords": ["校徽", "贴纸", "记号"], "weight": 1, "hint": "瓶身"},
            ],
            contact="操场器材室",
            public_location="操场器材室",
        ),
        _lost(
            title="蓝色保温杯",
            description="昨天中午在图书馆二楼丢了一个蓝色保温杯，杯身有校徽贴纸。",
            category="水杯",
            color="蓝色",
            brand="膳魔师",
            location="图书馆二楼",
            when=_at(y, 12, 10),
            marks=["校徽贴纸"],
        ),
        _lost(
            title="校园卡",
            description="今天中午在一食堂丢了一张校园卡，卡套是蓝色的。",
            category="校园卡",
            color="蓝色",
            brand="",
            location="一食堂",
            when=_at(0, 12, 20),
            marks=["蓝色卡套"],
            status="matched",
        ),
        _lost(
            title="黑色双肩包",
            description="今天早上在图书馆二楼丢了一个黑色双肩包，拉链上有个绿色青蛙挂件。",
            category="书包",
            color="黑色",
            brand="",
            location="图书馆二楼",
            when=_at(0, 8, 40),
            marks=["绿色青蛙挂件"],
        ),
    ]
    return cases


def reset_demo(verbose: bool = False) -> int:
    db.init_db()
    db.clear_all()
    count = 0
    for item in build_cases():
        db.create_case(item)
        count += 1
    db.add_event(None, None, "system", f"演示数据已重置，共 {count} 条案件")
    if verbose:
        print(f"seeded {count} cases")
    return count


if __name__ == "__main__":
    reset_demo(verbose=True)
