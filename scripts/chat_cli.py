#!/usr/bin/env python3
"""统一对话入口：学生端 / 管理端在同一个会话里用口令切换。

用法：
    python3 scripts/chat_cli.py
    python3 scripts/chat_cli.py --reset

进入后：
    /admin      输入管理员密码，进入管理端
    /exitadmin  返回学生端
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import config, db, seed  # noqa: E402
from agent.admin_agent import handle as admin_handle  # noqa: E402
from agent.admin_agent import store as admin_store  # noqa: E402
from agent.llm import llm  # noqa: E402
from agent.react_agent import handle as student_handle  # noqa: E402
from agent.react_agent import store as student_store  # noqa: E402


def print_trace(trace) -> None:
    if not trace:
        return
    symbols = {
        "llm": "🧠", "tool": "⚙", "error": "✖",
        "perceive": "◉", "plan": "◇", "observe": "·", "reflect": "↻",
    }
    for item in trace:
        symbol = symbols.get(item.get("type"), "·")
        print(f"    {symbol} [{item.get('title', '')}] {item.get('detail', '')}")


def model_mode() -> str:
    if llm.available:
        return f"在线 LLM：{config.LLM_MODEL} @ {config.LLM_BASE_URL}"
    if llm.disabled:
        return "模型已熔断（连续 API 失败）"
    return "模型不可用（未配置 LLM_API_KEY；不会降级到规则）"


def admin_login() -> bool:
    if not config.ADMIN_PASSWORD:
        print("[管理端] 未配置 ADMIN_PASSWORD，无法进入。")
        return False
    if config.ADMIN_PASSWORD == "admin123":
        print("⚠ 当前使用默认管理员密码 admin123，正式使用请修改 .env。")
    for attempt in range(3):
        try:
            entered = getpass.getpass("管理员密码：")
        except (EOFError, KeyboardInterrupt):
            print("\n[已取消管理员登录]")
            return False
        if entered == config.ADMIN_PASSWORD:
            return True
        print(f"密码错误（{attempt + 1}/3）")
    print("[管理员登录失败，仍在学生端]")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="重置演示数据")
    parser.add_argument("--show-trace", action="store_true", default=True)
    args = parser.parse_args()

    db.init_db()
    if args.reset or not db.list_cases(limit=1):
        count = seed.reset_demo()
        print(f"[已重置演示数据，共 {count} 条案件]\n")

    print("=" * 68)
    print("  校园失物招领 Agent · 小寻  (CLI 调试版)")
    print(f"  模型模式：{model_mode()}")
    print("  输入 /admin 进入管理端，/exitadmin 返回学生端")
    print("  /reset 重置数据，/trace 轨迹开关，/mode 查看模型，/quit 退出")
    print("=" * 68)
    print("  提示：先试拾得登记，再开新会话试失主认领")
    print()

    mode = "student"  # student | admin
    session_id = None
    admin_session_id = None
    show_trace = args.show_trace

    while True:
        prompt = "管理台 > " if mode == "admin" else "你 > "
        try:
            text = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not text:
            continue
        if text in ("/quit", "/exit", "q"):
            print("再见。")
            break

        if text == "/admin":
            if mode == "admin":
                print("[已经在管理端；用 /exitadmin 返回学生端]\n")
                continue
            if not llm.available:
                print(f"[管理端不可用] {model_mode()}\n")
                continue
            if not admin_login():
                print()
                continue
            if admin_session_id is None:
                admin_session_id = admin_store.create(config.ADMIN_NAME)["id"]
            mode = "admin"
            print(f"\n[已进入管理端 · 管理员：{config.ADMIN_NAME}]")
            print("  直接用自然语言查询/修改；写操作会先给预览，等你确认后再执行。")
            print("  /exitadmin 返回学生端。\n")
            continue

        if text == "/exitadmin":
            if mode == "admin":
                mode = "student"
                print("[已返回学生端]\n")
            else:
                print("[当前就在学生端]\n")
            continue

        if text == "/reset":
            count = seed.reset_demo()
            student_store.reset()
            admin_store.reset()
            session_id = None
            admin_session_id = None
            if mode == "admin":
                admin_session_id = admin_store.create(config.ADMIN_NAME)["id"]
            print(f"[已重置 {count} 条案件；学生端/管理端会话已清空]\n")
            continue

        if text == "/mode":
            print(f"[当前入口] {'管理端' if mode == 'admin' else '学生端'}")
            print(f"[模型模式] {model_mode()}")
            print(f"[接口] {config.LLM_BASE_URL}")
            print(f"[失败次数] {llm.failures}")
            print(f"[错误] {llm.last_error or '无'}\n")
            continue

        if text == "/trace":
            show_trace = not show_trace
            print(f"[决策轨迹：{'显示' if show_trace else '隐藏'}]\n")
            continue

        if text == "/new":
            if mode == "admin":
                admin_session_id = admin_store.create(config.ADMIN_NAME)["id"]
                print("[已开启新的管理会话]\n")
            else:
                session_id = None
                print("[已开启新的学生会话]\n")
            continue

        if text == "/help":
            print("  /admin 进入管理端  /exitadmin 返回学生端")
            print("  /new 新会话  /reset 重置数据  /trace 轨迹开关  /mode 模型状态  /quit 退出\n")
            continue

        if mode == "admin":
            result = admin_handle(admin_session_id, text)
        else:
            result = student_handle(session_id, text)
            session_id = result["session"]["id"]

        if result.get("error"):
            print("\n" + "!" * 60)
            print(f"  模型服务不可用：{result['error'].get('message')}")
            print("!" * 60)
        speaker = "管理台" if mode == "admin" else "小寻"
        print(f"\n{speaker} > {result.get('reply', '')}")
        if show_trace:
            print_trace(result.get("trace") or [])
        for card in result.get("cards") or []:
            if card.get("type") == "candidate":
                print(f"    [候选] {card['title']}  {card['score']:.2f}  {' / '.join(card.get('reasons', []))}")
            elif card.get("type") == "ticket":
                print(f"    [认领凭证] {card['code']}  {card['location']}")
            elif card.get("type") == "case_created":
                print(f"    [案件] {card['id']}  {card['title']}")
        if result.get("quick_replies"):
            print("    快捷回复：" + " | ".join(result["quick_replies"]))
        print()


if __name__ == "__main__":
    main()
