#!/usr/bin/env python3
"""校园失物招领 Agent · Web 服务。

零第三方依赖：stdlib http.server + SQLite + 原生 LLM tool calling。
启动：
    python3 server.py
    LF_WEB_PORT=8080 python3 server.py
"""
from __future__ import annotations

import hmac
import json
import mimetypes
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
sys.path.insert(0, str(ROOT))

from agent import config, db, seed  # noqa: E402
from agent.admin_agent import handle as admin_handle  # noqa: E402
from agent.admin_agent import store as admin_store  # noqa: E402
from agent.llm import llm  # noqa: E402
from agent.react_agent import handle as student_handle  # noqa: E402
from agent.react_agent import store as student_store  # noqa: E402


def _case_row(case: Dict[str, Any]) -> Dict[str, Any]:
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
        "claim_count": case.get("claim_count", 0),
        "created_at": case.get("created_at"),
        "updated_at": case.get("updated_at"),
    }


def _claim_row(claim: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": claim.get("id"),
        "case_id": claim.get("case_id"),
        "case_title": claim.get("case_title"),
        "case_role": claim.get("case_role"),
        "case_status": claim.get("case_status"),
        "status": claim.get("status"),
        "score": claim.get("score"),
        "code": claim.get("code"),
        "created_at": claim.get("created_at"),
    }


def admin_overview() -> Dict[str, Any]:
    return {
        "stats": db.admin_stats(),
        "cases": [_case_row(c) for c in db.search_cases_admin(limit=100)],
        "claims": [_claim_row(c) for c in db.search_claims_admin(limit=100)],
        "events": db.search_events_admin(limit=40),
    }


def admin_case_detail(case_id: str) -> Optional[Dict[str, Any]]:
    case = db.get_case(case_id)
    if not case:
        return None
    return {
        "case": case,
        "claims": db.claims_for_case(case_id),
        "events": db.events_for_case(case_id),
    }


class ThreadingServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server_version = "LostFoundAgent/0.3"

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("LF_WEB_QUIET") == "1":
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---------------------------------------------------------------- helpers

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 2_000_000:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _admin_ok(self, session_id: str) -> bool:
        session = admin_store.get(session_id)
        return bool(session and session.get("is_admin"))

    def _serve_file(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        rel = path.lstrip("/")
        if rel.startswith("static/"):
            rel = rel[len("static/"):]
        candidate = (WEB_DIR / rel).resolve()
        try:
            candidate.relative_to(WEB_DIR.resolve())
        except ValueError:
            self._send_json({"error": "forbidden"}, 403)
            return
        if not candidate.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        body = candidate.read_bytes()
        mime = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        if candidate.suffix == ".js":
            mime = "application/javascript"
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---------------------------------------------------------------- GET

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "llm": llm.available,
                    "model": config.LLM_MODEL,
                    "base_url": config.LLM_BASE_URL,
                    "admin_enabled": bool(config.ADMIN_PASSWORD),
                    "case_count": db.admin_stats()["total_cases"],
                }
            )
            return
        if path == "/api/admin/overview":
            qs = parse_qs(parsed.query)
            session_id = (qs.get("session_id") or [""])[0]
            if not self._admin_ok(session_id):
                self._send_json({"error": "unauthorized"}, 401)
                return
            self._send_json(admin_overview())
            return
        if path == "/api/admin/case":
            qs = parse_qs(parsed.query)
            session_id = (qs.get("session_id") or [""])[0]
            case_id = (qs.get("case_id") or [""])[0]
            if not self._admin_ok(session_id):
                self._send_json({"error": "unauthorized"}, 401)
                return
            detail = admin_case_detail(case_id)
            if not detail:
                self._send_json({"error": "not found"}, 404)
                return
            session = admin_store.get(session_id) or {}
            db.add_event(
                session_id,
                case_id,
                "admin_view_case",
                f"管理员 {session.get('name', '')} 通过管理台查看案件详情",
            )
            self._send_json(detail)
            return
        self._serve_file(path)

    # ---------------------------------------------------------------- POST

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            data = self._read_json()
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"请求体不合法：{exc}"}, 400)
            return

        try:
            if path == "/api/chat":
                result = student_handle(
                    data.get("session_id") or None,
                    str(data.get("message") or ""),
                )
                self._send_json(result)
                return

            if path == "/api/admin/login":
                password = str(data.get("password") or "")
                if not config.ADMIN_PASSWORD:
                    self._send_json({"error": "管理员密码未配置"}, 503)
                    return
                if not hmac.compare_digest(password, config.ADMIN_PASSWORD):
                    self._send_json({"error": "密码错误"}, 401)
                    return
                session = admin_store.create(config.ADMIN_NAME)
                self._send_json(
                    {
                        "ok": True,
                        "session_id": session["id"],
                        "name": config.ADMIN_NAME,
                        "default_password": config.ADMIN_PASSWORD == "admin123",
                    }
                )
                return

            if path == "/api/admin/session":
                session_id = str(data.get("session_id") or "")
                if not self._admin_ok(session_id):
                    self._send_json({"error": "unauthorized"}, 401)
                    return
                session = admin_store.create(config.ADMIN_NAME)
                self._send_json(
                    {
                        "ok": True,
                        "session_id": session["id"],
                        "name": config.ADMIN_NAME,
                    }
                )
                return

            if path == "/api/admin/chat":
                session_id = str(data.get("session_id") or "")
                if not self._admin_ok(session_id):
                    self._send_json({"error": "unauthorized"}, 401)
                    return
                result = admin_handle(session_id, str(data.get("message") or ""))
                self._send_json(result)
                return

            if path == "/api/reset":
                count = seed.reset_demo()
                student_store.reset()
                admin_store.reset()
                self._send_json({"ok": True, "count": count})
                return

            self._send_json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json({"error": f"服务端异常：{exc}"}, 500)


def main() -> None:
    port = int(os.environ.get("LF_WEB_PORT", "8000"))
    host = os.environ.get("LF_WEB_HOST", "127.0.0.1")
    db.init_db()
    if not db.list_cases(limit=1):
        seed.reset_demo()
    server = ThreadingServer((host, port), Handler)
    print("=" * 64)
    print("  寻物有迹 · 校园失物招领 Agent")
    print(f"  http://{host}:{port}")
    print(f"  模型：{config.LLM_MODEL}（{'在线' if llm.available else '不可用'}）")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
