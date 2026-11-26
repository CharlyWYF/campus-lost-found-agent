"""SQLite 持久层：案件、认领记录与审计事件。"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from . import config

_LOCK = threading.RLock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    category TEXT,
    title TEXT,
    description TEXT,
    color TEXT,
    brand TEXT,
    location TEXT,
    time_text TEXT,
    time_iso TEXT,
    marks TEXT DEFAULT '[]',
    secrets TEXT DEFAULT '[]',
    contact TEXT,
    public_location TEXT,
    session_id TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    session_id TEXT,
    answers TEXT DEFAULT '[]',
    score REAL DEFAULT 0,
    status TEXT,
    code TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    case_id TEXT,
    type TEXT,
    detail TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_cases_role_status ON cases(role, status);
CREATE INDEX IF NOT EXISTS idx_claims_case ON claims(case_id);
"""


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _LOCK, connect() as conn:
        conn.executescript(_SCHEMA)


def _loads(value: Any, fallback):
    if value is None or value == "":
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def row_to_case(row: sqlite3.Row) -> Dict[str, Any]:
    case = dict(row)
    case["marks"] = _loads(case.get("marks"), [])
    case["secrets"] = _loads(case.get("secrets"), [])
    return case


def public_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """给失主/候选人看的版本：不含验证细节、联系方式与具体保管点。"""
    return {
        "id": case["id"],
        "role": case["role"],
        "status": case["status"],
        "category": case.get("category"),
        "title": case.get("title"),
        "description": case.get("description"),
        "color": case.get("color"),
        "brand": case.get("brand"),
        "location": case.get("location"),
        "time_text": case.get("time_text"),
        "marks": case.get("marks", []),
        "created_at": case.get("created_at"),
    }


def create_case(data: Dict[str, Any]) -> Dict[str, Any]:
    role = data.get("role", "found")
    stamp = config.now()
    prefix = "F" if role == "found" else "L"
    day = stamp.strftime("%Y%m%d")
    with _LOCK, connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE id LIKE ?", (f"{prefix}-{day}-%",)
        ).fetchone()[0]
        case_id = data.get("id") or f"{prefix}-{day}-{count + 1:03d}"
        now_iso = data.get("created_at") or stamp.isoformat(timespec="seconds")
        conn.execute(
            """INSERT INTO cases
            (id, role, status, category, title, description, color, brand,
             location, time_text, time_iso, marks, secrets, contact,
             public_location, session_id, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                case_id,
                role,
                data.get("status", "open"),
                data.get("category"),
                data.get("title"),
                data.get("description"),
                data.get("color"),
                data.get("brand"),
                data.get("location"),
                data.get("time_text"),
                data.get("time_iso"),
                _dumps(data.get("marks", [])),
                _dumps(data.get("secrets", [])),
                data.get("contact"),
                data.get("public_location"),
                data.get("session_id"),
                now_iso,
                now_iso,
            ),
        )
        row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        return row_to_case(row)


def update_case(case_id: str, **fields) -> Optional[Dict[str, Any]]:
    if not fields:
        return get_case(case_id)
    allowed = {
        "role", "status", "category", "title", "description", "color", "brand",
        "location", "time_text", "time_iso", "marks", "secrets", "contact",
        "public_location", "session_id",
    }
    sets, values = [], []
    for key, value in fields.items():
        if key not in allowed:
            continue
        sets.append(f"{key}=?")
        values.append(_dumps(value) if key in {"marks", "secrets"} else value)
    if not sets:
        return get_case(case_id)
    sets.append("updated_at=?")
    values.append(config.now().isoformat(timespec="seconds"))
    values.append(case_id)
    with _LOCK, connect() as conn:
        conn.execute(f"UPDATE cases SET {', '.join(sets)} WHERE id=?", values)
        row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        return row_to_case(row) if row else None


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        return row_to_case(row) if row else None


def list_cases(
    role: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM cases WHERE 1=1"
    params: List[Any] = []
    if role:
        sql += " AND role=?"
        params.append(role)
    if status:
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        return [row_to_case(r) for r in conn.execute(sql, params).fetchall()]


def create_claim(
    case_id: str,
    session_id: str,
    answers: List[Dict[str, Any]],
    score: float,
    status: str,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    stamp = config.now()
    claim_id = f"C-{stamp.strftime('%Y%m%d%H%M%S')}-{abs(hash((case_id, session_id))) % 1000:03d}"
    with _LOCK, connect() as conn:
        conn.execute(
            """INSERT INTO claims (id, case_id, session_id, answers, score, status, code, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                claim_id,
                case_id,
                session_id,
                _dumps(answers),
                round(score, 4),
                status,
                code,
                stamp.isoformat(timespec="seconds"),
            ),
        )
    return {
        "id": claim_id,
        "case_id": case_id,
        "session_id": session_id,
        "answers": answers,
        "score": round(score, 4),
        "status": status,
        "code": code,
        "created_at": stamp.isoformat(timespec="seconds"),
    }


def list_claims(limit: int = 100) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM claims ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    claims = []
    for row in rows:
        item = dict(row)
        item["answers"] = _loads(item.get("answers"), [])
        claims.append(item)
    return claims


def add_event(
    session_id: Optional[str],
    case_id: Optional[str],
    event_type: str,
    detail: str,
) -> None:
    stamp = config.now().isoformat(timespec="seconds")
    with _LOCK, connect() as conn:
        conn.execute(
            "INSERT INTO events (session_id, case_id, type, detail, created_at) VALUES (?,?,?,?,?)",
            (session_id, case_id, event_type, detail, stamp),
        )


def list_events(limit: int = 80) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def stats() -> Dict[str, Any]:
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        by_status = {
            row["status"]: row["n"]
            for row in conn.execute(
                "SELECT status, COUNT(*) AS n FROM cases GROUP BY status"
            ).fetchall()
        }
        by_role = {
            row["role"]: row["n"]
            for row in conn.execute(
                "SELECT role, COUNT(*) AS n FROM cases GROUP BY role"
            ).fetchall()
        }
        claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        accepted = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE status='accepted'"
        ).fetchone()[0]
        rejected = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE status='rejected'"
        ).fetchone()[0]
    return {
        "total_cases": total,
        "by_status": by_status,
        "by_role": by_role,
        "claims": claims,
        "claims_accepted": accepted,
        "claims_rejected": rejected,
    }


def clear_all() -> None:
    with _LOCK, connect() as conn:
        conn.execute("DELETE FROM cases")
        conn.execute("DELETE FROM claims")
        conn.execute("DELETE FROM events")


def _dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# ------------------------------------------------------------------ 管理员查询

def search_cases_admin(
    keyword: str = "",
    role: str = "",
    status: str = "",
    category: str = "",
    location: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 30,
) -> List[Dict[str, Any]]:
    sql = (
        "SELECT c.*, "
        "(SELECT COUNT(*) FROM claims cl WHERE cl.case_id=c.id) AS claim_count "
        "FROM cases c WHERE 1=1"
    )
    params: List[Any] = []
    if role:
        sql += " AND c.role=?"
        params.append(role)
    if status:
        sql += " AND c.status=?"
        params.append(status)
    if category:
        sql += " AND c.category LIKE ?"
        params.append(f"%{category}%")
    if location:
        sql += " AND c.location LIKE ?"
        params.append(f"%{location}%")
    if keyword:
        like = f"%{keyword}%"
        sql += (
            " AND (c.title LIKE ? OR c.description LIKE ? OR c.location LIKE ?"
            " OR c.category LIKE ? OR c.color LIKE ? OR c.brand LIKE ? OR c.marks LIKE ?)"
        )
        params.extend([like] * 7)
    if date_from:
        sql += " AND c.created_at >= ?"
        params.append(date_from)
    if date_to:
        value = date_to + "T23:59:59" if len(date_to) == 10 else date_to
        sql += " AND c.created_at <= ?"
        params.append(value)
    sql += " ORDER BY c.created_at DESC LIMIT ?"
    params.append(max(1, min(100, int(limit))))
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [row_to_case(r) for r in rows]


def search_claims_admin(
    status: str = "",
    case_id: str = "",
    code: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 30,
) -> List[Dict[str, Any]]:
    sql = (
        "SELECT cl.*, c.title AS case_title, c.role AS case_role, "
        "c.location AS case_location, c.status AS case_status "
        "FROM claims cl LEFT JOIN cases c ON c.id=cl.case_id WHERE 1=1"
    )
    params: List[Any] = []
    if status:
        sql += " AND cl.status=?"
        params.append(status)
    if case_id:
        sql += " AND cl.case_id=?"
        params.append(case_id)
    if code:
        sql += " AND cl.code LIKE ?"
        params.append(f"%{code}%")
    if date_from:
        sql += " AND cl.created_at >= ?"
        params.append(date_from)
    if date_to:
        value = date_to + "T23:59:59" if len(date_to) == 10 else date_to
        sql += " AND cl.created_at <= ?"
        params.append(value)
    sql += " ORDER BY cl.created_at DESC LIMIT ?"
    params.append(max(1, min(100, int(limit))))
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["answers"] = _loads(item.get("answers"), [])
        result.append(item)
    return result


def get_claim(claim_id: str) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT cl.*, c.title AS case_title, c.role AS case_role, "
            "c.location AS case_location, c.status AS case_status "
            "FROM claims cl LEFT JOIN cases c ON c.id=cl.case_id WHERE cl.id=?",
            (claim_id,),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["answers"] = _loads(item.get("answers"), [])
    return item


def get_claim_by_code(code: str) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT cl.*, c.title AS case_title, c.public_location, c.contact "
            "FROM claims cl LEFT JOIN cases c ON c.id=cl.case_id WHERE cl.code=?",
            (code,),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["answers"] = _loads(item.get("answers"), [])
    return item


def update_claim_status(claim_id: str, status: str) -> Optional[Dict[str, Any]]:
    with _LOCK, connect() as conn:
        conn.execute("UPDATE claims SET status=? WHERE id=?", (status, claim_id))
    return get_claim(claim_id)


def claims_for_case(case_id: str) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM claims WHERE case_id=? ORDER BY created_at DESC",
            (case_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["answers"] = _loads(item.get("answers"), [])
        result.append(item)
    return result


def events_for_case(case_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE case_id=? ORDER BY id DESC LIMIT ?",
            (case_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def search_events_admin(
    case_id: str = "",
    event_type: str = "",
    limit: int = 40,
) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM events WHERE 1=1"
    params: List[Any] = []
    if case_id:
        sql += " AND case_id=?"
        params.append(case_id)
    if event_type:
        sql += " AND type=?"
        params.append(event_type)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, min(200, int(limit))))
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def admin_stats() -> Dict[str, Any]:
    with connect() as conn:
        by_status = {
            r["status"]: r["n"]
            for r in conn.execute("SELECT status, COUNT(*) n FROM cases GROUP BY status").fetchall()
        }
        by_role = {
            r["role"]: r["n"]
            for r in conn.execute("SELECT role, COUNT(*) n FROM cases GROUP BY role").fetchall()
        }
        top_categories = [
            {"category": r["category"], "n": r["n"]}
            for r in conn.execute(
                "SELECT category, COUNT(*) n FROM cases WHERE category IS NOT NULL "
                "GROUP BY category ORDER BY n DESC LIMIT 8"
            ).fetchall()
        ]
        top_locations = [
            {"location": r["location"], "n": r["n"]}
            for r in conn.execute(
                "SELECT location, COUNT(*) n FROM cases WHERE location IS NOT NULL "
                "GROUP BY location ORDER BY n DESC LIMIT 8"
            ).fetchall()
        ]
        claims = {
            r["status"]: r["n"]
            for r in conn.execute("SELECT status, COUNT(*) n FROM claims GROUP BY status").fetchall()
        }
        total_cases = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        open_cases = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE status IN ('open','matched','verifying')"
        ).fetchone()[0]
        unresolved = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE status IN ('open','matched','verifying') "
            "AND created_at < datetime('now','-7 days')"
        ).fetchone()[0]
    return {
        "total_cases": total_cases,
        "open_cases": open_cases,
        "overdue_7d": unresolved,
        "by_status": by_status,
        "by_role": by_role,
        "claims": claims,
        "top_categories": top_categories,
        "top_locations": top_locations,
    }
