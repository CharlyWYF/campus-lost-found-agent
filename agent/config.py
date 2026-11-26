"""全局配置：路径、模型接入与演示时间。"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"
DOCS_DIR = ROOT / "docs"


def _load_dotenv() -> None:
    """极简 .env 读取，避免为一个演示项目引入额外依赖。"""
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()

HOST = os.environ.get("LF_HOST", "127.0.0.1")
PORT = int(os.environ.get("LF_PORT", "8000"))
DB_PATH = Path(os.environ.get("LF_DB_PATH", str(DATA_DIR / "lostfound.db")))

# 任意 OpenAI 兼容接口均可：DeepSeek / OpenAI / 通义 / 本地 vLLM ...
LLM_API_KEY = (
    os.environ.get("LLM_API_KEY")
    or os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "45"))
LLM_MODE = os.environ.get("LLM_MODE", "auto")  # auto | live | mock

# 管理员模式：课程 demo 默认密码，正式使用请改 .env
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
ADMIN_NAME = os.environ.get("ADMIN_NAME", "值班老师")

# 固定演示时间，便于测试与复现；不设置则使用本机当前时间。
_demo_now = os.environ.get("LF_DEMO_NOW", "").strip()


def now() -> datetime:
    if _demo_now:
        try:
            return datetime.fromisoformat(_demo_now)
        except ValueError:
            pass
    return datetime.now()


def llm_enabled() -> bool:
    if LLM_MODE == "mock":
        return False
    if LLM_MODE == "live":
        return bool(LLM_API_KEY)
    return bool(LLM_API_KEY)
