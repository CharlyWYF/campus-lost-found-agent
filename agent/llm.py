"""OpenAI 兼容的 LLM 客户端；没有 Key 时全部走规则引擎。"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from . import config


class LLMError(RuntimeError):
    """模型调用失败。新架构下不允许静默回退到规则。"""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


class LLMClient:
    def __init__(self) -> None:
        self.api_key = config.LLM_API_KEY
        self.base_url = config.LLM_BASE_URL.rstrip("/")
        self.model = config.LLM_MODEL
        self.timeout = config.LLM_TIMEOUT
        self.last_error = ""
        self.failures = 0
        self.disabled = False
        self.max_failures = 2

    @property
    def available(self) -> bool:
        return bool(self.api_key) and config.llm_enabled() and not self.disabled

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        if not self.available:
            return None
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            return self._post(payload, timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            self.last_error = f"HTTP {exc.code}: {detail}"
            # 部分网关不支持 response_format，自动降级重试一次
            if json_mode and exc.code == 400:
                payload.pop("response_format", None)
                try:
                    return self._post(payload, timeout)
                except Exception as retry_exc:  # noqa: BLE001
                    self.last_error = f"{type(retry_exc).__name__}: {retry_exc}"
            self._record_failure()
        except Exception as exc:  # noqa: BLE001 - 网络不可用时必须优雅降级
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._record_failure()
        return None

    def _record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.max_failures:
            self.disabled = True

    def reset_breaker(self) -> None:
        self.failures = 0
        self.disabled = False
        self.last_error = ""

    def _post_raw(self, payload: Dict[str, Any], timeout: Optional[float]) -> Dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "campus-lostfound-agent/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)

    def _post(self, payload: Dict[str, Any], timeout: Optional[float]) -> str:
        parsed = self._post_raw(payload, timeout)
        content = parsed["choices"][0]["message"]["content"]
        self.failures = 0
        return content

    def chat_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """原生 tool calling，返回完整 message（含 tool_calls / reasoning_content）。

        这里故意不吞异常：新架构要求 API 失败必须显式报错。
        """
        if not self.available:
            detail = "已熔断" if self.disabled else "未配置 LLM_API_KEY"
            raise LLMError(f"模型服务不可用：{detail}", self.last_error)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            parsed = self._post_raw(payload, timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            self.last_error = f"HTTP {exc.code}: {detail}"
            self._record_failure()
            raise LLMError(f"模型接口返回 HTTP {exc.code}", self.last_error) from exc
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._record_failure()
            raise LLMError("模型调用失败", self.last_error) from exc
        self.failures = 0
        try:
            return parsed["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("模型返回格式异常", str(parsed)[:500]) from exc

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.15,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        content = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            json_mode=True,
            timeout=timeout,
        )
        if not content:
            return None
        return _parse_json_object(content)

    def chat_json_strict(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        message = self.chat_message(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            timeout=timeout,
        )
        content = message.get("content") or ""
        parsed = _parse_json_object(content)
        if parsed is None:
            raise LLMError("模型未返回合法 JSON", content[:500])
        return parsed

    def timed_json(self, system: str, user: str) -> tuple[Optional[Dict[str, Any]], int]:
        started = time.time()
        result = self.chat_json(system, user)
        return result, int((time.time() - started) * 1000)


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start:end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


llm = LLMClient()
