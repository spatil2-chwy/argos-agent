"""HTTP-backed provider transport for local display resources."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from argos_src.provider_api.errors import ProviderError, ProviderTimeout
from argos_src.provider_api.manifest import ProviderManifest
from argos_src.provider_api.wire import (
    OP_DISPLAY_AWAIT_RESPONSE,
    OP_DISPLAY_COMMAND,
    OP_DISPLAY_HEALTH,
    OP_DISPLAY_STATE,
)


DEFAULT_HTTP_TIMEOUT_MS = 3000
DEFAULT_AWAIT_POLL_SEC = 0.25


class HttpProviderClient:
    """Provider client for simple local HTTP capability servers."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        connect_endpoints: list[str] | tuple[str, ...] | None = None,
        timeout_ms: int | None = None,
        resource_id: str | None = None,
        manifest: ProviderManifest | None = None,
        urlopen_fn: Callable[..., Any] | None = None,
    ) -> None:
        endpoint = (
            base_url
            or (connect_endpoints[0] if connect_endpoints else "")
            or os.getenv("ARGOS_HTTP_PROVIDER_BASE_URL", "")
        )
        self.base_url = str(endpoint or "").strip().rstrip("/")
        if not self.base_url:
            raise ValueError("HTTP provider transport requires a base URL endpoint.")
        self.timeout_ms = int(timeout_ms or DEFAULT_HTTP_TIMEOUT_MS)
        if self.timeout_ms <= 0:
            raise ValueError("HTTP provider timeout must be > 0")
        self._resource_id = str(resource_id or "").strip()
        self._manifest = manifest
        self._urlopen = urlopen_fn or urlopen

    def start(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def get_manifest(self) -> ProviderManifest | None:
        return self._manifest

    def request(
        self,
        *,
        resource_id: str,
        operation: str,
        args: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        del resource_id
        op = str(operation or "").strip()
        payload = dict(args or {})
        rendered_timeout_ms = int(timeout_ms or payload.pop("timeout_ms", self.timeout_ms))
        if op == OP_DISPLAY_COMMAND:
            return self._post_json("/display", payload, timeout_ms=rendered_timeout_ms)
        if op == OP_DISPLAY_STATE:
            return self._get_json("/state", timeout_ms=rendered_timeout_ms)
        if op == OP_DISPLAY_HEALTH:
            return self._get_json("/health", timeout_ms=rendered_timeout_ms)
        if op == OP_DISPLAY_AWAIT_RESPONSE:
            return self._await_response(payload, timeout_ms=rendered_timeout_ms)
        raise ProviderError(f"Unsupported HTTP provider operation: {op}")

    def publish_event(
        self,
        *,
        resource_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        del resource_id
        if str(event_type or "").strip() == OP_DISPLAY_COMMAND:
            self.request(
                resource_id=self._resource_id,
                operation=OP_DISPLAY_COMMAND,
                args=dict(data or {}),
            )

    def subscribe_event(
        self,
        *,
        resource_id: str,
        event_type: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> Callable[[], None]:
        del resource_id, event_type, callback
        return lambda: None

    def _url(self, path: str) -> str:
        return urljoin(f"{self.base_url}/", path.lstrip("/"))

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout_ms: int,
    ) -> dict[str, Any]:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            self._url(path),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._read_json(request, timeout_ms=timeout_ms)

    def _get_json(self, path: str, *, timeout_ms: int) -> dict[str, Any]:
        request = Request(self._url(path), method="GET")
        return self._read_json(request, timeout_ms=timeout_ms)

    def _read_json(self, request: Request, *, timeout_ms: int) -> dict[str, Any]:
        try:
            with self._urlopen(request, timeout=max(0.001, timeout_ms / 1000.0)) as response:
                raw = response.read()
        except URLError as exc:
            raise ProviderError(f"HTTP provider request failed: {exc}") from exc
        except TimeoutError as exc:
            raise ProviderTimeout("HTTP provider request timed out") from exc
        except Exception as exc:
            raise ProviderError(f"HTTP provider request failed: {exc}") from exc
        if not raw:
            return {"ok": True}
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ProviderError(f"HTTP provider returned invalid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ProviderError("HTTP provider response must be a JSON object")
        return decoded

    def _await_response(
        self,
        payload: dict[str, Any],
        *,
        timeout_ms: int,
    ) -> dict[str, Any]:
        request_id = str(payload.get("requestId") or payload.get("request_id") or "").strip()
        if not request_id:
            raise ProviderError("display.await_response requires requestId.")
        deadline = time.time() + (timeout_ms / 1000.0)
        poll_sec = float(payload.get("poll_sec") or DEFAULT_AWAIT_POLL_SEC)
        while time.time() < deadline:
            response = self._get_json("/response", timeout_ms=min(timeout_ms, 1000))
            if self._response_matches(response, request_id):
                return response
            time.sleep(max(0.05, poll_sec))
        raise ProviderTimeout(
            f"Timed out waiting for display response requestId={request_id}"
        )

    @staticmethod
    def _response_matches(response: dict[str, Any], request_id: str) -> bool:
        if str(response.get("requestId") or response.get("request_id") or "").strip() == request_id:
            return True
        nested = response.get("response")
        if isinstance(nested, dict):
            return (
                str(nested.get("requestId") or nested.get("request_id") or "").strip()
                == request_id
            )
        return False


__all__ = ["HttpProviderClient"]
