"""HTTP-backed provider transport for resource-scoped JSON providers."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from argos_src.provider_api.errors import ProviderError, ProviderTimeout
from argos_src.provider_api.manifest import ProviderManifest
from argos_src.provider_api.namespaces import (
    normalize_provider_prefix,
    provider_resource_prefix,
)
from argos_src.provider_api.wire import (
    OP_DISPLAY_AWAIT_RESPONSE,
    OP_DISPLAY_COMMAND,
    OP_DISPLAY_HEALTH,
    OP_DISPLAY_IMAGE,
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
        key_prefix: str | None = None,
        timeout_ms: int | None = None,
        resource_id: str | None = None,
        manifest: ProviderManifest | None = None,
        auth_token_env: str | None = None,
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
        self.key_prefix = normalize_provider_prefix(
            key_prefix or os.getenv("ARGOS_HTTP_PROVIDER_KEY_PREFIX", "")
        )
        self.timeout_ms = int(timeout_ms or DEFAULT_HTTP_TIMEOUT_MS)
        if self.timeout_ms <= 0:
            raise ValueError("HTTP provider timeout must be > 0")
        self._resource_id = str(resource_id or "").strip()
        if not self._resource_id:
            raise ValueError("HTTP provider transport requires resource_id.")
        self._manifest = manifest
        self._auth_token_env = str(auth_token_env or "").strip()
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
    ) -> Any:
        op = str(operation or "").strip()
        payload = dict(args or {})
        rendered_resource_id = self._effective_resource_id(resource_id)
        rendered_timeout_ms = int(
            timeout_ms or payload.pop("timeout_ms", self.timeout_ms)
        )
        if op == OP_DISPLAY_COMMAND:
            return self._post_json(
                self._resource_path(rendered_resource_id, "display"),
                payload,
                timeout_ms=rendered_timeout_ms,
            )
        if op == OP_DISPLAY_IMAGE:
            return self._post_json(
                self._resource_path(rendered_resource_id, "image"),
                payload,
                timeout_ms=rendered_timeout_ms,
            )
        if op == OP_DISPLAY_AWAIT_RESPONSE:
            return self._await_response(
                payload,
                timeout_ms=rendered_timeout_ms,
                resource_id=rendered_resource_id,
            )
        operation_name = self._http_operation_name(rendered_resource_id, op)
        if operation_name == "state" or op == OP_DISPLAY_STATE:
            return self._get_json(
                self._resource_path(rendered_resource_id, "state"),
                timeout_ms=rendered_timeout_ms,
            )
        if operation_name == "health" or op == OP_DISPLAY_HEALTH:
            return self._get_json(
                self._resource_path(rendered_resource_id, "health"),
                timeout_ms=rendered_timeout_ms,
            )
        return self._post_json(
            self._resource_path(
                rendered_resource_id,
                f"request/{operation_name}",
            ),
            payload,
            timeout_ms=rendered_timeout_ms,
        )

    def publish_event(
        self,
        *,
        resource_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        rendered_event_type = str(event_type or "").strip()
        if rendered_event_type:
            self.request(
                resource_id=resource_id,
                operation=rendered_event_type,
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

    def _effective_resource_id(self, resource_id: str | None = None) -> str:
        rendered = str(resource_id or self._resource_id or "").strip()
        if not rendered:
            raise ValueError("resource_id must not be empty")
        return rendered

    def _resource_path(self, resource_id: str, leaf: str) -> str:
        rendered_leaf = str(leaf or "").strip().strip("/")
        if not rendered_leaf:
            raise ValueError("HTTP provider path leaf must not be empty")
        return (
            f"/{provider_resource_prefix(self.key_prefix, resource_id)}/"
            f"{rendered_leaf}"
        )

    def _http_operation_name(self, resource_id: str, operation: str) -> str:
        rendered = str(operation or "").strip().strip("/")
        if not rendered:
            raise ProviderError("HTTP provider operation must not be empty.")
        resource = (
            self._manifest.resource_by_id(resource_id)
            if self._manifest is not None
            else None
        )
        resource_kind = str(getattr(resource, "kind", "") or "").strip()
        prefix = f"{resource_kind}."
        if resource_kind and rendered.startswith(prefix):
            rendered = rendered[len(prefix) :]
        elif "." in rendered:
            rendered = rendered.split(".", 1)[1]
        return rendered.replace(".", "_").strip("/")

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout_ms: int,
    ) -> Any:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            self._url(path),
            data=data,
            headers=self._headers(content_type="application/json"),
            method="POST",
        )
        return self._read_json(request, timeout_ms=timeout_ms)

    def _get_json(self, path: str, *, timeout_ms: int) -> Any:
        request = Request(
            self._url(path),
            headers=self._headers(),
            method="GET",
        )
        return self._read_json(request, timeout_ms=timeout_ms)

    def _headers(self, *, content_type: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = content_type
        if self._auth_token_env:
            token = os.getenv(self._auth_token_env, "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    def _read_json(self, request: Request, *, timeout_ms: int) -> Any:
        try:
            with self._urlopen(
                request,
                timeout=max(0.001, timeout_ms / 1000.0),
            ) as response:
                raw = response.read()
        except HTTPError as exc:
            if exc.code == 408:
                raise ProviderTimeout("HTTP provider request timed out") from exc
            raise ProviderError(f"HTTP provider request failed: {exc}") from exc
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
        return decoded

    def _await_response(
        self,
        payload: dict[str, Any],
        *,
        timeout_ms: int,
        resource_id: str,
    ) -> Any:
        request_id = str(
            payload.get("requestId") or payload.get("request_id") or ""
        ).strip()
        if not request_id:
            raise ProviderError("display.await_response requires requestId.")
        deadline = time.time() + (timeout_ms / 1000.0)
        poll_sec = float(payload.get("poll_sec") or DEFAULT_AWAIT_POLL_SEC)
        path = self._resource_path(resource_id, "response")
        while time.time() < deadline:
            response = self._get_json(path, timeout_ms=min(timeout_ms, 1000))
            if self._response_matches(response, request_id):
                return response
            time.sleep(max(0.05, poll_sec))
        raise ProviderTimeout(
            f"Timed out waiting for display response requestId={request_id}"
        )

    @staticmethod
    def _response_matches(response: Any, request_id: str) -> bool:
        if not isinstance(response, dict):
            return False
        rendered_request_id = str(
            response.get("requestId") or response.get("request_id") or ""
        ).strip()
        if rendered_request_id == request_id:
            return True
        nested = response.get("response")
        if isinstance(nested, dict):
            nested_request_id = str(
                nested.get("requestId") or nested.get("request_id") or ""
            ).strip()
            return nested_request_id == request_id
        return False

__all__ = ["HttpProviderClient"]
