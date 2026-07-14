"""Transport-neutral provider client protocol."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from argos_src.provider_api.manifest import ProviderManifest


class ProviderClient(Protocol):
    """Low-level provider/resource request, event, and manifest contract."""

    def start(self) -> None:
        """Start transport threads or subscriptions."""

    def shutdown(self) -> None:
        """Release transport resources."""

    def get_manifest(self) -> ProviderManifest | None:
        """Return a local or provider-advertised manifest when available."""

    def request(
        self,
        *,
        resource_id: str,
        operation: str,
        args: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> Any:
        """Send one resource-scoped request and return its raw JSON result."""

    def publish_event(
        self,
        *,
        resource_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Publish one resource-scoped event."""

    def subscribe_event(
        self,
        *,
        resource_id: str,
        event_type: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> Callable[[], None]:
        """Subscribe to resource events and return an unsubscribe callback."""


__all__ = ["ProviderClient"]
