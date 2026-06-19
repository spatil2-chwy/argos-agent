"""Provider client construction helpers."""

from __future__ import annotations

import os

from argos_src.provider_api.client import ProviderClient
from argos_src.provider_api.fake import FakeProviderClient

DEFAULT_PROVIDER_TRANSPORT = "zenoh"


def create_provider_client(
    *,
    transport: str | None = None,
    key_prefix: str | None = None,
    connect_endpoints: list[str] | tuple[str, ...] | None = None,
    resource_id: str | None = None,
    manifest=None,
) -> ProviderClient:
    """Create a provider client for the selected transport."""
    selected = str(
        transport
        or os.getenv("ARGOS_PROVIDER_TRANSPORT", "")
        or DEFAULT_PROVIDER_TRANSPORT
    ).strip().lower()
    if selected == "fake":
        return FakeProviderClient()
    if selected == "http":
        if not key_prefix or not resource_id:
            raise ValueError(
                "HTTP provider transport requires manifest-derived key_prefix "
                "and resource_id."
            )
        from argos_src.provider_api.transports.http import HttpProviderClient

        return HttpProviderClient(
            connect_endpoints=connect_endpoints,
            key_prefix=key_prefix,
            resource_id=resource_id,
            manifest=manifest,
        )
    if selected in {"zenoh", "bridge"}:
        if not key_prefix or not resource_id:
            raise ValueError(
                "Zenoh provider transport requires manifest-derived key_prefix "
                "and resource_id."
            )
        from argos_src.provider_api.transports.zenoh import ZenohProviderClient

        return ZenohProviderClient(
            key_prefix=key_prefix,
            connect_endpoints=connect_endpoints,
            resource_id=resource_id,
            manifest=manifest,
        )
    if selected in {"ros", "ros2", "legacy_ros2"}:
        raise ValueError(
            "ROS transports are not supported in the Argos runtime. Run ROS/SDK "
            "code in the external provider and use ARGOS_PROVIDER_TRANSPORT=zenoh."
        )
    raise ValueError(
        f"Unsupported ARGOS provider transport '{selected}'. "
        "Expected one of: zenoh, http, fake."
    )


__all__ = ["DEFAULT_PROVIDER_TRANSPORT", "create_provider_client"]
