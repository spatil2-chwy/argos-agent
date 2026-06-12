"""Provider/resource namespace helpers."""

from __future__ import annotations


MANIFEST_KEY_SEGMENT = "manifest"
RESOURCES_KEY_SEGMENT = "resources"
REQUEST_KEY_SEGMENT = "request"
RESPONSE_KEY_SEGMENT = "response"
EVENT_KEY_SEGMENT = "event"
STATE_KEY_SEGMENT = "state"


def normalize_provider_prefix(prefix: str) -> str:
    rendered = str(prefix or "").strip().strip("/")
    if not rendered:
        raise ValueError("provider key prefix must not be empty")
    return rendered


def provider_manifest_key(prefix: str) -> str:
    return f"{normalize_provider_prefix(prefix)}/{MANIFEST_KEY_SEGMENT}"


def provider_resource_prefix(prefix: str, resource_id: str) -> str:
    rendered_resource = str(resource_id or "").strip().strip("/")
    if not rendered_resource:
        raise ValueError("resource_id must not be empty")
    return (
        f"{normalize_provider_prefix(prefix)}/"
        f"{RESOURCES_KEY_SEGMENT}/{rendered_resource}"
    )


def provider_request_key(prefix: str, resource_id: str, request_id: str) -> str:
    return (
        f"{provider_resource_prefix(prefix, resource_id)}/"
        f"{REQUEST_KEY_SEGMENT}/{str(request_id or '').strip()}"
    )


def provider_response_key(prefix: str, resource_id: str, request_id: str) -> str:
    return (
        f"{provider_resource_prefix(prefix, resource_id)}/"
        f"{RESPONSE_KEY_SEGMENT}/{str(request_id or '').strip()}"
    )


def provider_event_key(prefix: str, resource_id: str, event_type: str = "*") -> str:
    rendered_event = str(event_type or "*").strip().strip("/") or "*"
    return (
        f"{provider_resource_prefix(prefix, resource_id)}/"
        f"{EVENT_KEY_SEGMENT}/{rendered_event}"
    )


def provider_state_key(prefix: str, resource_id: str, state_name: str) -> str:
    rendered_state = str(state_name or "").strip().strip("/")
    if not rendered_state:
        raise ValueError("state_name must not be empty")
    return (
        f"{provider_resource_prefix(prefix, resource_id)}/"
        f"{STATE_KEY_SEGMENT}/{rendered_state}"
    )


__all__ = [
    "EVENT_KEY_SEGMENT",
    "MANIFEST_KEY_SEGMENT",
    "REQUEST_KEY_SEGMENT",
    "RESOURCES_KEY_SEGMENT",
    "RESPONSE_KEY_SEGMENT",
    "STATE_KEY_SEGMENT",
    "normalize_provider_prefix",
    "provider_event_key",
    "provider_manifest_key",
    "provider_request_key",
    "provider_resource_prefix",
    "provider_response_key",
    "provider_state_key",
]
