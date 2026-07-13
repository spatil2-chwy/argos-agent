"""Runtime facade for optional interaction display resources."""

from __future__ import annotations

import base64
from io import BytesIO
import logging
import threading
import time
from typing import Any
from uuid import uuid4

from argos_src.provider_api.client import ProviderClient
from argos_src.provider_api.errors import ProviderError, ProviderTimeout, is_provider_error
from argos_src.provider_api.wire import (
    OP_DISPLAY_AWAIT_RESPONSE,
    OP_DISPLAY_COMMAND,
    OP_DISPLAY_HEALTH,
    OP_DISPLAY_IMAGE,
    OP_DISPLAY_STATE,
)


logger = logging.getLogger(__name__)

DEFAULT_SUBTITLE_DURATION_MS = 5000
DEFAULT_REVIEW_TIMEOUT_SEC = 30.0


class DisplayRuntime:
    """Small high-level API for the robot's optional local interaction screen."""

    def __init__(
        self,
        *,
        client: ProviderClient | None = None,
        resource_id: str = "",
        enabled: bool = True,
    ) -> None:
        self._client = client
        self._resource_id = str(resource_id or "").strip()
        self._enabled = bool(enabled and client is not None and self._resource_id)
        self._last_face = ""
        self._last_subtitle = ""
        self._modal_lock = threading.RLock()

    @property
    def is_configured(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled or self._client is None:
            return
        starter = getattr(self._client, "start", None)
        if callable(starter):
            starter()

    def shutdown(self) -> None:
        if self._client is None:
            return
        shutdown = getattr(self._client, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                logger.exception("Failed to stop display client cleanly")

    def health(self) -> dict[str, Any]:
        if not self._enabled:
            return {"ok": False, "configured": False}
        return self._request(OP_DISPLAY_HEALTH, {}, timeout_ms=1000)

    def state(self) -> dict[str, Any]:
        if not self._enabled:
            return {"ok": False, "configured": False}
        return self._request(OP_DISPLAY_STATE, {}, timeout_ms=1000)

    def set_face(self, face: str) -> bool:
        rendered = str(face or "").strip()
        if not rendered:
            return False
        if rendered == self._last_face:
            return True
        ok = self.command({"type": "face", "face": rendered})
        if ok:
            self._last_face = rendered
        return ok

    def show_subtitle(
        self,
        text: str,
        *,
        duration_ms: int = DEFAULT_SUBTITLE_DURATION_MS,
    ) -> bool:
        rendered = str(text or "").strip()
        if not rendered:
            return self.clear()
        if rendered == self._last_subtitle:
            return True
        ok = self.command(
            {
                "type": "subtitle",
                "text": rendered,
                "durationMs": int(duration_ms),
            }
        )
        if ok:
            self._last_subtitle = rendered
        return ok

    def show_message(self, text: str) -> bool:
        rendered = str(text or "").strip()
        if not rendered:
            return False
        return self.command({"type": "message", "text": rendered})

    def show_countdown(self, seconds: int) -> bool:
        return self.command({"type": "countdown", "seconds": int(seconds)})

    def show_image_message_preview(
        self,
        *,
        image_url: str,
        title: str = "Captured Image",
        message: str = "",
        hold_sec: float = 5.0,
        timeout_ms: int = 2000,
        clear_after: bool = True,
    ) -> bool:
        rendered_image_url = str(image_url or "").strip()
        rendered_message = str(message or "").strip()
        if not rendered_image_url or not rendered_message:
            return False
        with self._modal_lock:
            sent = self.command(
                {
                    "type": "image_message_preview",
                    "imageUrl": rendered_image_url,
                    "title": str(title or "").strip() or "Captured Image",
                    "message": rendered_message,
                },
                timeout_ms=timeout_ms,
                critical=True,
            )
            if not sent:
                return False
            if hold_sec > 0:
                time.sleep(float(hold_sec))
            if clear_after:
                self.clear()
            return True

    def show_live_image(
        self,
        *,
        data_url: str = "",
        image_url: str = "",
        title: str = "Camera",
        ttl_ms: int = 1000,
        timeout_ms: int = 250,
    ) -> bool:
        payload: dict[str, Any] = {
            "title": str(title or "").strip() or "Camera",
            "ttlMs": int(ttl_ms),
        }
        rendered_data_url = str(data_url or "").strip()
        rendered_image_url = str(image_url or "").strip()
        if rendered_data_url:
            payload["dataUrl"] = rendered_data_url
        elif rendered_image_url:
            payload["imageUrl"] = rendered_image_url
        else:
            return False
        return self._image(payload, timeout_ms=timeout_ms)

    def clear_live_image(self, *, timeout_ms: int = 250) -> bool:
        return self._image({"type": "clear"}, timeout_ms=timeout_ms)

    def clear(self) -> bool:
        self._last_subtitle = ""
        return self.command({"type": "clear"})

    def reset(self) -> bool:
        self._last_face = ""
        self._last_subtitle = ""
        return self.command({"type": "reset"})

    def _forget_transient_view_cache(self) -> None:
        self._last_face = ""
        self._last_subtitle = ""

    def show_idle(self) -> None:
        self._forget_transient_view_cache()
        self.set_face("happy")

    def show_alert(self) -> None:
        self._forget_transient_view_cache()
        self.set_face("think")

    def show_recording(self) -> None:
        self._forget_transient_view_cache()
        self.set_face("think")
        self.show_subtitle("Recording...")

    def show_thinking(self) -> None:
        self._forget_transient_view_cache()
        self.show_message("Thinking...")

    def show_speaking(self) -> None:
        self._forget_transient_view_cache()
        self.set_face("excited")

    def command(
        self,
        payload: dict[str, Any],
        *,
        timeout_ms: int = 1000,
        critical: bool = False,
    ) -> bool:
        if not self._enabled:
            return False
        with self._modal_lock:
            try:
                self._request(
                    OP_DISPLAY_COMMAND,
                    dict(payload or {}),
                    timeout_ms=timeout_ms,
                )
                return True
            except Exception as exc:
                if critical or not is_provider_error(exc):
                    logger.warning("Display command failed: %s", exc)
                else:
                    logger.debug("Display command failed: %s", exc)
                return False

    def _image(self, payload: dict[str, Any], *, timeout_ms: int = 250) -> bool:
        if not self._enabled:
            return False
        with self._modal_lock:
            try:
                self._request(
                    OP_DISPLAY_IMAGE,
                    dict(payload or {}),
                    timeout_ms=timeout_ms,
                )
                return True
            except Exception as exc:
                if not is_provider_error(exc):
                    logger.warning("Display image update failed: %s", exc)
                else:
                    logger.debug("Display image update failed: %s", exc)
                return False

    def review_face_capture(
        self,
        *,
        image_url: str,
        request_id: str | None = None,
        title: str = "Face Capture Preview",
        accept_label: str = "Accept",
        reject_label: str = "Reject",
        timeout_sec: float = DEFAULT_REVIEW_TIMEOUT_SEC,
    ) -> dict[str, Any]:
        if not self._enabled:
            return {"available": False, "accepted": False, "status": "unconfigured"}
        with self._modal_lock:
            rendered_request_id = str(
                request_id or f"face-capture-{uuid4().hex[:12]}"
            ).strip()
            sent = self.command(
                {
                    "type": "face_capture_preview",
                    "requestId": rendered_request_id,
                    "imageUrl": str(image_url or "").strip(),
                    "title": title,
                    "acceptLabel": accept_label,
                    "rejectLabel": reject_label,
                },
                timeout_ms=2000,
                critical=True,
            )
            if not sent:
                return {
                    "available": False,
                    "accepted": False,
                    "status": "display_unavailable",
                    "requestId": rendered_request_id,
                }
            try:
                response = self._request(
                    OP_DISPLAY_AWAIT_RESPONSE,
                    {"requestId": rendered_request_id},
                    timeout_ms=max(1, int(float(timeout_sec) * 1000)),
                )
            except ProviderTimeout:
                self.clear()
                return {
                    "available": True,
                    "accepted": False,
                    "status": "review_timeout",
                    "requestId": rendered_request_id,
                }
            except ProviderError as exc:
                logger.warning("Display review failed: %s", exc)
                self.clear()
                return {
                    "available": False,
                    "accepted": False,
                    "status": "display_unavailable",
                    "requestId": rendered_request_id,
                }
            accepted = bool(response.get("accepted", False))
            action = str(response.get("action") or ("accept" if accepted else "reject"))
            return {
                "available": True,
                "accepted": accepted,
                "action": action,
                "status": "accepted" if accepted else "rejected",
                "requestId": rendered_request_id,
                "response": response,
            }

    def review_text_prompt(
        self,
        *,
        title: str,
        message: str,
        request_id: str | None = None,
        accept_label: str = "Accept",
        reject_label: str = "Reject",
        timeout_sec: float = DEFAULT_REVIEW_TIMEOUT_SEC,
    ) -> dict[str, Any]:
        """Show a blocking accept/reject prompt using the existing preview UI."""
        image_url = _text_prompt_data_url(title=title, message=message)
        return self.review_face_capture(
            image_url=image_url,
            request_id=request_id or f"text-prompt-{uuid4().hex[:12]}",
            title=title,
            accept_label=accept_label,
            reject_label=reject_label,
            timeout_sec=timeout_sec,
        )

    def _request(
        self,
        operation: str,
        args: dict[str, Any],
        *,
        timeout_ms: int,
    ) -> dict[str, Any]:
        if self._client is None:
            raise ProviderError("Display client is not configured.")
        return self._client.request(
            resource_id=self._resource_id,
            operation=operation,
            args=args,
            timeout_ms=timeout_ms,
        )


def _text_prompt_data_url(*, title: str, message: str) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - exercised only in stripped envs.
        raise RuntimeError("Pillow is required to render display text prompts.") from exc

    width, height = 1280, 720
    margin = 72
    image = Image.new("RGB", (width, height), color=(14, 14, 18))
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    y = margin
    rendered_title = str(title or "Confirm").strip() or "Confirm"
    draw.text((margin, y), rendered_title, fill=(255, 255, 255), font=title_font)
    y += 64

    rendered_message = str(message or "").strip()
    lines: list[str] = []
    for paragraph in rendered_message.splitlines() or [""]:
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > 48 and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)

    for line in lines[:12]:
        draw.text((margin, y), line, fill=(230, 230, 235), font=body_font)
        y += 48

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
