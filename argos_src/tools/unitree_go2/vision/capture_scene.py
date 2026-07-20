"""Tool to capture a live camera snapshot for visual reasoning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import cv2
from langchain_core.tools import tool

from argos_src.media.image_encoding import preprocess_image
from argos_src.tools.common.tool_response import tool_response_json

DEFAULT_CAMERA_RESOURCE = "head_realsense"
MAX_CACHED_FRAME_AGE_SEC = 2.0


def get_capture_scene_tool(
    face_service: Any,
    default_camera_resource: str = DEFAULT_CAMERA_RESOURCE,
) -> Callable:
    """Build a multimodal tool that captures one current camera frame."""

    @tool(response_format="content_and_artifact")
    def capture_scene(camera_resource: str = default_camera_resource) -> tuple[str, dict]:
        """Capture a single current camera image and return it to the model for scene understanding.

        Use this when a user asks what is visible at a place (e.g. available snacks, items on a shelf,
        signs, room status). This only captures from the robot's current position. If needed,
        navigate first, then call this tool.
        """
        image, cached_resource, captured_at_unix = face_service.get_cached_latest_frame(
            camera_resource_id=camera_resource,
            max_age_sec=MAX_CACHED_FRAME_AGE_SEC,
        )
        if image is None:
            return (
                tool_response_json(
                    success=False,
                    status="error",
                    message=(
                        f"I don't have a recent cached camera frame from {camera_resource} right now. "
                        "The face camera loop may still be starting up."
                    ),
                    result_source="immediate",
                ),
                {"images": [], "audios": []},
            )

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encoded = preprocess_image(rgb_image)
        except Exception as exc:
            return (
                tool_response_json(
                    success=False,
                    status="error",
                    message=f"Captured a frame but failed to encode it: {exc}",
                    result_source="immediate",
                ),
                {"images": [], "audios": []},
            )

        height, width = image.shape[:2]
        captured_at = datetime.fromtimestamp(
            captured_at_unix,
            tz=timezone.utc,
        ).isoformat(timespec="seconds")
        content = tool_response_json(
            success=True,
            status="completed",
            message="Captured one scene image for visual reasoning.",
            result_source="immediate",
            data={
                "camera_resource": cached_resource,
                "captured_at": captured_at,
                "resolution": f"{width}x{height}",
                "instruction": "Analyze this image to answer the user.",
            },
        )
        return content, {"images": [encoded], "audios": []}

    return capture_scene
