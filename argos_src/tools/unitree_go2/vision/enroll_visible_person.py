"""Tool to validate and enroll one visible person from the live camera."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel, Field

from argos_src.tools.base import BaseTool
from argos_src.tools.common.tool_response import tool_response_json
from argos_src.tools.unitree_go2.vision.capture_scene import DEFAULT_CAMERA_RESOURCE


class _EnrollVisiblePersonInput(BaseModel):
    official_name: str = Field(
        ...,
        description="Confirmed official full name to store for the visible person being enrolled.",
    )
    username: str | None = Field(
        default=None,
        description=(
            "Verified employee username from resolve_employee_identity, if one was returned. "
            "Do not guess this."
        ),
    )


class EnrollVisiblePersonTool(BaseTool):
    """Validate the live scene and enroll exactly one visible person when safe."""

    name: str = "enroll_visible_person"
    description: str = (
        "Register a new visible person using the live camera. "
        "Use this only after the person has confirmed their full name and that they are the only person in view and are ready to be remembered. "
        "When registration was validated through the employee directory, pass only the verified username returned by resolve_employee_identity; other employee fields are loaded locally. "
    )
    args_schema: Type[BaseModel] = _EnrollVisiblePersonInput
    face_service: Any = Field(exclude=True)
    employee_directory_service: Any | None = Field(default=None, exclude=True)
    default_camera_resource: str = Field(default=DEFAULT_CAMERA_RESOURCE, exclude=True)
    display_runtime: Any | None = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        official_name: str,
        username: str | None = None,
    ) -> str:
        employee_profile = None
        lookup = getattr(self.employee_directory_service, "get_verified_profile", None)
        if callable(lookup):
            employee_profile = lookup(
                username=username or "",
                official_name=official_name,
            )
        enroll_kwargs = {
            "official_name": official_name,
            "username": username or "",
            "employee_profile": employee_profile,
            "camera_resource_id": self.default_camera_resource,
        }
        if self.display_runtime is not None:
            enroll_kwargs["display_runtime"] = self.display_runtime
        result = self.face_service.enroll_visible_person(**enroll_kwargs)
        success = bool(result.get("success", False))
        status = str(result.get("status", "") or ("completed" if success else "error"))
        message = str(result.get("message", "") or "")
        extras = {
            key: value
            for key, value in result.items()
            if key not in {"success", "status", "message"}
        }
        return tool_response_json(
            success=success,
            status=status,
            message=message,
            **extras,
        )


def get_enroll_visible_person_tool(
    face_service: Any,
    default_camera_resource: str = DEFAULT_CAMERA_RESOURCE,
    employee_directory_service: Any | None = None,
    display_runtime: Any | None = None,
) -> BaseTool:
    """Return the enrollment tool bound to the active face service."""
    return EnrollVisiblePersonTool(
        face_service=face_service,
        employee_directory_service=employee_directory_service,
        default_camera_resource=default_camera_resource,
        display_runtime=display_runtime,
    )
