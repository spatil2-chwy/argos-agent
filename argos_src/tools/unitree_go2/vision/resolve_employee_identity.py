"""Tool to resolve a spoken employee name against the site-scoped directory."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel, Field

from argos_src.tools.base import BaseTool
from argos_src.tools.common.tool_response import tool_response_json


class _ResolveEmployeeIdentityInput(BaseModel):
    shared_first_name: str = Field(
        ...,
        description=(
            "The person's official first name for registration."
        ),
    )
    shared_last_name: str = Field(
        ...,
        description="The person's official last name for registration.",
    )
    shared_name: str = Field(
        default="",
        description=(
            "Optional full name for logging or backward compatibility. "
            "Prefer sending first and last name separately."
        ),
    )


class ResolveEmployeeIdentityTool(BaseTool):
    """Look up an employee at the robot's configured site."""

    name: str = "resolve_employee_identity"
    description: str = (
        "Look up a person's employee record at the robot's configured site during registration. "
        "Use this after an unrecognized person shares their official first and last name. "
        "Use returned verified employee metadata such as username, title, job family, manager, cost center, and tenure to help disambiguate between plausible matches before enrollment. "
        "Returns single_match, multiple_matches, needs_clarification, no_match, "
        "directory_unavailable, or invalid_input, with up to 3 candidates including the matched employee profile."
    )
    args_schema: Type[BaseModel] = _ResolveEmployeeIdentityInput
    employee_directory_service: Any = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        shared_first_name: str,
        shared_last_name: str,
        shared_name: str = "",
    ) -> str:
        result = self.employee_directory_service.resolve_identity(
            shared_first_name=shared_first_name,
            shared_last_name=shared_last_name,
            shared_name=shared_name,
        )
        success = bool(result.get("success", False))
        status = str(result.get("status", "") or ("completed" if success else "error"))
        message = str(result.get("message", "") or "")
        extras = {
            key: value
            for key, value in result.items()
            if key not in {"success", "status", "message", "data"}
        }
        return tool_response_json(
            success=success,
            status=status,
            message=message,
            data=result.get("data"),
            **extras,
        )


def get_resolve_employee_identity_tool(
    employee_directory_service: Any,
) -> BaseTool:
    """Return the employee lookup tool bound to the active directory service."""
    return ResolveEmployeeIdentityTool(
        employee_directory_service=employee_directory_service,
    )
