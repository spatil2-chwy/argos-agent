from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from argos_src.employee_directory.service import EmployeeDirectoryService


REPO_ROOT = Path(__file__).resolve().parents[5]
MODULE_PATH = (
    REPO_ROOT / "argos_src/tools/unitree_go2/vision/resolve_employee_identity.py"
)


def _load_tool_module():
    module_name = "test_argos_resolve_employee_identity_tool_module"
    sys.modules.pop(module_name, None)
    sys.modules.pop("argos_src.tools.base", None)
    sys.modules.pop("pydantic", None)

    pydantic_mod = type(sys)("pydantic")

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _field(default=None, **_kwargs):
        return default

    pydantic_mod.BaseModel = _BaseModel
    pydantic_mod.Field = _field
    sys.modules["pydantic"] = pydantic_mod

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, _query: str, _params: tuple[Any, ...]) -> None:
        return None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


class _FakeConnection:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._rows)

    def close(self) -> None:
        return None


class _UnavailableDirectory:
    def resolve_identity(
        self,
        shared_first_name: str = "",
        shared_last_name: str = "",
        shared_name: str = "",
    ) -> dict[str, Any]:
        return {
            "success": False,
            "status": "directory_unavailable",
            "message": "Directory not ready yet.",
            "data": {
                "site_code": "BOS3",
                "candidate_count": 0,
                "candidates": [],
            },
        }


def _build_service(rows: list[tuple[Any, ...]]) -> EmployeeDirectoryService:
    service = EmployeeDirectoryService(
        site_code="BOS3",
        env_loader=lambda: None,
        connector_factory=lambda: _FakeConnection(rows),
    )
    service.load_directory()
    return service


def test_resolve_employee_identity_tool_returns_directory_unavailable_payload():
    module = _load_tool_module()
    tool = module.get_resolve_employee_identity_tool(_UnavailableDirectory())

    payload = json.loads(tool._run("Sakshee", "Patil", "Sakshee Patil"))

    assert payload["success"] is False
    assert payload["status"] == "directory_unavailable"
    assert payload["data"]["candidates"] == []


def test_resolve_employee_identity_tool_caps_candidates_at_three():
    module = _load_tool_module()
    service = _build_service(
        [
            ("Alex Kim", "Alex", "Kim", "Manager", "5 years"),
            ("Alex Kim", "Alex", "Kim", "Director", "8 years"),
            ("Alex Kim", "Alex", "Kim", "Engineer", "2 years"),
            ("Alex Kim", "Alex", "Kim", "Analyst", "1 year"),
        ]
    )
    tool = module.get_resolve_employee_identity_tool(service)

    payload = json.loads(tool._run("Alex", "Kim", "Alex Kim"))

    assert payload["success"] is True
    assert payload["status"] == "multiple_matches"
    assert payload["candidate_count"] == 3
    assert len(payload["data"]["candidates"]) == 3


def test_resolve_employee_identity_tool_redacts_internal_directory_fields():
    module = _load_tool_module()
    service = _build_service(
        [
            (
                "Sakshee Patil",
                "Sakshee",
                "Patil",
                "AI Technologist II",
                "2 years",
                "spatil2",
                "Artificial Intelligence",
                "Information Technology",
                "Analyst",
                "C05",
                "Dan Burns",
                "AI and Data Innovation",
                "Jeff Greenfield",
                "AI & Data",
            ),
        ]
    )
    tool = module.get_resolve_employee_identity_tool(service)

    payload = json.loads(tool._run("Sakshee", "Patil", "Sakshee Patil"))

    assert payload["success"] is True
    assert payload["status"] == "single_match"
    assert payload["data"]["candidates"] == [
        {
            "official_name": "Sakshee Patil",
            "employee_name": "Sakshee Patil",
            "username": "spatil2",
            "business_title": "AI Technologist II",
            "tenure": "2 years",
            "match_score": 100.0,
        }
    ]

    verified_profile = service.get_verified_profile(
        username="spatil2",
        official_name="Sakshee Patil",
    )
    assert verified_profile is not None
    assert verified_profile["job_family"] == "Artificial Intelligence"
    assert verified_profile["manager_name"] == "Dan Burns"
    assert verified_profile["cost_center"] == "AI and Data Innovation"


def test_resolve_employee_identity_tool_passes_separate_name_fields():
    module = _load_tool_module()
    captured: dict[str, Any] = {}

    class _SpyDirectory:
        def resolve_identity(
            self,
            shared_first_name: str = "",
            shared_last_name: str = "",
            shared_name: str = "",
        ) -> dict[str, Any]:
            captured["shared_first_name"] = shared_first_name
            captured["shared_last_name"] = shared_last_name
            captured["shared_name"] = shared_name
            return {
                "success": True,
                "status": "no_match",
                "message": "No match",
                "data": {
                    "site_code": "BOS3",
                    "candidate_count": 0,
                    "candidates": [],
                },
            }

    tool = module.get_resolve_employee_identity_tool(_SpyDirectory())

    payload = json.loads(tool._run("Sakshee", "Patil", "Sakshee Patil"))

    assert payload["success"] is True
    assert captured == {
        "shared_first_name": "Sakshee",
        "shared_last_name": "Patil",
        "shared_name": "Sakshee Patil",
    }
