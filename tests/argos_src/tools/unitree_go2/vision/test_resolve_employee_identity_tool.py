from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
MODULE_PATH = (
    REPO_ROOT / "argos_src/tools/unitree_go2/vision/resolve_employee_identity.py"
)


def _load_tool_module():
    module_name = "test_argos_resolve_employee_identity_tool_module"
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeIdentityMemory:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {
            "success": False,
            "status": "directory_unavailable",
            "message": "Directory not ready yet.",
            "data": {
                "site_code": "BOS3",
                "candidate_count": 0,
                "candidates": [],
            },
        }
        self.calls: list[dict[str, str]] = []

    def resolve_identity(
        self,
        *,
        shared_first_name: str = "",
        shared_last_name: str = "",
        shared_name: str = "",
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "shared_first_name": shared_first_name,
                "shared_last_name": shared_last_name,
                "shared_name": shared_name,
            }
        )
        return dict(self.response)


def test_resolve_employee_identity_tool_returns_identity_memory_payload():
    module = _load_tool_module()
    tool = module.get_resolve_employee_identity_tool(_FakeIdentityMemory())

    payload = json.loads(tool._run("Sakshee", "Patil", "Sakshee Patil"))

    assert payload["success"] is False
    assert payload["status"] == "directory_unavailable"
    assert payload["data"]["candidates"] == []


def test_resolve_employee_identity_tool_passes_separate_name_fields():
    module = _load_tool_module()
    memory = _FakeIdentityMemory(
        {
            "success": True,
            "status": "single_match",
            "message": "Resolved",
            "data": {
                "site_code": "BOS3",
                "candidate_count": 1,
                "candidates": [{"official_name": "Sakshee Patil"}],
            },
        }
    )
    tool = module.get_resolve_employee_identity_tool(memory)

    payload = json.loads(tool._run("Sakshee", "Patil", "Sakshee Patil"))

    assert payload["success"] is True
    assert memory.calls == [
        {
            "shared_first_name": "Sakshee",
            "shared_last_name": "Patil",
            "shared_name": "Sakshee Patil",
        }
    ]
