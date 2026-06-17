from __future__ import annotations

import json

from argos_src.tools.unitree_go2.vision.enroll_visible_person import (
    get_enroll_visible_person_tool,
)


def test_enroll_visible_person_tool_rehydrates_employee_profile_locally():
    captured = {}

    class _FaceService:
        def enroll_visible_person(self, **kwargs):
            captured.update(kwargs)
            return {
                "success": True,
                "status": "enrolled",
                "message": "enrolled",
                "person_id": "person-1",
            }

    class _Directory:
        def get_verified_profile(self, *, username: str = "", official_name: str = ""):
            assert username == "spatil2"
            assert official_name == "Sakshee Patil"
            return {
                "official_name": "Sakshee Patil",
                "employee_name": "Sakshee Patil",
                "username": "spatil2",
                "business_title": "AI Technologist II",
            }

    tool = get_enroll_visible_person_tool(
        _FaceService(),
        employee_directory_service=_Directory(),
        default_camera_resource="head_realsense",
    )

    payload = json.loads(
        tool._run(
            official_name="Sakshee Patil",
            username="spatil2",
        )
    )

    assert payload["success"] is True
    assert payload["person_id"] == "person-1"
    assert captured == {
        "official_name": "Sakshee Patil",
        "username": "spatil2",
        "employee_profile": {
            "official_name": "Sakshee Patil",
            "employee_name": "Sakshee Patil",
            "username": "spatil2",
            "business_title": "AI Technologist II",
        },
        "camera_resource_id": "head_realsense",
    }
