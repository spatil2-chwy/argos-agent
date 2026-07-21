from __future__ import annotations

import pytest

from argos_src.agent.preference_types import PreferenceSegment, PreferenceSegmentTurn
from argos_src.identity_memory.tailwag_http import TailwagHttpIdentityMemoryClient


class _StrictProviderClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def request(self, *, resource_id, operation, args=None, timeout_ms=None):
        del timeout_ms
        self.calls.append((operation, {"resource_id": resource_id, **dict(args or {})}))
        if operation == "memory.biometrics_face_search":
            return {
                "candidates": [
                    {"person_id": "person-1", "display_name": "Alex", "score": 0.91}
                ],
                "recognized": True,
                "status": "accepted",
                "reason": "matched",
                "top_score": 0.91,
            }
        if operation == "memory.biometrics_voice_search":
            return {
                "candidates": [
                    {"person_id": "person-1", "display_name": "Alex", "score": 0.87}
                ],
                "recognized": True,
                "status": "accepted",
                "reason": "matched",
                "top_score": 0.87,
            }
        if operation in {
            "memory.biometrics_face_references",
            "memory.biometrics_voice_references",
        }:
            modality = "face" if "face" in operation else "voice"
            return {
                "saved": True,
                "status": "saved",
                "reason": "saved",
                "person_id": (args or {}).get("person_id", ""),
                "reference_id": f"{modality}-ref-1",
            }
        if operation in {
            "memory.biometrics_face_observations",
            "memory.biometrics_voice_observations",
        }:
            modality = "face" if "face" in operation else "voice"
            return {
                "accepted": True,
                "status": "accepted",
                "reason": "accepted",
                "person_id": (args or {}).get("person_id", ""),
                "modality": modality,
            }
        if operation in {"memory.identity_verified_profile", "memory.people_profile"}:
            return None
        if operation == "memory.person_context":
            return {
                "person_id": (args or {}).get("person_id", ""),
                "context_markdown": "[PERSON MEMORY]\nPreferences:\n- likes robot demos",
                "generated_at": "2026-07-10T00:00:00+00:00",
            }
        if operation == "memory.episodes_record":
            return {"episode_id": (args or {}).get("episode", {}).get("id", "")}
        raise AssertionError(f"Unexpected operation: {operation}")

    def shutdown(self):
        return None


def _segment() -> PreferenceSegment:
    return PreferenceSegment(
        segment_id="segment-1",
        person_id="person-1",
        turns=(
            PreferenceSegmentTurn(
                turn_id="turn-1",
                person_id="person-1",
                user_text="I like short demos.",
                assistant_text="Got it.",
            ),
        ),
    )


@pytest.mark.parametrize(
    ("robot_id", "robot_display_name", "message"),
    [
        ("", "Cody", "robot_id"),
        ("cody", "", "robot_display_name"),
        ("   ", "Cody", "robot_id"),
        ("cody", "   ", "robot_display_name"),
    ],
)
def test_tailwag_requires_nonblank_robot_identity(
    robot_id, robot_display_name, message
):
    with pytest.raises(ValueError, match=message):
        TailwagHttpIdentityMemoryClient(
            provider_client=_StrictProviderClient(),
            resource_id="memory",
            robot_id=robot_id,
            robot_display_name=robot_display_name,
        )


def test_tailwag_search_calls_match_http_provider_contract():
    provider = _StrictProviderClient()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
        robot_id="cody",
        robot_display_name="Cody",
        site_code="BOS3",
    )

    face = client.search_face(embedding=(0.1, 0.2), limit=3)
    voice = client.search_voice(embedding=(0.3, 0.4), limit=4)

    assert face.recognized is True
    assert face.top_score == 0.91
    assert voice.recognized is True
    assert voice.top_score == 0.87
    assert [name for name, _payload in provider.calls] == [
        "memory.biometrics_face_search",
        "memory.biometrics_voice_search",
    ]
    face_call = provider.calls[0][1]
    voice_call = provider.calls[1][1]
    assert face_call["resource_id"] == "memory"
    assert face_call["embedding"] == pytest.approx([0.1, 0.2])
    assert face_call["limit"] == 3
    assert face_call["site_code"] == "BOS3"
    assert voice_call["embedding"] == pytest.approx([0.3, 0.4])
    assert voice_call["limit"] == 4
    assert voice_call["site_code"] == "BOS3"


def test_tailwag_biometric_write_calls_match_http_provider_contract():
    provider = _StrictProviderClient()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
        robot_id="cody",
        robot_display_name="Cody",
    )

    face = client.enroll_face_reference(
        person_id="person-1",
        embedding=(0.1, 0.2),
        metadata={"source": "test"},
    )
    voice = client.enroll_voice_reference(
        person_id="person-1",
        embedding=(0.3, 0.4),
        metadata={"source": "test"},
    )
    face_update = client.observe_face_embedding(
        person_id="person-1",
        embedding=(0.5, 0.6),
        evidence={"owner_source": "audio_face_agree"},
    )
    voice_update = client.observe_voice_embedding(
        person_id="person-1",
        embedding=(0.7, 0.8),
        evidence={"owner_source": "audio_face_agree"},
    )

    assert face.saved is True
    assert voice.saved is True
    assert face_update.accepted is True
    assert voice_update.accepted is True
    assert [name for name, _payload in provider.calls] == [
        "memory.biometrics_face_references",
        "memory.biometrics_voice_references",
        "memory.biometrics_face_observations",
        "memory.biometrics_voice_observations",
    ]


def test_tailwag_optional_profiles_preserve_http_null_results():
    provider = _StrictProviderClient()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
        robot_id="cody",
        robot_display_name="Cody",
    )

    assert client.get_verified_profile(username="missing", official_name="Missing") is None
    assert client.person_profile("person-missing") is None
    assert [name for name, _payload in provider.calls] == [
        "memory.identity_verified_profile",
        "memory.people_profile",
    ]


@pytest.mark.parametrize(
    "directory_profile_lines",
    [
        ["Title: Robotics Software Engineer I Co-op", "Manager: Brian Waite"],
        "['Title: Robotics Software Engineer I Co-op', 'Manager: Brian Waite']",
    ],
)
def test_tailwag_person_profile_normalizes_directory_response_shapes(
    directory_profile_lines,
):
    class _ProfileProvider(_StrictProviderClient):
        def request(self, *, resource_id, operation, args=None, timeout_ms=None):
            if operation == "memory.people_profile":
                self.calls.append(
                    (operation, {"resource_id": resource_id, **dict(args or {})})
                )
                return {
                    "person_id": "person-1",
                    "display_name": "Alex",
                    "directory_profile_lines": directory_profile_lines,
                }
            return super().request(
                resource_id=resource_id,
                operation=operation,
                args=args,
                timeout_ms=timeout_ms,
            )

    provider = _ProfileProvider()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
        robot_id="cody",
        robot_display_name="Cody",
    )

    profile = client.person_profile("person-1")

    assert profile is not None
    assert profile.directory_profile_lines == (
        "Title: Robotics Software Engineer I Co-op",
        "Manager: Brian Waite",
    )
    assert [name for name, _payload in provider.calls] == ["memory.people_profile"]


def test_tailwag_person_context_uses_markdown_provider_contract():
    provider = _StrictProviderClient()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
        robot_id="cody",
        robot_display_name="Cody",
    )

    context = client.person_context("person-1", current_text="robot demo")

    assert context.context_markdown == "[PERSON MEMORY]\nPreferences:\n- likes robot demos"
    assert context.profile_lines == ()
    assert context.followup_lines == ()
    assert context.preferred_language == "English"
    assert [name for name, _payload in provider.calls] == ["memory.person_context"]
    call = provider.calls[0][1]
    assert call["resource_id"] == "memory"
    assert call["person_id"] == "person-1"
    assert call["current_text"] == "robot demo"
    assert call["semantic_scope"] is None
    assert call["memory_limit"] == 12


def test_tailwag_record_episode_defaults_to_no_memory_extraction():
    provider = _StrictProviderClient()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
        robot_id="cody",
        robot_display_name="Cody",
    )

    result = client.record_episode({"id": "episode-1", "turns": []})

    assert result == {"episode_id": "episode-1"}
    assert provider.calls == [
        (
            "memory.episodes_record",
            {
                "resource_id": "memory",
                "episode": {"id": "episode-1", "turns": []},
                "extract_memory": False,
            },
        )
    ]


def test_tailwag_record_episode_explicit_true_opts_in_to_memory_extraction():
    provider = _StrictProviderClient()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
        robot_id="cody",
        robot_display_name="Cody",
    )

    client.record_episode({"id": "episode-1", "turns": []}, extract_memory=True)

    assert provider.calls[0][1]["extract_memory"] is True


def test_tailwag_live_episode_ingestion_defaults_to_no_memory_extraction():
    provider = _StrictProviderClient()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
        robot_id="cody",
        robot_display_name="Cody",
    )

    client.extract_and_store_segment(_segment(), reason="turn_complete")

    assert provider.calls[0][0] == "memory.episodes_record"
    assert provider.calls[0][1]["extract_memory"] is False
    assert provider.calls[0][1]["episode"]["participants"] == [
        {"id": "person-1", "role": "speaker", "source": "live_chat"}
    ]
    assert provider.calls[0][1]["episode"]["robots"] == [
        {
            "id": "cody",
            "display_name": "Cody",
            "role": "host",
            "source": "argos",
        }
    ]
    assert len(provider.calls) == 1


def test_tailwag_live_episode_ingestion_explicit_true_opts_in_to_memory_extraction():
    provider = _StrictProviderClient()
    client = TailwagHttpIdentityMemoryClient(
        provider_client=provider,
        resource_id="memory",
        robot_id="cody",
        robot_display_name="Cody",
        extract_live_turn_memory=True,
    )

    client.extract_and_store_segment(_segment(), reason="turn_complete")

    assert provider.calls[0][0] == "memory.episodes_record"
    assert provider.calls[0][1]["extract_memory"] is True
    assert provider.calls[0][1]["episode"]["robots"][0]["id"] == "cody"
    assert len(provider.calls) == 1
