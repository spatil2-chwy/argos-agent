from __future__ import annotations

import inspect
import sys
import types

import pytest

from argos_src.agent.preference_types import PreferenceSegment, PreferenceSegmentTurn
from argos_src.memory_provider.tailwag import TailwagMemoryProvider


class _TailwagInput:
    allowed_fields: frozenset[str] = frozenset()

    def __init__(self, **kwargs):
        unknown_fields = set(kwargs) - self.allowed_fields
        if unknown_fields:
            unexpected = ", ".join(sorted(unknown_fields))
            raise TypeError(f"unexpected Tailwag input fields: {unexpected}")
        self.face_embedding = None
        self.audio_embedding = None
        for key, value in kwargs.items():
            setattr(self, key, value)


class _PersonInput(_TailwagInput):
    allowed_fields = frozenset(
        {
            "id",
            "display_name",
            "email",
            "consent_status",
            "face_embedding",
            "audio_embedding",
            "role",
            "source",
        }
    )


class _PlaceInput(_TailwagInput):
    allowed_fields = frozenset({"building_code", "room_id"})


class _EpisodeInput(_TailwagInput):
    allowed_fields = frozenset(
        {
            "id",
            "episode_type",
            "start_time",
            "end_time",
            "transcript",
            "retention_class",
            "place",
            "participants",
        }
    )


@pytest.fixture(autouse=True)
def fake_tailwag_memory_module(monkeypatch):
    module = types.ModuleType("tailwag_memory")
    module.PersonInput = _PersonInput
    module.PlaceInput = _PlaceInput
    module.EpisodeInput = _EpisodeInput
    monkeypatch.setitem(sys.modules, "tailwag_memory", module)


class FakeTailwagClient:
    def __init__(self, *, context: str = "") -> None:
        self.context = context
        self.context_person_id = ""
        self.context_kwargs = {}
        self.recorded = []
        self.upserts = []
        self.archived = []
        self.rekeys = []
        self.raise_context = False
        self.raise_record = False
        self.raise_archive = False
        self.semantic_search_kwargs = None

    def person_context(self, person_id: str, **kwargs):
        self.context_person_id = person_id
        self.context_kwargs = kwargs
        if self.raise_context:
            raise RuntimeError("tailwag context unavailable")
        return self.context

    def record_episode(self, episode, *, extract_memory: bool = True):
        if self.raise_record:
            raise RuntimeError("tailwag record unavailable")
        self.recorded.append((episode, extract_memory))
        return episode.id

    def rekey_person_by_email(self, email: str, new_person_id: str) -> bool:
        self.rekeys.append((email, new_person_id))
        return True

    def upsert_person(self, person):
        self.upserts.append(person)
        return person.id

    def archive_person(self, person_id: str) -> bool:
        if self.raise_archive:
            raise RuntimeError("tailwag archive unavailable")
        self.archived.append(person_id)
        return True

    def search_semantic_memory(self, **kwargs):
        self.semantic_search_kwargs = kwargs
        return {
            "episodes": [
                {
                    "episode_id": "episode-1",
                    "transcript": "Robot demos are scheduled.",
                    "score": 0.7,
                    "start_time": "2026-06-01T10:00:00Z",
                    "end_time": "2026-06-01T10:05:00Z",
                    "building_code": "BOS",
                    "room_id": "lab",
                }
            ],
            "memory_items": [
                {
                    "memory_id": "memory-1",
                    "person_id": "person-1",
                    "kind": "preference",
                    "key": "drink",
                    "summary": "Likes tea.",
                    "source": "extracted",
                    "source_ref": "episode-1",
                    "status": "active",
                    "observed_at": "",
                    "created_at": "",
                    "updated_at": "",
                    "due_at": "",
                    "expires_at": "",
                    "metadata": {},
                    "score": 0.9,
                }
            ],
        }


def _provider_for(client: FakeTailwagClient, **kwargs) -> TailwagMemoryProvider:
    return TailwagMemoryProvider(client_factory=lambda: client, **kwargs)


def _segment(
    segment_id: str,
    person_id: str,
    user_text: str,
    assistant_text: str = "",
) -> PreferenceSegment:
    return PreferenceSegment(
        segment_id=segment_id,
        person_id=person_id,
        turns=(
            PreferenceSegmentTurn(
                turn_id=f"{segment_id}-turn",
                person_id=person_id,
                user_text=user_text,
                assistant_text=assistant_text,
            ),
        ),
    )


def test_person_context_parses_tailwag_prompt_projection_and_preferred_language():
    client = FakeTailwagClient(
        context="""[PERSON MEMORY]
Preferences:
- Likes concise robot updates.
- preferred language: Spanish
- Likes concise robot updates.

Potential Follow-Ups:
- Ask how the demo went.
"""
    )
    provider = _provider_for(client)

    context = provider.person_context("person-1", current_text="demo follow-up")

    assert client.context_person_id == "person-1"
    assert client.context_kwargs == {"current_text": "demo follow-up"}
    assert context.profile_lines == (
        "Likes concise robot updates.",
        "preferred language: Spanish",
    )
    assert context.followup_lines == ("Ask how the demo went.",)
    assert context.preferred_language == "Spanish"


def test_person_context_uses_fallbacks_when_tailwag_is_unavailable():
    client = FakeTailwagClient()
    client.raise_context = True
    provider = _provider_for(client)

    context = provider.person_context(
        "person-1",
        fallback_profile_lines=("fallback profile",),
        fallback_followup_lines=("fallback follow-up",),
    )

    assert context.profile_lines == ("fallback profile",)
    assert context.followup_lines == ("fallback follow-up",)
    assert context.preferred_language == "English"


def test_extract_and_store_segment_records_tailwag_episode_without_biometrics():
    client = FakeTailwagClient()
    provider = _provider_for(
        client,
        site_code="BOS3",
        place_room_id="lobby",
        retention_class="standard",
        extract_live_turn_memory=False,
    )
    segment = _segment(
        "seg-1",
        "person-1",
        "I like robot demos.",
        "I'll remember that.",
    )

    provider.extract_and_store_segment(segment)

    assert len(client.recorded) == 1
    episode, extract_memory = client.recorded[0]
    assert extract_memory is False
    assert episode.id.startswith("argos:conversation:")
    assert episode.episode_type == "conversation"
    assert episode.retention_class == "standard"
    assert episode.place.building_code == "BOS3"
    assert episode.place.room_id == "lobby"
    assert episode.participants[0].id == "person-1"
    assert episode.participants[0].role == "speaker"
    assert episode.participants[0].source == "live_chat"
    assert episode.participants[0].face_embedding is None
    assert episode.participants[0].audio_embedding is None
    assert not hasattr(episode, "summary")
    assert "User: I like robot demos." in episode.transcript
    assert "Assistant: I'll remember that." in episode.transcript
    assert "seg-1-turn" not in episode.transcript


def test_segments_append_to_one_episode_until_idle_timeout():
    client = FakeTailwagClient()
    provider = _provider_for(client, site_code="BOS3")

    provider.extract_and_store_segment(
        _segment("seg-1", "person-1", "First turn.", "First reply."),
        reason="speaker_handoff",
    )
    provider.extract_and_store_segment(
        _segment("seg-2", "person-2", "Second turn.", "Second reply."),
        reason="idle_timeout",
    )
    provider.extract_and_store_segment(
        _segment("seg-3", "person-1", "New conversation.", "New reply."),
    )

    first_episode = client.recorded[0][0]
    second_episode = client.recorded[1][0]
    third_episode = client.recorded[2][0]
    assert first_episode.id == second_episode.id
    assert third_episode.id != second_episode.id
    assert [participant.id for participant in second_episode.participants] == [
        "person-1",
        "person-2",
    ]
    assert "First turn." in second_episode.transcript
    assert "Second turn." in second_episode.transcript
    assert "New conversation." not in second_episode.transcript
    assert "New conversation." in third_episode.transcript


def test_record_encounter_rekeys_by_email_and_upserts_without_embeddings():
    client = FakeTailwagClient()
    provider = _provider_for(client)

    assert provider.record_encounter(
        person_id="person-asha",
        name="Asha",
        metadata={"email": " Asha.Example@Example.COM "},
    )

    assert client.rekeys == [("asha.example@example.com", "person-asha")]
    assert len(client.upserts) == 1
    person = client.upserts[0]
    assert person.id == "person-asha"
    assert person.display_name == "Asha"
    assert person.email == "asha.example@example.com"
    assert person.role == "participant"
    assert person.source == "argos"
    assert person.face_embedding is None
    assert person.audio_embedding is None


def test_tailwag_provider_search_semantic_memory_uses_episodes_and_memory_items():
    client = FakeTailwagClient()
    provider = _provider_for(client)

    results = provider.search_semantic_memory(
        text="demos",
        person_id="person-1",
        building_code="BOS",
        limit=50,
    )

    assert client.semantic_search_kwargs == {
        "text": "demos",
        "person_id": "person-1",
        "building_code": "BOS",
        "limit": 50,
    }
    assert results["episodes"] == [
        {
            "episode_id": "episode-1",
            "transcript": "Robot demos are scheduled.",
            "score": 0.7,
            "start_time": "2026-06-01T10:00:00Z",
            "end_time": "2026-06-01T10:05:00Z",
            "building_code": "BOS",
            "room_id": "lab",
        }
    ]
    assert results["memory_items"] == [
        {
            "memory_id": "memory-1",
            "person_id": "person-1",
            "kind": "preference",
            "key": "drink",
            "summary": "Likes tea.",
            "source": "extracted",
            "source_ref": "episode-1",
            "status": "active",
            "observed_at": "",
            "created_at": "",
            "updated_at": "",
            "due_at": "",
            "expires_at": "",
            "metadata": {},
            "score": 0.9,
        }
    ]


def test_archive_person_delegates_to_tailwag_and_falls_back_on_errors():
    client = FakeTailwagClient()
    provider = _provider_for(client)

    assert provider.archive_person("person-1") is True
    assert client.archived == ["person-1"]

    client.raise_archive = True
    assert provider.archive_person("person-2") is False
    assert provider.archive_person("") is False


def test_health_and_record_errors_return_no_memory_fallbacks():
    def failing_client_factory():
        raise RuntimeError("missing tailwag")

    assert (
        TailwagMemoryProvider(client_factory=failing_client_factory).health() is False
    )

    client = FakeTailwagClient()
    client.raise_record = True
    provider = _provider_for(client)

    provider.extract_and_store_segment(
        _segment("seg-1", "person-1", "This should not crash."),
        reason="idle_timeout",
    )

    client.raise_record = False
    provider.extract_and_store_segment(_segment("seg-2", "person-1", "Next turn."))

    assert len(client.recorded) == 1
    assert "This should not crash." not in client.recorded[0][0].transcript
    assert "Next turn." in client.recorded[0][0].transcript


def test_site_blocks_are_deferred_until_tailwag_exposes_contract():
    provider = _provider_for(FakeTailwagClient())

    assert provider.site_blocks("BOS3", current_person_id="person-1") == ()


def test_factory_uses_tailwag_provider_without_constructing_sqlite_memory():
    import argos_src.agent.factory as factory

    source = inspect.getsource(factory.create_agent)

    assert "TailwagMemoryProvider" in source
    assert "MemoryStore(" not in source
    assert "MemoryContextCompiler(" not in source
