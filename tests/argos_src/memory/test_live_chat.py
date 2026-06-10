from __future__ import annotations


def test_live_chat_memory_writer_maps_operations_to_memory_items(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.live_chat import write_live_chat_memory_items

    store = MemoryStore(tmp_path / "memory.sqlite3")
    operations = {
        "ops": [
            {
                "op": "create",
                "kind": "preference",
                "key": "preferred_language",
                "summary": "preferred language: Spanish",
                "value": {"field": "preferred_language", "value": "Spanish"},
            },
            {
                "op": "create",
                "kind": "fact",
                "key": "birthday",
                "summary": "birthday: April 23",
                "value": {"field": "birthday", "value": "April 23"},
            },
            {
                "op": "create",
                "kind": "preference",
                "key": "likes_blue_waters_at_the_cape",
                "summary": "likes: blue waters at the Cape",
            },
            {
                "op": "create",
                "kind": "pet",
                "key": "pet_luna",
                "summary": "pet: Luna (dog): was sick",
                "value": {"name": "Luna", "kind": "dog", "notes": "was sick"},
            },
            {
                "op": "create",
                "kind": "boundary",
                "key": "avoid_surprise_loud_noises",
                "summary": "boundary: avoid surprise loud noises",
            },
            {
                "op": "create",
                "kind": "note",
                "key": "robot_social_memory_work",
                "summary": "User is working on robot social memory.",
            },
            {
                "op": "create",
                "kind": "note",
                "key": "parents_cape_cod_weekend",
                "summary": (
                    "User's parents are visiting the U.S. for the first time and "
                    "they plan to go to Cape Cod this weekend."
                ),
                "expires_at": "2099-06-01T00:00:00+00:00",
            },
            {
                "op": "create",
                "kind": "note",
                "key": "team_guess",
                "summary": "team: Robotics",
            },
            {
                "op": "create",
                "kind": "followup",
                "key": "luna_recovery",
                "summary": "Luna is recovering after surgery.",
                "due_at": "2099-05-16T00:00:00+00:00",
                "expires_at": "2099-05-20T00:00:00+00:00",
            }
        ],
    }

    write_live_chat_memory_items(
        store,
        person_id="person-1",
        operations=operations,
        source_ref="segment-1",
    )

    items = store.list_active_items(scope_type="person", scope_id="person-1", limit=10)
    summaries = {item.summary for item in items}

    assert "preferred language: Spanish" in summaries
    assert "birthday: April 23" in summaries
    assert "likes: blue waters at the Cape" in summaries
    assert "pet: Luna (dog): was sick" in summaries
    assert "boundary: avoid surprise loud noises" in summaries
    assert "User is working on robot social memory." in summaries
    assert any("Cape Cod this weekend" in summary for summary in summaries)
    assert "Luna is recovering after surgery." in summaries
    assert "team: Robotics" not in summaries
    by_summary = {item.summary: item.kind for item in items}
    assert by_summary["birthday: April 23"] == "fact"
    assert by_summary["pet: Luna (dog): was sick"] == "pet"


def test_live_chat_memory_writer_dedupes_repeated_operations(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.live_chat import write_live_chat_memory_items

    store = MemoryStore(tmp_path / "memory.sqlite3")
    operations = {
        "ops": [
            {
                "op": "create",
                "kind": "preference",
                "key": "likes_playful_greetings",
                "summary": "likes: playful greetings",
            },
        ],
    }

    write_live_chat_memory_items(
        store,
        person_id="person-1",
        operations=operations,
        source_ref="segment-1",
    )
    write_live_chat_memory_items(
        store,
        person_id="person-1",
        operations=operations,
        source_ref="segment-1",
    )

    items = store.list_active_items(scope_type="person", scope_id="person-1")

    assert len(items) == 1
    assert items[0].summary == "likes: playful greetings"


def test_live_chat_memory_writer_upserts_notes_by_stable_key(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.live_chat import write_live_chat_memory_items

    store = MemoryStore(tmp_path / "memory.sqlite3")
    first = {
        "ops": [
            {
                "op": "create",
                "kind": "note",
                "key": "robot_social_memory_work",
                "summary": "User works on robot social memory.",
            },
        ],
    }
    write_live_chat_memory_items(
        store,
        person_id="person-1",
        operations=first,
        source_ref="segment-1",
    )
    existing = store.list_active_items(scope_type="person", scope_id="person-1")
    second = {
        "ops": [
            {
                "op": "update",
                "memory_id": existing[0].memory_id,
                "summary": (
                    "User works on robot social memory and is refining "
                    "memory extraction."
                ),
            },
        ],
    }
    write_live_chat_memory_items(
        store,
        person_id="person-1",
        operations=second,
        source_ref="segment-2",
    )

    items = store.list_active_items(scope_type="person", scope_id="person-1")

    assert len(items) == 1
    assert items[0].kind == "note"
    assert items[0].key == "robot_social_memory_work"
    assert items[0].summary == (
        "User works on robot social memory and is refining memory extraction."
    )


def test_live_chat_memory_writer_upserts_pets_by_name(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.live_chat import write_live_chat_memory_items

    store = MemoryStore(tmp_path / "memory.sqlite3")
    first = {
        "ops": [
            {
                "op": "create",
                "kind": "pet",
                "key": "pet_mochi",
                "summary": "pet: Mochi (dog)",
                "value": {"name": "Mochi", "kind": "dog"},
            },
        ],
    }
    write_live_chat_memory_items(
        store,
        person_id="person-1",
        operations=first,
        source_ref="segment-1",
    )
    existing = store.list_active_items(scope_type="person", scope_id="person-1")
    second = {
        "ops": [
            {
                "op": "update",
                "memory_id": existing[0].memory_id,
                "summary": "pet: Mochi (dog): puppy",
                "value": {"name": "Mochi", "kind": "dog", "notes": "puppy"},
            },
        ],
    }
    write_live_chat_memory_items(
        store,
        person_id="person-1",
        operations=second,
        source_ref="segment-2",
    )

    items = store.list_active_items(scope_type="person", scope_id="person-1")

    assert len(items) == 1
    assert items[0].key == "pet_mochi"
    assert items[0].summary == "pet: Mochi (dog): puppy"


def test_live_chat_memory_writer_archives_by_memory_id(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.live_chat import write_live_chat_memory_items

    store = MemoryStore(tmp_path / "memory.sqlite3")
    memory_id = store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="followup",
        key="luna_recovery",
        summary="Luna is recovering from surgery.",
        source="live_chat",
    )

    write_live_chat_memory_items(
        store,
        person_id="person-1",
        operations={"ops": [{"op": "archive", "memory_id": memory_id}]},
        source_ref="segment-2",
    )

    assert store.list_active_items(scope_type="person", scope_id="person-1") == []
    assert store.get_item(memory_id).status == "archived"


def test_candidate_memory_payload_is_capped_and_relevant(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.live_chat import _candidate_memory_payload

    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="preference",
        key="preferred_language",
        summary="preferred language: Spanish",
        source="live_chat",
    )
    store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="pet",
        key="pet_luna",
        summary="pet: Luna (dog): recovering from surgery",
        source="live_chat",
        metadata={"value": {"name": "Luna", "kind": "dog"}},
    )
    store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="note",
        key="unrelated_trip",
        summary="Visited Seattle last winter.",
        source="live_chat",
    )

    payload = _candidate_memory_payload(
        store,
        "person-1",
        "Luna is doing much better after surgery.",
        limit=2,
    )

    assert [item["key"] for item in payload] == ["preferred_language", "pet_luna"]
    assert payload[1]["value"] == {"name": "Luna", "kind": "dog"}


def test_candidate_memory_payload_pins_active_followups(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.live_chat import _candidate_memory_payload

    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="followup",
        key="luna_recovery",
        summary="Luna is recovering from surgery.",
        source="live_chat",
    )

    payload = _candidate_memory_payload(
        store,
        "person-1",
        "She's all better now.",
        limit=2,
    )

    assert [item["key"] for item in payload] == ["luna_recovery"]


def test_preference_extractor_writes_memory_store(tmp_path, monkeypatch):
    from argos_src.agent.preference_types import (
        PreferenceSegment,
        PreferenceSegmentTurn,
    )
    from argos_src.memory import MemoryStore
    from argos_src.memory import live_chat

    person_id = "person-test"
    memory_store = MemoryStore(tmp_path / "memory.sqlite3")

    extractor = object.__new__(live_chat.PreferenceExtractor)
    extractor.memory_store = memory_store
    extractor._perf_now = lambda: 0.0
    extractor._estimate_text_generation_cost = lambda *_args, **_kwargs: {}

    class _Latency:
        def timing(self, *_args, **_kwargs):
            return None

        def emit(self, *_args, **_kwargs):
            return None

    extractor.latency = _Latency()
    extractor.structured_llm = object()

    monkeypatch.setattr(
        live_chat,
        "_invoke_structured_llm",
        lambda *_args, **_kwargs: {
            "parsed": {
                "update": True,
                "ops": [
                    {
                        "op": "create",
                        "kind": "preference",
                        "key": "preferred_language",
                        "summary": "preferred language: Spanish",
                        "value": {"field": "preferred_language", "value": "Spanish"},
                    },
                    {
                        "op": "create",
                        "kind": "preference",
                        "key": "likes_robot_demos",
                        "summary": "likes: robot demos",
                    },
                ],
            },
            "raw": object(),
            "parsing_error": None,
        },
    )
    segment = PreferenceSegment(
        segment_id="segment-1",
        person_id=person_id,
        turns=(
            PreferenceSegmentTurn(
                turn_id="turn-1",
                person_id=person_id,
                user_text="I like robot demos and prefer Spanish.",
                assistant_text="Got it.",
            ),
        ),
    )

    extractor.extract_and_store_segment(segment)

    items = memory_store.list_active_items(scope_type="person", scope_id=person_id)

    assert {item.summary for item in items} == {
        "preferred language: Spanish",
        "likes: robot demos",
    }
