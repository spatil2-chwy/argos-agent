from __future__ import annotations

from datetime import datetime, timedelta, timezone


def test_recent_encounter_block_includes_age_and_relation():
    from argos_src.memory.models import MemoryItem
    from argos_src.memory.prompting import format_recent_encounters_block

    now = datetime(2026, 5, 14, 19, 0, tzinfo=timezone.utc)
    item = MemoryItem(
        memory_id="mem-1",
        scope_type="person",
        scope_id="person-alex",
        kind="encounter",
        key="latest",
        summary="Met Alex.",
        source="robot",
        status="active",
        observed_at=(now - timedelta(minutes=10)).isoformat(),
        metadata={"name": "Alex Kim", "relation_label": "same manager"},
    )

    block = format_recent_encounters_block([item], now=now)

    assert "You met Alex Kim 10 minutes ago" in block
    assert "same manager" in block


def test_person_profile_lines_prioritize_preferences_and_pets():
    from argos_src.memory.models import MemoryItem
    from argos_src.memory.prompting import format_person_profile_lines

    items = [
        MemoryItem(
            memory_id="mem-note",
            scope_type="person",
            scope_id="person-1",
            kind="note",
            key="work",
            summary="works on robot social interaction",
            source="live_chat",
            observed_at="2026-05-14T10:00:00+00:00",
        ),
        MemoryItem(
            memory_id="mem-like",
            scope_type="person",
            scope_id="person-1",
            kind="preference",
            key="likes_ocean",
            summary="likes: blue waters at the Cape",
            source="live_chat",
            observed_at="2026-05-14T09:00:00+00:00",
        ),
        MemoryItem(
            memory_id="mem-pet",
            scope_type="person",
            scope_id="person-1",
            kind="pet",
            key="pet_mochi",
            summary="pet: Mochi (dog, puppy)",
            source="live_chat",
            observed_at="2026-05-14T08:00:00+00:00",
            metadata={"field": "pets"},
        ),
    ]

    lines = format_person_profile_lines(items)

    assert lines == (
        "likes: blue waters at the Cape",
        "pet: Mochi (dog, puppy)",
        "works on robot social interaction",
    )


def test_person_profile_lines_include_all_structured_and_cap_notes():
    from argos_src.memory.models import MemoryItem
    from argos_src.memory.prompting import format_person_profile_lines

    items = [
        MemoryItem(
            memory_id="mem-bday",
            scope_type="person",
            scope_id="person-1",
            kind="fact",
            key="birthday",
            summary="birthday: April 23",
            source="live_chat",
            observed_at="2026-05-14T08:00:00+00:00",
        )
    ]
    for index in range(12):
        items.append(
            MemoryItem(
                memory_id=f"mem-note-{index}",
                scope_type="person",
                scope_id="person-1",
                kind="note",
                key=f"note_{index}",
                summary=f"note {index}",
                source="live_chat",
                observed_at=f"2026-05-14T08:{index:02d}:00+00:00",
            )
        )

    lines = format_person_profile_lines(items, note_limit=10)

    assert "birthday: April 23" in lines
    assert len([line for line in lines if line.startswith("note ")]) == 10


def test_person_profile_lines_cap_structured_items():
    from argos_src.memory.models import MemoryItem
    from argos_src.memory.prompting import format_person_profile_lines

    items = [
        MemoryItem(
            memory_id=f"mem-like-{index}",
            scope_type="person",
            scope_id="person-1",
            kind="preference",
            key=f"likes_{index}",
            summary=f"likes: item {index}",
            source="live_chat",
            observed_at=f"2026-05-14T08:{index:02d}:00+00:00",
        )
        for index in range(5)
    ]

    lines = format_person_profile_lines(items, structured_limit=3)

    assert len(lines) == 3


def test_person_context_exposes_preferred_language(tmp_path):
    from argos_src.memory import MemoryContextCompiler, MemoryStore

    memory_store = MemoryStore(tmp_path / "memory.sqlite3")
    memory_store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="preference",
        key="preferred_language",
        summary="preferred language: Spanish",
        source="live_chat",
        metadata={"field": "preferred_language", "value": "Spanish"},
    )

    context = MemoryContextCompiler(memory_store).person_context("person-1")

    assert context.preferred_language == "Spanish"
    assert "preferred language: Spanish" in context.profile_lines


def test_followup_lines_hide_items_until_due():
    from argos_src.memory.models import MemoryItem
    from argos_src.memory.prompting import format_followup_lines

    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    items = [
        MemoryItem(
            memory_id="mem-future",
            scope_type="person",
            scope_id="person-1",
            kind="followup",
            key="cape_cod_trip",
            summary="Cape Cod trip with their parents planned for mid-May.",
            source="live_chat",
            due_at="2026-05-18T13:00:00+00:00",
            expires_at="2026-05-22T23:59:00+00:00",
        ),
        MemoryItem(
            memory_id="mem-due",
            scope_type="person",
            scope_id="person-1",
            kind="followup",
            key="luna_recovery",
            summary="Luna is recovering from surgery.",
            source="live_chat",
            due_at="2026-05-14T13:00:00+00:00",
            expires_at="2026-05-20T23:59:00+00:00",
        ),
    ]

    assert format_followup_lines(items, now=now) == (
        "Luna is recovering from surgery.",
    )


def test_site_blocks_include_only_relation_relevant_encounters(tmp_path):
    from argos_src.identity import IdentityStore
    from argos_src.memory import MemoryContextCompiler, MemoryStore

    identity_store = IdentityStore(tmp_path / "identity.sqlite3")
    current_id = identity_store.create_person(
        name="Sakshee Patil",
        metadata={"manager_name": "Dan Burns"},
    )
    memory_store = MemoryStore(tmp_path / "memory.sqlite3")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    memory_store.record_encounter(
        person_id="person-alex",
        name="Alex Kim",
        site_code="BOS3",
        metadata={"name": "Alex Kim", "manager_name": "Dan Burns"},
        observed_at=(now - timedelta(minutes=10)).isoformat(),
    )
    memory_store.record_encounter(
        person_id="person-riley",
        name="Riley Lee",
        site_code="BOS3",
        metadata={"name": "Riley Lee", "manager_name": "Different Manager"},
        observed_at=(now - timedelta(minutes=5)).isoformat(),
    )

    compiler = MemoryContextCompiler(memory_store, identity_store=identity_store)

    blocks = compiler.site_blocks("BOS3", current_person_id=current_id)
    rendered = "\n".join(blocks)

    assert "Alex Kim" in rendered
    assert "same manager" in rendered
    assert "Riley Lee" not in rendered
