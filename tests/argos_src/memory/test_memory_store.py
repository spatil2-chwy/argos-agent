from __future__ import annotations

from datetime import datetime, timedelta, timezone


def test_memory_store_upserts_and_filters_active_items(tmp_path):
    from argos_src.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="preference",
        key="likes_cape",
        summary="likes: blue waters at the Cape",
        source="live_chat",
    )
    store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="followup",
        key="old_dog_update",
        summary="Ask about Luna.",
        source="live_chat",
        expires_at="2026-01-01T00:00:00+00:00",
    )
    archived_id = store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="note",
        key="archived",
        summary="hidden note",
        source="live_chat",
    )
    assert store.archive_item(archived_id) is True

    items = store.list_active_items(scope_type="person", scope_id="person-1")

    assert [item.summary for item in items] == ["likes: blue waters at the Cape"]


def test_memory_store_dedupes_by_scope_kind_key_and_source(tmp_path):
    from argos_src.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory.sqlite3")
    first = store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="preference",
        key="likes_cape",
        summary="likes: Cape beaches",
        source="live_chat",
    )
    second = store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="preference",
        key="likes_cape",
        summary="likes: blue waters at the Cape",
        source="live_chat",
    )

    items = store.list_active_items(scope_type="person", scope_id="person-1")

    assert second == first
    assert len(items) == 1
    assert items[0].summary == "likes: blue waters at the Cape"


def test_memory_store_filters_items_by_source(tmp_path):
    from argos_src.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="preference",
        key="favorite_snack",
        summary="likes dark chocolate.",
        source="live_chat",
    )
    store.upsert_item(
        scope_type="person",
        scope_id="person-1",
        kind="fact",
        key="project_alpha",
        summary="working on project Alpha.",
        source="slack",
    )

    active = store.list_active_items(
        scope_type="person",
        scope_id="person-1",
        source="slack",
    )
    all_items = store.list_items(
        scope_type="person",
        scope_id="person-1",
        source="slack",
    )

    assert [item.source for item in active] == ["slack"]
    assert [item.summary for item in all_items] == ["working on project Alpha."]


def test_memory_store_records_recent_encounters(tmp_path):
    from argos_src.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory.sqlite3")
    observed = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    store.record_encounter(
        person_id="person-alex",
        name="Alex Kim",
        site_code="BOS3",
        metadata={"business_function": "AI & Data"},
        observed_at=observed,
    )

    recent = store.list_recent_encounters(
        site_code="BOS3",
        since=datetime.now(timezone.utc) - timedelta(minutes=5),
    )

    assert len(recent) == 1
    assert recent[0].metadata["name"] == "Alex Kim"
    assert recent[0].metadata["site_code"] == "BOS3"
    assert recent[0].expires_at


def test_memory_store_recent_encounters_limits_after_site_filter(tmp_path):
    from argos_src.memory import MemoryStore

    store = MemoryStore(tmp_path / "memory.sqlite3")
    base = datetime.now(timezone.utc).replace(microsecond=0)
    for index in range(5):
        store.record_encounter(
            person_id=f"person-wrong-{index}",
            name=f"Wrong Site {index}",
            site_code="SEA1",
            observed_at=(base + timedelta(seconds=index + 10)).isoformat(),
        )
    store.record_encounter(
        person_id="person-right",
        name="Right Site",
        site_code="BOS3",
        observed_at=base.isoformat(),
    )

    recent = store.list_recent_encounters(
        site_code="BOS3",
        since=base - timedelta(minutes=1),
        limit=5,
    )

    assert [item.metadata["name"] for item in recent] == ["Right Site"]
