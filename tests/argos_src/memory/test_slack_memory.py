from __future__ import annotations

from types import SimpleNamespace


def test_slack_prompt_attributes_mentions_to_mentioned_people():
    from argos_src.memory.slack.extract import build_slack_extraction_prompt
    from argos_src.memory.slack.models import SlackChannelWindow, SlackMessage
    from argos_src.memory.slack.normalize import normalize_message
    from argos_src.memory.slack.models import SlackUserProfile

    profiles = {
        "UOLIVIA": SlackUserProfile(
            slack_user_id="UOLIVIA",
            username="olivia",
            display_name="Olivia Ordonez",
            person_id="person_olivia_ordonez",
        ),
        "UJOSEPH": SlackUserProfile(
            slack_user_id="UJOSEPH",
            username="joseph",
            display_name="Joseph Papagno",
            person_id="person_joseph_papagno",
        ),
        "UTHOMAS": SlackUserProfile(
            slack_user_id="UTHOMAS",
            username="thomas",
            display_name="Thomas Walewski",
            person_id="person_thomas_walewski",
        ),
    }
    message = normalize_message(
        {
            "type": "message",
            "user": "UOLIVIA",
            "text": (
                "Happy 2 year Chewy-versary <@UJOSEPH> and <@UTHOMAS>!"
            ),
            "ts": "1780339980.000100",
        },
        channel_id="C123",
        channel_name="argos-test",
        user_profiles=profiles,
    )
    assert message is not None
    window = SlackChannelWindow(
        channel_name="argos-test",
        channel_id="C123",
        site_code="BOS3",
        start_ts="1780339000.000000",
        end_ts="1780340800.000000",
        messages=(message,),
    )

    prompt = build_slack_extraction_prompt(
        window=window,
        current_date="2026-06-01",
        current_time="16:30 EDT",
        candidate_memories=[],
    )

    assert "Olivia Ordonez (@olivia):" in prompt
    assert "@Joseph Papagno (@joseph)" in prompt
    assert "mentioned_users: Joseph Papagno (@joseph), Thomas Walewski (@thomas)" in prompt
    assert "slack_user_id" not in prompt
    assert "person_id=" not in prompt
    assert "person_olivia_ordonez" not in prompt
    assert "C123:1780339980.000100" not in prompt
    assert "Do not store that milestone" in prompt
    assert '"target_users":' in prompt
    assert "milestone: completed 2 years at Chewy on 2026-06-01." in prompt


def test_slack_writer_creates_mentioned_people_milestones(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.slack.models import SlackUserProfile
    from argos_src.memory.slack.writer import write_slack_memory_operations

    store = MemoryStore(tmp_path / "memory.sqlite3")
    affected = write_slack_memory_operations(
        store,
        source_ref="C123:1780339980.000100",
        operations={
            "update": True,
            "ops": [
                {
                    "op": "create",
                    "scope_type": "person",
                    "target_users": ["@joseph", "@thomas"],
                    "kind": "fact",
                    "key": "chewy_versary_2_year_2026_06_01",
                    "summary": "milestone: completed 2 years at Chewy on 2026-06-01.",
                    "source_refs": ["C123:1780339980.000100"],
                },
            ],
        },
        slack_user_profiles={
            "UJOSEPH": SlackUserProfile(
                slack_user_id="UJOSEPH",
                username="joseph",
                display_name="Joseph Papagno",
                person_id="person_joseph_papagno",
            ),
            "UTHOMAS": SlackUserProfile(
                slack_user_id="UTHOMAS",
                username="thomas",
                display_name="Thomas Walewski",
                person_id="person_thomas_walewski",
            ),
        },
    )

    assert len(affected) == 2
    joseph = store.list_items(scope_type="person", scope_id="person_joseph_papagno")
    thomas = store.list_items(scope_type="person", scope_id="person_thomas_walewski")
    olivia = store.list_items(scope_type="person", scope_id="person_olivia_ordonez")
    assert joseph[0].source == "slack"
    assert joseph[0].summary == "milestone: completed 2 years at Chewy on 2026-06-01."
    assert thomas[0].summary == "milestone: completed 2 years at Chewy on 2026-06-01."
    assert joseph[0].metadata["slack_user_id"] == "UJOSEPH"
    assert thomas[0].metadata["slack_user_id"] == "UTHOMAS"
    assert olivia == []


def test_slack_writer_stages_unmapped_person_memory(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.slack.models import SlackUserProfile
    from argos_src.memory.slack.pending import list_pending_slack_memory
    from argos_src.memory.slack.writer import write_slack_memory_operations

    store = MemoryStore(tmp_path / "memory.sqlite3")
    affected = write_slack_memory_operations(
        store,
        source_ref="C123:1780339980.000100",
        operations={
            "update": True,
            "ops": [
                {
                    "op": "create",
                    "scope_type": "person",
                    "target_users": ["@newperson"],
                    "kind": "fact",
                    "key": "chewy_versary_2_year_2026_06_01",
                    "summary": "milestone: completed 2 years at Chewy on 2026-06-01.",
                    "source_refs": ["C123:1780339980.000100"],
                }
            ],
        },
        slack_user_profiles={
            "UNEW": SlackUserProfile(
                slack_user_id="UNEW",
                username="newperson",
                display_name="New Person",
                real_name="New Person",
                email="new.person@example.com",
            )
        },
    )

    assert len(affected) == 1
    assert store.list_items(scope_type="person", scope_id="UNEW") == []
    pending = list_pending_slack_memory(store, slack_user_id="UNEW")
    assert len(pending) == 1
    assert pending[0]["summary"] == "milestone: completed 2 years at Chewy on 2026-06-01."
    assert pending[0]["slack_email"] == "new.person@example.com"


def test_slack_candidate_memories_only_include_slack_site_events(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.slack.extract import candidate_memory_payload

    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.upsert_item(
        scope_type="person",
        scope_id="person_sakshee",
        kind="preference",
        key="comfort_food",
        summary="favorite comfort food is dal rice.",
        source="live_chat",
    )
    store.upsert_item(
        scope_type="site",
        scope_id="BOS3",
        kind="office_event",
        key="from_live_chat",
        summary="live-chat site event should not be sent to Slack extractor.",
        source="live_chat",
    )
    slack_site_id = store.upsert_item(
        scope_type="site",
        scope_id="BOS3",
        kind="office_event",
        key="name_tags",
        summary="name tags are being printed.",
        source="slack",
    )

    payload = candidate_memory_payload(store, site_code="BOS3")

    assert payload == [
        {
            "memory_id": slack_site_id,
            "scope_type": "site",
            "scope_id": "BOS3",
            "kind": "office_event",
            "key": "name_tags",
            "summary": "name tags are being printed.",
            "source": "slack",
        }
    ]


def test_slack_writer_does_not_update_or_archive_live_chat_memory(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.slack.writer import write_slack_memory_operations

    store = MemoryStore(tmp_path / "memory.sqlite3")
    live_chat_id = store.upsert_item(
        scope_type="person",
        scope_id="person_sakshee",
        kind="preference",
        key="comfort_food",
        summary="favorite comfort food is dal rice.",
        source="live_chat",
    )

    affected = write_slack_memory_operations(
        store,
        operations={
            "update": True,
            "ops": [
                {
                    "op": "update",
                    "scope_type": "person",
                    "memory_id": live_chat_id,
                    "summary": "changed by Slack",
                },
                {
                    "op": "archive",
                    "scope_type": "person",
                    "memory_id": live_chat_id,
                },
            ],
        },
    )

    item = store.get_item(live_chat_id)
    assert affected == []
    assert item is not None
    assert item.summary == "favorite comfort food is dal rice."
    assert item.status == "active"


def test_promote_pending_slack_memory_after_identity_link(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.slack.models import SlackUserProfile
    from argos_src.memory.slack.pending import (
        list_pending_slack_memory,
        promote_pending_slack_memory,
        upsert_pending_slack_memory,
    )

    store = MemoryStore(tmp_path / "memory.sqlite3")
    upsert_pending_slack_memory(
        store,
        profile=SlackUserProfile(
            slack_user_id="UNEW",
            display_name="New Person",
            email="new.person@example.com",
        ),
        kind="fact",
        key="favorite_snack",
        summary="preference: likes dark chocolate.",
        source_ref="C123:1780339980.000100",
    )

    affected = promote_pending_slack_memory(
        store,
        slack_user_id="UNEW",
        person_id="person_new_person",
    )

    assert len(affected) == 1
    items = store.list_items(scope_type="person", scope_id="person_new_person")
    assert len(items) == 1
    assert items[0].source == "slack"
    assert items[0].metadata["slack_user_id"] == "UNEW"
    assert list_pending_slack_memory(store, slack_user_id="UNEW") == []
    promoted = list_pending_slack_memory(store, slack_user_id="UNEW", status="promoted")
    assert promoted[0]["promoted_person_id"] == "person_new_person"


def test_promote_resolved_pending_slack_memory_from_identity_store(tmp_path):
    from argos_src.identity.store import IdentityStore
    from argos_src.memory import MemoryStore
    from argos_src.memory.slack.identity import SlackIdentityResolver
    from argos_src.memory.slack.models import SlackUserProfile
    from argos_src.memory.slack.pending import (
        list_pending_slack_memory,
        promote_resolved_pending_slack_memory,
        upsert_pending_slack_memory,
    )

    store = MemoryStore(tmp_path / "memory.sqlite3")
    identity_store = IdentityStore(tmp_path / "identity.sqlite3")
    identity_store.create_person(
        name="New Person",
        person_id="person_new_person",
        metadata={"username": "newperson"},
    )
    upsert_pending_slack_memory(
        store,
        profile=SlackUserProfile(
            slack_user_id="UNEW",
            username="newperson",
            display_name="New Person",
        ),
        kind="fact",
        key="favorite_snack",
        summary="preference: likes dark chocolate.",
        source_ref="C123:1780339980.000100",
        metadata={"slack_username": "newperson"},
    )

    affected = promote_resolved_pending_slack_memory(
        store,
        identity_resolver=SlackIdentityResolver(identity_store),
    )

    assert len(affected) == 1
    assert len(store.list_items(scope_type="person", scope_id="person_new_person")) == 1
    assert list_pending_slack_memory(store, slack_user_id="UNEW") == []


def test_slack_writer_forces_site_scope_to_configured_site(tmp_path):
    from argos_src.memory import MemoryStore
    from argos_src.memory.slack.writer import write_slack_memory_operations

    store = MemoryStore(tmp_path / "memory.sqlite3")
    affected = write_slack_memory_operations(
        store,
        default_site_code="BOS3",
        operations={
            "update": True,
            "ops": [
                {
                    "op": "create",
                    "scope_type": "site",
                    "scope_id": "WRONG",
                    "kind": "office_event",
                    "key": "snacks_today",
                    "summary": "2026-06-01: Snacks are in the kitchen.",
                    "expires_at": "2026-06-01T17:00:00-04:00",
                }
            ],
        },
    )

    assert len(affected) == 1
    assert store.list_items(scope_type="site", scope_id="WRONG") == []
    assert len(store.list_items(scope_type="site", scope_id="BOS3")) == 1


def test_slack_service_enabled_without_token_is_safe(tmp_path, monkeypatch):
    from argos_src.memory import MemoryStore
    from argos_src.memory.slack import SlackMemoryService

    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    store = MemoryStore(tmp_path / "memory.sqlite3")
    profile = SimpleNamespace(
        enabled=True,
        bot_token_env="SLACK_BOT_TOKEN",
        lookback_minutes=30,
        poll_interval_sec=1800.0,
        channels=(SimpleNamespace(name="argos-test", channel_id="C123"),),
    )
    service = SlackMemoryService(profile=profile, memory_store=store)

    service.run_once()


def test_slack_service_strips_reactions_from_raw_messages():
    from argos_src.memory.slack import SlackMemoryService

    cleaned = SlackMemoryService._message_without_reactions(
        {
            "type": "message",
            "user": "U123",
            "text": "hello",
            "ts": "1780339980.000100",
            "reactions": [{"name": "thumbsup", "users": ["U999"]}],
        }
    )

    assert cleaned == {
        "type": "message",
        "user": "U123",
        "text": "hello",
        "ts": "1780339980.000100",
    }
