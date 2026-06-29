from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import argos_src.memory_provider.slack as slack_module
from argos_src.memory_provider.slack import TailwagSlackMemoryService
from argos_src.profile_config import SlackMemoryChannelProfile, SlackMemoryProfile


class FakeEpisodeRecorder:
    retention_class = "privacy-reviewed"


@pytest.fixture()
def fake_tailwag_slack_ingestion(monkeypatch):
    calls = SimpleNamespace(web_clients=[], pollers=[], poll_calls=[])

    class FakeSlackWebApiClient:
        def __init__(self, token, *, include_email=True):
            self.token = token
            self.include_email = include_email
            calls.web_clients.append(self)

    class FakeSlackMemoryPoller:
        def __init__(
            self,
            slack_client,
            episode_recorder,
            state_path,
            *,
            retention_class,
            active_thread_hours,
        ):
            self.slack_client = slack_client
            self.episode_recorder = episode_recorder
            self.state_path = state_path
            self.retention_class = retention_class
            self.active_thread_hours = active_thread_hours
            self.poll_calls = []
            calls.pollers.append(self)

        def poll_once(
            self,
            channel_id,
            *,
            backfill_hours,
            force_backfill,
            history_limit,
            reply_limit,
            extract_memory,
        ):
            call = {
                "channel_id": channel_id,
                "backfill_hours": backfill_hours,
                "force_backfill": force_backfill,
                "history_limit": history_limit,
                "reply_limit": reply_limit,
                "extract_memory": extract_memory,
            }
            self.poll_calls.append(call)
            calls.poll_calls.append(call)
            return SimpleNamespace(
                checked_threads=3,
                ingested_threads=2,
                armed_without_backfill=1,
            )

    tailwag_module = ModuleType("tailwag_memory")
    tailwag_module.__path__ = []
    slack_ingestion_module = ModuleType("tailwag_memory.slack_ingestion")
    slack_ingestion_module.SlackWebApiClient = FakeSlackWebApiClient
    slack_ingestion_module.SlackMemoryPoller = FakeSlackMemoryPoller
    monkeypatch.setitem(sys.modules, "tailwag_memory", tailwag_module)
    monkeypatch.setitem(
        sys.modules,
        "tailwag_memory.slack_ingestion",
        slack_ingestion_module,
    )
    return calls


def test_service_configures_tailwag_poller_with_token_email_state_and_retention(
    fake_tailwag_slack_ingestion,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("ARGOS_TEST_SLACK_TOKEN", "xoxb-test-token")
    state_path = tmp_path / "slack-state.json"
    profile = SlackMemoryProfile(
        enabled=True,
        bot_token_env="ARGOS_TEST_SLACK_TOKEN",
        state_path=str(state_path),
        active_thread_hours=12.5,
        include_email=True,
        channels=(
            SlackMemoryChannelProfile(name="operator-label", channel_id="C123"),
        ),
    )

    service = TailwagSlackMemoryService(
        profile=profile,
        episode_recorder=FakeEpisodeRecorder(),
    )
    poller = service._get_poller()

    assert len(fake_tailwag_slack_ingestion.web_clients) == 1
    slack_client = fake_tailwag_slack_ingestion.web_clients[0]
    assert slack_client.token == "xoxb-test-token"
    assert slack_client.include_email is True
    assert poller.state_path == state_path
    assert poller.retention_class == "privacy-reviewed"
    assert poller.active_thread_hours == 12.5
    assert service._get_poller() is poller


def test_service_defaults_state_path_token_env_and_email_inclusion(
    fake_tailwag_slack_ingestion,
    monkeypatch,
):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-default-token")
    profile = SlackMemoryProfile(
        enabled=True,
        channels=(SlackMemoryChannelProfile(name="argos-test", channel_id="C123"),),
    )

    service = TailwagSlackMemoryService(
        profile=profile,
        episode_recorder=FakeEpisodeRecorder(),
    )
    poller = service._get_poller()

    assert fake_tailwag_slack_ingestion.web_clients[0].token == "xoxb-default-token"
    assert fake_tailwag_slack_ingestion.web_clients[0].include_email is True
    assert poller.state_path == Path(".tailwag/slack-state.json")
    assert poller.active_thread_hours == 24.0


def test_poll_once_passes_channel_ids_and_polling_parameters(
    fake_tailwag_slack_ingestion,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("ARGOS_TEST_SLACK_TOKEN", "xoxb-test-token")
    profile = SlackMemoryProfile(
        enabled=True,
        bot_token_env="ARGOS_TEST_SLACK_TOKEN",
        state_path=str(tmp_path / "state.json"),
        backfill_hours=2.0,
        force_backfill=True,
        history_limit=50,
        reply_limit=75,
        extract_memory=False,
        channels=(
            SlackMemoryChannelProfile(
                name="operator-label",
                channel_id="C123",
                backfill_hours=1.0,
            ),
        ),
    )
    service = TailwagSlackMemoryService(
        profile=profile,
        episode_recorder=FakeEpisodeRecorder(),
    )

    service.poll_once()

    assert fake_tailwag_slack_ingestion.poll_calls == [
        {
            "channel_id": "C123",
            "backfill_hours": 1.0,
            "force_backfill": True,
            "history_limit": 50,
            "reply_limit": 75,
            "extract_memory": False,
        },
    ]


def test_poll_once_skips_channels_without_channel_id(
    fake_tailwag_slack_ingestion,
    monkeypatch,
):
    monkeypatch.setenv("ARGOS_TEST_SLACK_TOKEN", "xoxb-test-token")
    warnings = []
    monkeypatch.setattr(
        slack_module.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    profile = SlackMemoryProfile(
        enabled=True,
        bot_token_env="ARGOS_TEST_SLACK_TOKEN",
        channels=(
            SlackMemoryChannelProfile(name=" "),
            SlackMemoryChannelProfile(name="argos-test", channel_id="C123"),
        ),
    )
    service = TailwagSlackMemoryService(
        profile=profile,
        episode_recorder=FakeEpisodeRecorder(),
    )

    service.poll_once()

    assert [call["channel_id"] for call in fake_tailwag_slack_ingestion.poll_calls] == [
        "C123"
    ]
    assert any("without channel_id" in message for message in warnings)


def test_poll_once_with_no_channels_makes_no_channel_poll_calls(
    fake_tailwag_slack_ingestion,
    monkeypatch,
):
    monkeypatch.setenv("ARGOS_TEST_SLACK_TOKEN", "xoxb-test-token")
    profile = SlackMemoryProfile(
        enabled=True,
        bot_token_env="ARGOS_TEST_SLACK_TOKEN",
        channels=(),
    )
    service = TailwagSlackMemoryService(
        profile=profile,
        episode_recorder=FakeEpisodeRecorder(),
    )

    service.poll_once()

    assert len(fake_tailwag_slack_ingestion.pollers) == 1
    assert fake_tailwag_slack_ingestion.poll_calls == []


def test_poll_once_without_token_fails_before_constructing_tailwag_client(
    fake_tailwag_slack_ingestion,
    monkeypatch,
):
    monkeypatch.delenv("ARGOS_TEST_SLACK_TOKEN", raising=False)
    profile = SlackMemoryProfile(
        enabled=True,
        bot_token_env="ARGOS_TEST_SLACK_TOKEN",
        channels=(SlackMemoryChannelProfile(name="argos-test", channel_id="C123"),),
    )
    service = TailwagSlackMemoryService(
        profile=profile,
        episode_recorder=FakeEpisodeRecorder(),
    )

    with pytest.raises(
        RuntimeError,
        match="ARGOS_TEST_SLACK_TOKEN is required for Tailwag Slack memory polling",
    ):
        service.poll_once()

    assert fake_tailwag_slack_ingestion.web_clients == []
    assert fake_tailwag_slack_ingestion.pollers == []


def test_start_background_is_idempotent_and_shutdown_stops_polling(monkeypatch):
    poll_started = threading.Event()
    release_poll = threading.Event()
    poll_calls = []
    profile = SlackMemoryProfile(
        enabled=True,
        poll_interval_sec=1.0,
        channels=(SlackMemoryChannelProfile(name="argos-test", channel_id="C123"),),
    )
    service = TailwagSlackMemoryService(
        profile=profile,
        episode_recorder=FakeEpisodeRecorder(),
    )

    def poll_once():
        poll_calls.append("poll")
        poll_started.set()
        release_poll.wait(timeout=1.0)

    monkeypatch.setattr(service, "poll_once", poll_once)

    service.start_background()
    assert poll_started.wait(timeout=1.0)
    first_thread = service._thread
    service.start_background()

    assert service._thread is first_thread
    release_poll.set()
    service.shutdown()

    assert poll_calls == ["poll"]
    assert service._thread is None
