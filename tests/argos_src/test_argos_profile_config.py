from pathlib import Path

import pytest

from argos_src.memory.constants import DEFAULT_MEMORY_DB_PATH
from argos_src.profile_config import DEFAULT_FACE_DB_PATH, ProfileValidationError, _parse_profile


def test_robot_profile_parses_bridge_settings():
    profile = _parse_profile(
        {
            "name": "robot-bridge",
            "robot": {
                "id": "puffle",
                "family": "unitree_go2",
                "display_name": "Puffle",
                "bridge": {
                    "transport": "zenoh",
                    "key_prefix": "pair/robots/puffle",
                    "connect_endpoints": ["tcp/127.0.0.1:7447"],
                },
            },
        },
        profile_path=Path("/tmp/robot-bridge.yaml"),
        framework_config={},
    )

    assert profile.robot_family == "unitree_go2"
    assert profile.robot.id == "puffle"
    assert profile.robot.family == "unitree_go2"
    assert profile.robot.display_name == "Puffle"
    assert profile.robot.bridge.transport == "zenoh"
    assert profile.robot.bridge.key_prefix == "pair/robots/puffle"
    assert profile.robot.bridge.connect_endpoints == ("tcp/127.0.0.1:7447",)


def test_legacy_robot_family_populates_robot_profile():
    profile = _parse_profile(
        {
            "name": "legacy-family",
            "robot_family": "spot",
        },
        profile_path=Path("/tmp/legacy-family.yaml"),
        framework_config={},
    )

    assert profile.robot_family == "spot"
    assert profile.robot.family == "spot"
    assert profile.robot.bridge.key_prefix is None


def test_robot_family_conflict_is_rejected():
    with pytest.raises(ProfileValidationError, match="must match"):
        _parse_profile(
            {
                "name": "family-conflict",
                "robot_family": "spot",
                "robot": {
                    "family": "unitree_go2",
                },
            },
            profile_path=Path("/tmp/family-conflict.yaml"),
            framework_config={},
        )


def test_realtime_defaults_do_not_inherit_legacy_asr_tts_config():
    profile = _parse_profile(
        {"name": "minimal"},
        profile_path=Path("/tmp/minimal.yaml"),
        framework_config={
            "asr": {
                "recording_device_name": "legacy-mic",
                "language": "fr",
                "vad_threshold": 0.1,
                "silence_grace_period": 9.0,
                "wake_word_threshold": 0.9,
            },
            "tts": {"speaker_device_name": "legacy-speaker"},
        },
    )

    assert profile.realtime.input_device == "pulse"
    assert profile.realtime.output_device == "pulse"
    assert profile.realtime.language is None
    assert profile.realtime.vad_threshold == 0.8
    assert profile.realtime.silence_grace_period == 0.3
    assert profile.realtime.wake_threshold == 0.5
    assert profile.realtime.audio_output_speed == 0.9


def test_legacy_interaction_state_topic_is_rejected():
    with pytest.raises(ProfileValidationError, match="realtime"):
        _parse_profile(
            {
                "name": "legacy-topic",
                "realtime": {"interaction_state_topic": "/go2/interaction_state"},
            },
            profile_path=Path("/tmp/legacy-topic.yaml"),
            framework_config={},
        )


def test_realtime_temperature_is_rejected_after_realtime_ga_migration():
    with pytest.raises(ProfileValidationError, match="temperature"):
        _parse_profile(
            {
                "name": "legacy-temperature",
                "realtime": {"temperature": 0.8},
            },
            profile_path=Path("/tmp/legacy-temperature.yaml"),
            framework_config={},
        )


def test_legacy_agent_section_is_rejected():
    with pytest.raises(ProfileValidationError, match="Move prompt_file to realtime.prompt_file"):
        _parse_profile(
            {
                "name": "legacy-agent",
                "agent": {
                    "prompt_file": "static_interaction_prompt.md",
                },
                "realtime": {"model": "gpt-realtime-1.5"},
            },
            profile_path=Path("/tmp/legacy-agent.yaml"),
            framework_config={},
        )


def test_realtime_prompt_file_is_loaded_from_realtime_namespace():
    profile = _parse_profile(
        {
            "name": "prompt-test",
            "realtime": {
                "prompt_file": "static_interaction_prompt.md",
            },
        },
        profile_path=Path("/tmp/prompt-test.yaml"),
        framework_config={},
    )

    assert profile.realtime.prompt_file == "static_interaction_prompt.md"


def test_embodiment_gesture_defaults_are_disabled():
    profile = _parse_profile(
        {"name": "embodiment-defaults"},
        profile_path=Path("/tmp/embodiment-defaults.yaml"),
        framework_config={},
    )

    assert profile.embodiment.gestures.enabled is False
    assert profile.embodiment.gestures.preset == "auto"
    assert profile.embodiment.gestures.tilt_enabled is True
    assert profile.embodiment.gestures.nodding_enabled is True


def test_embodiment_gesture_states_can_be_disabled_independently():
    profile = _parse_profile(
        {
            "name": "embodiment-state-toggles",
            "embodiment": {
                "gestures": {
                    "enabled": True,
                    "tilt_enabled": False,
                    "nodding_enabled": True,
                }
            },
        },
        profile_path=Path("/tmp/embodiment-state-toggles.yaml"),
        framework_config={},
    )

    assert profile.embodiment.gestures.enabled is True
    assert profile.embodiment.gestures.tilt_enabled is False
    assert profile.embodiment.gestures.nodding_enabled is True


def test_unknown_gesture_preset_is_rejected():
    with pytest.raises(ProfileValidationError, match="Unknown gesture preset"):
        _parse_profile(
            {
                "name": "bad-gesture-preset",
                "embodiment": {
                    "gestures": {
                        "enabled": True,
                        "preset": "nope",
                    }
                },
            },
            profile_path=Path("/tmp/bad-gesture-preset.yaml"),
            framework_config={},
        )


def test_employee_directory_defaults_to_disabled():
    profile = _parse_profile(
        {"name": "employee-directory-defaults"},
        profile_path=Path("/tmp/employee-directory-defaults.yaml"),
        framework_config={},
    )

    assert profile.employee_directory.enabled is False
    assert profile.employee_directory.site_code == ""


def test_employee_directory_requires_site_code_when_enabled():
    with pytest.raises(ProfileValidationError, match="employee_directory.site_code"):
        _parse_profile(
            {
                "name": "employee-directory-enabled",
                "employee_directory": {
                    "enabled": True,
                },
            },
            profile_path=Path("/tmp/employee-directory-enabled.yaml"),
            framework_config={},
        )


def test_employee_directory_rejects_unknown_keys():
    with pytest.raises(ProfileValidationError, match="employee_directory"):
        _parse_profile(
            {
                "name": "employee-directory-unknown",
                "employee_directory": {
                    "enabled": False,
                    "site_code": "",
                    "extra": True,
                },
            },
            profile_path=Path("/tmp/employee-directory-unknown.yaml"),
            framework_config={},
        )


def test_employee_directory_tool_is_removed_when_directory_disabled():
    profile = _parse_profile(
        {
            "name": "employee-directory-disabled-tool",
            "tools": {
                "enabled_tool_names": [
                    "unitree_go2.vision.resolve_employee_identity",
                ]
            },
            "employee_directory": {
                "enabled": False,
                "site_code": "",
            },
        },
        profile_path=Path("/tmp/employee-directory-disabled-tool.yaml"),
        framework_config={},
    )

    assert profile.tools.enabled_tool_names == ()


def test_employee_directory_tool_is_kept_when_directory_enabled():
    profile = _parse_profile(
        {
            "name": "employee-directory-enabled-tool",
            "tools": {
                "enabled_tool_names": [
                    "unitree_go2.vision.resolve_employee_identity",
                ]
            },
            "employee_directory": {
                "enabled": True,
                "site_code": "BOS3",
            },
        },
        profile_path=Path("/tmp/employee-directory-enabled-tool.yaml"),
        framework_config={},
    )

    assert profile.tools.enabled_tool_names == (
        "unitree_go2.vision.resolve_employee_identity",
    )
    assert profile.employee_directory.site_code == "BOS3"


def test_face_db_path_resolves_from_repo_root_not_cwd():
    profile = _parse_profile(
        {
            "name": "face-db-path",
            "face_recognition": {
                "db_path": "argos_src/face_recognition/db",
            },
        },
        profile_path=Path("/tmp/face-db-path.yaml"),
        framework_config={},
    )

    assert profile.face_recognition.db_path == DEFAULT_FACE_DB_PATH


def test_face_owner_turn_profile_is_configurable():
    profile = _parse_profile(
        {
            "name": "owner-turn",
            "face_recognition": {
                "owner_turn": {
                    "enabled": True,
                    "camera_info_topic": "/camera/color/camera_info",
                    "camera_yaw_offset_rad": 0.1,
                    "deadband_deg": 2.0,
                    "turn_gain": 0.7,
                    "max_turn_deg": 20.0,
                    "angular_speed_rad_s": 0.9,
                    "command_hz": 40.0,
                    "delay_after_recording_sec": 0.1,
                    "odom_frame": "odom",
                    "robot_frame": "base_link",
                    "yaw_tolerance_deg": 1.2,
                    "max_duration_sec": 1.4,
                    "slow_zone_deg": 7.0,
                    "min_angular_speed_rad_s": 0.2,
                },
            },
        },
        profile_path=Path("/tmp/owner-turn.yaml"),
        framework_config={},
    )

    owner_turn = profile.face_recognition.owner_turn
    assert owner_turn.enabled is True
    assert owner_turn.camera_info_topic == "/camera/color/camera_info"
    assert owner_turn.camera_yaw_offset_rad == pytest.approx(0.1)
    assert owner_turn.deadband_deg == pytest.approx(2.0)
    assert owner_turn.turn_gain == pytest.approx(0.7)
    assert owner_turn.max_turn_deg == pytest.approx(20.0)
    assert owner_turn.angular_speed_rad_s == pytest.approx(0.9)
    assert owner_turn.command_hz == pytest.approx(40.0)
    assert owner_turn.delay_after_recording_sec == pytest.approx(0.1)
    assert owner_turn.odom_frame == "odom"
    assert owner_turn.robot_frame == "base_link"
    assert owner_turn.yaw_tolerance_deg == pytest.approx(1.2)
    assert owner_turn.max_duration_sec == pytest.approx(1.4)
    assert owner_turn.slow_zone_deg == pytest.approx(7.0)
    assert owner_turn.min_angular_speed_rad_s == pytest.approx(0.2)


def test_memory_store_path_defaults_and_resolves_from_repo_root():
    profile = _parse_profile(
        {
            "name": "memory-db-path",
            "memory_store": {
                "db_path": "argos_src/memory/db/memory.sqlite3",
            },
        },
        profile_path=Path("/tmp/memory-db-path.yaml"),
        framework_config={},
    )

    assert profile.memory_store.db_path == str(DEFAULT_MEMORY_DB_PATH)


def test_slack_memory_profile_parses_channel_wiring():
    profile = _parse_profile(
        {
            "name": "slack-memory",
            "slack_memory": {
                "enabled": True,
                "start_with_agent": False,
                "bot_token_env": "CHEWY_SLACK_BOT_TOKEN",
                "poll_interval_sec": 900.0,
                "lookback_minutes": 45,
                "channels": [
                    {
                        "name": "argos-test",
                        "channel_id": "C123",
                        "site_code": "BOS3",
                        "person_memory_enabled": True,
                        "site_memory_enabled": False,
                        "include_threads": True,
                        "max_messages_per_window": 50,
                    }
                ],
            },
        },
        profile_path=Path("/tmp/slack-memory.yaml"),
        framework_config={},
    )

    assert profile.slack_memory.enabled is True
    assert profile.slack_memory.start_with_agent is False
    assert profile.slack_memory.bot_token_env == "CHEWY_SLACK_BOT_TOKEN"
    assert profile.slack_memory.poll_interval_sec == 900.0
    assert profile.slack_memory.lookback_minutes == 45
    assert len(profile.slack_memory.channels) == 1
    channel = profile.slack_memory.channels[0]
    assert channel.name == "argos-test"
    assert channel.channel_id == "C123"
    assert channel.site_code == "BOS3"
    assert channel.site_memory_enabled is False
    assert channel.include_threads is True


def test_slack_memory_channel_requires_name():
    with pytest.raises(ProfileValidationError, match="slack_memory.channels"):
        _parse_profile(
            {
                "name": "bad-slack-memory",
                "slack_memory": {
                    "enabled": True,
                    "channels": [{"channel_id": "C123"}],
                },
            },
            profile_path=Path("/tmp/bad-slack-memory.yaml"),
            framework_config={},
        )


def test_speaker_recognition_rejects_internal_backend_keys():
    with pytest.raises(ProfileValidationError, match="speaker_recognition"):
        _parse_profile(
            {
                "name": "speaker-flags",
                "speaker_recognition": {
                    "enabled": True,
                    "backend": "my_totally_real_backend",
                },
            },
            profile_path=Path("/tmp/speaker-flags.yaml"),
            framework_config={},
        )


def test_speaker_recognition_threshold_knobs_are_configurable():
    profile = _parse_profile(
        {
            "name": "speaker-thresholds",
            "speaker_recognition": {
                "enabled": True,
                "query_match_threshold": 0.81,
                "query_margin_threshold": 0.11,
                "reference_update_threshold": 0.57,
                "enroll_min_rms_level": 420.0,
                "max_clipped_fraction": 0.05,
            },
        },
        profile_path=Path("/tmp/speaker-thresholds.yaml"),
        framework_config={},
    )

    policy = profile.speaker_recognition.policy
    assert policy.query_match_threshold == pytest.approx(0.81)
    assert policy.query_margin_threshold == pytest.approx(0.11)
    assert policy.reference_update_threshold == pytest.approx(0.57)
    assert policy.enroll_min_rms_level == pytest.approx(420.0)
    assert policy.max_clipped_fraction == pytest.approx(0.05)


def test_speaker_recognition_rejects_removed_word_count_knobs():
    with pytest.raises(ProfileValidationError, match="speaker_recognition"):
        _parse_profile(
            {
                "name": "speaker-old-knobs",
                "speaker_recognition": {
                    "enabled": True,
                    "query_min_words": 2,
                    "enroll_min_words": 4,
                },
            },
            profile_path=Path("/tmp/speaker-old-knobs.yaml"),
            framework_config={},
        )
