"""Scenario profile loading and validation for the Argos realtime runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import tomli
import yaml
from argos_src.face_recognition.constants import DEFAULT_FACE_DB_PATH
from argos_src.identity.constants import DEFAULT_IDENTITY_DB_PATH
from argos_src.memory.constants import DEFAULT_MEMORY_DB_PATH
from argos_src.agent.gesture_runtime import resolve_gesture_preset_name
from argos_src.speaker_recognition.constants import DEFAULT_SPEAKER_DB_PATH
from argos_src.speaker_recognition.models import SpeakerRecognitionPolicy
from argos_src.provider_api.manifest import (
    ManifestValidationError,
    ProviderManifest,
    ProviderResource,
    load_provider_manifest,
)
from argos_src.tools.tool_ids import (
    ROBOT_FAMILY_UNITREE_GO2,
    SUPPORTED_ROBOT_FAMILIES,
    required_capability_ids_for_tool_id,
    resolve_builtin_tool_name,
    resolve_builtin_tool_names,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_CONFIG_PATH = REPO_ROOT / "config.toml"
PROFILES_DIR = REPO_ROOT / "config" / "profiles"
ARGOS_MODELS_DIR = REPO_ROOT / "resources" / "wake_words"
DEFAULT_PROFILE_NAME = "static_interaction"
DEFAULT_ROBOT_FAMILY = ROBOT_FAMILY_UNITREE_GO2


class ProfileValidationError(ValueError):
    """Raised when a scenario profile is missing or invalid."""


@dataclass(frozen=True)
class NavigationProfile:
    locations_file: Optional[str]
    startup_patrol_route: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeBaseProfile:
    kind: str
    root_dir: str
    tool_name: str
    description: str
    k: int = 4


@dataclass(frozen=True)
class ToolsProfile:
    enabled_tool_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmployeeDirectoryProfile:
    enabled: bool
    site_code: str


@dataclass(frozen=True)
class ProactiveGreetingProfile:
    recognized_enabled: bool
    unknown_enabled: bool
    recognized_cooldown_sec: float
    unknown_cooldown_sec: float


@dataclass(frozen=True)
class FaceOwnerTurnProfile:
    enabled: bool
    camera_yaw_offset_rad: float
    deadband_deg: float
    turn_gain: float
    max_turn_deg: float
    angular_speed_rad_s: float
    command_hz: float
    delay_after_recording_sec: float
    odom_frame: str
    robot_frame: str
    yaw_tolerance_deg: float
    max_duration_sec: float
    slow_zone_deg: float
    min_angular_speed_rad_s: float


@dataclass(frozen=True)
class FaceDepthGateProfile:
    enabled: bool
    sync_slop_sec: float
    sync_queue_size: int
    capture_timeout_sec: float
    max_face_depth_m: float
    min_valid_samples: int
    patch_size: int
    search_radius_px: int
    max_valid_depth_m: float


@dataclass(frozen=True)
class PreferenceExtractionProfile:
    enabled: bool


@dataclass(frozen=True)
class FaceRecognitionProfile:
    enabled: bool
    db_path: str
    loop_interval_sec: float
    recognition_threshold: float
    depth_gate: FaceDepthGateProfile
    owner_turn: FaceOwnerTurnProfile
    preference_extraction: PreferenceExtractionProfile
    proactive_greeting: ProactiveGreetingProfile


@dataclass(frozen=True)
class SpeakerRecognitionProfile:
    enabled: bool
    policy: SpeakerRecognitionPolicy


@dataclass(frozen=True)
class IdentityStoreProfile:
    db_path: str


@dataclass(frozen=True)
class MemoryStoreProfile:
    db_path: str


@dataclass(frozen=True)
class SlackMemoryChannelProfile:
    name: str
    channel_id: str = ""
    site_code: str = ""
    person_memory_enabled: bool = True
    site_memory_enabled: bool = True
    include_threads: bool = True
    max_messages_per_window: int = 200


@dataclass(frozen=True)
class SlackMemoryProfile:
    enabled: bool = False
    start_with_agent: bool = False
    bot_token_env: str = "SLACK_BOT_TOKEN"
    poll_interval_sec: float = 1800.0
    lookback_minutes: int = 30
    channels: tuple[SlackMemoryChannelProfile, ...] = ()


@dataclass(frozen=True)
class RealtimeAdmissionProfile:
    block_during_speaking: bool
    open_on_face_presence: bool
    open_on_interaction_states: tuple[str, ...]
    open_on_wake_window: bool


@dataclass(frozen=True)
class RealtimeProfile:
    prompt_file: Optional[str]
    model: str
    voice: str
    audio_output_speed: float
    transcription_model: Optional[str]
    input_device: Optional[str]
    output_device: Optional[str]
    noise_reduction: Optional[str]
    language: Optional[str]
    vad_threshold: float
    silence_grace_period: float
    wake_word: str
    wake_word_model: str
    wake_threshold: float
    wake_window_sec: float
    input_sample_rate: int
    output_sample_rate: int
    input_block_size: int
    max_output_tokens: Optional[int]
    admission: RealtimeAdmissionProfile


@dataclass(frozen=True)
class EngagementProfile:
    coalescer_debounce_sec: float
    coalescer_max_wait_sec: float
    alert_timeout_sec: float
    cooldown_sec: float
    speaking_timeout_sec: float
    startup_patrol_delay_sec: float
    patrol_next_hop_delay_sec: float


@dataclass(frozen=True)
class StartupProfile:
    prepare_robot: bool
    service_timeout_sec: float
    fail_on_prepare_error: bool


@dataclass(frozen=True)
class ProviderBindingProfile:
    transport: Optional[str] = None
    key_prefix: Optional[str] = None
    connect_endpoints: Optional[tuple[str, ...]] = None
    provider_id: Optional[str] = None
    resource_id: Optional[str] = None


@dataclass(frozen=True)
class RobotProfile:
    id: str = ""
    family: str = DEFAULT_ROBOT_FAMILY
    display_name: str = ""
    bridge: ProviderBindingProfile = field(default_factory=ProviderBindingProfile)


@dataclass(frozen=True)
class ResourceSelectionsProfile:
    primary_robot: str = ""
    face_camera: str = ""
    scene_camera: str = ""
    lidar: str = ""


@dataclass(frozen=True)
class BatteryProfile:
    enabled: bool
    low_battery_pct: float
    charging_ready_pct: float


@dataclass(frozen=True)
class GestureEmbodimentProfile:
    enabled: bool
    preset: str
    tilt_enabled: bool = True
    nodding_enabled: bool = True


@dataclass(frozen=True)
class EmbodimentProfile:
    gestures: GestureEmbodimentProfile = field(
        default_factory=lambda: GestureEmbodimentProfile(
            enabled=False,
            preset="auto",
            tilt_enabled=True,
            nodding_enabled=True,
        )
    )


@dataclass(frozen=True)
class ScenarioProfile:
    name: str
    source_path: Path
    framework_config: dict[str, Any] = field(repr=False)
    manifest_id: str = ""
    manifest: ProviderManifest | None = None
    resources: ResourceSelectionsProfile = field(default_factory=ResourceSelectionsProfile)
    robot: RobotProfile = field(default_factory=RobotProfile)
    robot_family: str = DEFAULT_ROBOT_FAMILY
    navigation: NavigationProfile = field(
        default_factory=lambda: NavigationProfile(None, ())
    )
    tools: ToolsProfile = field(default_factory=ToolsProfile)
    employee_directory: EmployeeDirectoryProfile = field(
        default_factory=lambda: EmployeeDirectoryProfile(
            enabled=False,
            site_code="",
        )
    )
    knowledge_bases: tuple[KnowledgeBaseProfile, ...] = ()
    face_recognition: FaceRecognitionProfile = field(
        default_factory=lambda: FaceRecognitionProfile(
            enabled=True,
            db_path=DEFAULT_FACE_DB_PATH,
            loop_interval_sec=1.0,
            recognition_threshold=0.6,
            depth_gate=FaceDepthGateProfile(
                enabled=False,
                sync_slop_sec=0.12,
                sync_queue_size=10,
                capture_timeout_sec=1.5,
                max_face_depth_m=2.0,
                min_valid_samples=2,
                patch_size=3,
                search_radius_px=12,
                max_valid_depth_m=10.0,
            ),
            owner_turn=FaceOwnerTurnProfile(
                enabled=False,
                camera_yaw_offset_rad=0.0,
                deadband_deg=3.0,
                turn_gain=1.0,
                max_turn_deg=25.0,
                angular_speed_rad_s=0.8,
                command_hz=50.0,
                delay_after_recording_sec=0.05,
                odom_frame="odom",
                robot_frame="base_link",
                yaw_tolerance_deg=1.5,
                max_duration_sec=1.5,
                slow_zone_deg=8.0,
                min_angular_speed_rad_s=0.25,
            ),
            preference_extraction=PreferenceExtractionProfile(enabled=False),
            proactive_greeting=ProactiveGreetingProfile(
                recognized_enabled=True,
                unknown_enabled=True,
                recognized_cooldown_sec=45.0,
                unknown_cooldown_sec=30.0,
            ),
        )
    )
    identity_store: IdentityStoreProfile = field(
        default_factory=lambda: IdentityStoreProfile(
            db_path=DEFAULT_IDENTITY_DB_PATH,
        )
    )
    memory_store: MemoryStoreProfile = field(
        default_factory=lambda: MemoryStoreProfile(
            db_path=str(DEFAULT_MEMORY_DB_PATH),
        )
    )
    slack_memory: SlackMemoryProfile = field(default_factory=SlackMemoryProfile)
    speaker_recognition: SpeakerRecognitionProfile = field(
        default_factory=lambda: SpeakerRecognitionProfile(
            enabled=True,
            policy=SpeakerRecognitionPolicy(
                backend="speechbrain_ecapa",
                db_path=DEFAULT_SPEAKER_DB_PATH,
                query_min_voiced_sec=0.8,
                query_match_threshold=0.60,
                query_margin_threshold=0.08,
                reference_update_threshold=0.55,
                enroll_min_voiced_sec=2.0,
                enroll_max_voiced_sec=0.0,
                enroll_min_rms_level=350.0,
                max_clipped_fraction=0.02,
                explicit_prompt_after_silent_failures=2,
            ),
        )
    )
    realtime: RealtimeProfile = field(
        default_factory=lambda: RealtimeProfile(
            prompt_file=None,
            model="gpt-realtime-1.5",
            voice="cedar",
            audio_output_speed=0.9,
            transcription_model="gpt-4o-mini-transcribe",
            input_device="pulse",
            output_device="pulse",
            noise_reduction="near_field",
            language=None,
            vad_threshold=0.8,
            silence_grace_period=0.3,
            wake_word="hey puffle",
            wake_word_model="hey puffle",
            wake_threshold=0.5,
            wake_window_sec=5.0,
            input_sample_rate=24000,
            output_sample_rate=24000,
            input_block_size=2400,
            max_output_tokens=None,
            admission=RealtimeAdmissionProfile(
                block_during_speaking=True,
                open_on_face_presence=True,
                open_on_interaction_states=("alert", "cooldown"),
                open_on_wake_window=True,
            ),
        )
    )
    engagement: EngagementProfile = field(
        default_factory=lambda: EngagementProfile(
            coalescer_debounce_sec=0.4,
            coalescer_max_wait_sec=2.0,
            alert_timeout_sec=15.0,
            cooldown_sec=7.0,
            speaking_timeout_sec=30.0,
            startup_patrol_delay_sec=2.0,
            patrol_next_hop_delay_sec=5.0,
        )
    )
    startup: StartupProfile = field(
        default_factory=lambda: StartupProfile(
            prepare_robot=False,
            service_timeout_sec=10.0,
            fail_on_prepare_error=True,
        )
    )
    battery: BatteryProfile = field(
        default_factory=lambda: BatteryProfile(
            enabled=True,
            low_battery_pct=30.0,
            charging_ready_pct=90.0,
        )
    )
    embodiment: EmbodimentProfile = field(default_factory=EmbodimentProfile)


def resolve_profile_path(name_or_path: str) -> Path:
    """Resolve a scenario profile selection to an on-disk YAML path."""
    raw = str(name_or_path or "").strip()
    if not raw:
        raise ProfileValidationError("Profile selection is required.")

    candidate = Path(raw)
    looks_like_path = (
        candidate.is_absolute()
        or candidate.parent != Path(".")
        or raw.startswith(".")
        or os.sep in raw
    )

    if looks_like_path:
        path = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        resolved = path.resolve()
        if not resolved.exists():
            raise ProfileValidationError(f"Profile file not found: {resolved}")
        return resolved

    filename = raw if raw.endswith((".yaml", ".yml")) else f"{raw}.yaml"
    resolved = (PROFILES_DIR / filename).resolve()
    if not resolved.exists():
        raise ProfileValidationError(
            f"Profile '{raw}' not found under {PROFILES_DIR}."
        )
    return resolved


def load_scenario_profile(
    selection: Optional[str] = None,
    *,
    require_explicit: bool = False,
) -> ScenarioProfile:
    """Load and validate an Argos scenario profile."""
    chosen = selection
    if not chosen:
        if require_explicit:
            raise ProfileValidationError(
                "Profile selection is required. Use --profile <name-or-path>."
            )
        chosen = DEFAULT_PROFILE_NAME

    profile_path = resolve_profile_path(chosen)
    framework_config = _load_framework_config()
    payload = _load_yaml(profile_path)
    return _parse_profile(
        payload,
        profile_path=profile_path,
        framework_config=framework_config,
    )


def resolve_prompt_file(prompt_file: Optional[str]) -> Optional[Path]:
    """Resolve a prompt path using the existing Argos prompt lookup rules."""
    if not prompt_file:
        return None
    from argos_src.resource_paths import resolve_prompt_path

    return resolve_prompt_path(prompt_file)


def resolve_locations_file(locations_file: Optional[str]) -> Optional[Path]:
    """Resolve a navigation locations file path."""
    if not locations_file:
        return None

    candidate = Path(locations_file)
    if candidate.is_absolute():
        return candidate
    if candidate.parent == Path("."):
        from argos_src.nav_support.locations import resolve_map_locations_path

        return resolve_map_locations_path(locations_file)
    return (REPO_ROOT / candidate).resolve()


def resolve_repo_path(path_value: str) -> Path:
    """Resolve a repo-relative or absolute path value."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_wake_word_model(wake_word: str) -> str:
    """Resolve a wake-word phrase or model path to the string expected by OpenWakeWord."""
    raw = str(wake_word or "").strip()
    if not raw:
        raise ProfileValidationError("realtime.wake_word must not be empty.")

    if _looks_like_wake_word_model_path(raw):
        resolved = resolve_repo_path(raw)
        if not resolved.exists():
            raise ProfileValidationError(f"Wake-word model file not found: {resolved}")
        if resolved.suffix.lower() != ".onnx":
            raise ProfileValidationError(
                f"Wake-word model must be an ONNX file: {resolved}"
            )
        return str(resolved)

    matches = _find_named_wake_word_models(raw)
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise ProfileValidationError(
            f"Multiple local wake-word models match '{raw}': {rendered}"
        )
    if matches:
        return str(matches[0])
    return raw


def apply_agent_cli_overrides(
    profile: ScenarioProfile,
    *,
    map_file: Optional[str] = None,
    startup_patrol_route: Optional[list[str]] = None,
    prompt_file: Optional[str] = None,
) -> ScenarioProfile:
    """Apply navigation/prompt CLI overrides to a loaded scenario profile."""
    updated_navigation = profile.navigation
    updated_realtime = profile.realtime

    if map_file is not None:
        updated_navigation = replace(updated_navigation, locations_file=map_file)
    if startup_patrol_route is not None:
        updated_navigation = replace(
            updated_navigation,
            startup_patrol_route=tuple(startup_patrol_route),
        )
    if prompt_file is not None:
        updated_realtime = replace(updated_realtime, prompt_file=prompt_file)

    return replace(profile, navigation=updated_navigation, realtime=updated_realtime)


def apply_audio_cli_overrides(
    profile: ScenarioProfile,
    *,
    map_file: Optional[str] = None,
    patrol_route: Optional[list[str]] = None,
    prompt_file: Optional[str] = None,
    wake_word: Optional[str] = None,
    wake_threshold: Optional[float] = None,
    wake_window_sec: Optional[float] = None,
    silence_grace_period: Optional[float] = None,
    speaker_channels: Optional[int] = None,
) -> ScenarioProfile:
    """Apply shell overrides for the realtime speech runtime."""
    del speaker_channels
    profile = apply_agent_cli_overrides(
        profile,
        map_file=map_file,
        startup_patrol_route=patrol_route,
        prompt_file=prompt_file,
    )

    updated_realtime = profile.realtime
    if wake_word is not None:
        updated_realtime = replace(
            updated_realtime,
            wake_word=wake_word,
            wake_word_model=resolve_wake_word_model(wake_word),
        )
    if wake_threshold is not None:
        updated_realtime = replace(
            updated_realtime,
            wake_threshold=float(wake_threshold),
        )
    if wake_window_sec is not None:
        updated_realtime = replace(
            updated_realtime,
            wake_window_sec=float(wake_window_sec),
        )
    if silence_grace_period is not None:
        updated_realtime = replace(
            updated_realtime,
            silence_grace_period=float(silence_grace_period),
        )
    return replace(profile, realtime=updated_realtime)


def _load_framework_config() -> dict[str, Any]:
    if not FRAMEWORK_CONFIG_PATH.exists():
        return {}
    with open(FRAMEWORK_CONFIG_PATH, "rb") as f:
        try:
            return dict(tomli.load(f))
        except TypeError:
            return dict(tomli.loads(f.read().decode("utf-8")))


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ProfileValidationError(
            f"Profile {path} must contain a top-level mapping."
        )
    return dict(data)


def _load_manifest_for_profile(manifest_id: str) -> ProviderManifest:
    if not manifest_id:
        raise ProfileValidationError("profile.manifest is required.")
    try:
        return load_provider_manifest(manifest_id)
    except ManifestValidationError as exc:
        raise ProfileValidationError(str(exc)) from exc


def _parse_resource_selections(
    data: dict[str, Any],
    *,
    manifest: ProviderManifest,
) -> ResourceSelectionsProfile:
    selections = ResourceSelectionsProfile(
        primary_robot=_pop_optional_str(data, "primary_robot", default="") or "",
        face_camera=_pop_optional_str(data, "face_camera", default="") or "",
        scene_camera=_pop_optional_str(data, "scene_camera", default="") or "",
        lidar=_pop_optional_str(data, "lidar", default="") or "",
    )
    _reject_unknown(data, "resources")

    primary_robot = selections.primary_robot
    if not primary_robot:
        primary_robot = _default_resource_id(manifest, kind="robot_base")
        selections = replace(selections, primary_robot=primary_robot)
    if not selections.scene_camera:
        scene_camera = _default_resource_id(manifest, capability_id="camera.rgb")
        selections = replace(selections, scene_camera=scene_camera)
    if not selections.face_camera:
        face_camera = selections.scene_camera or _default_resource_id(
            manifest,
            capability_id="camera.rgb",
        )
        selections = replace(selections, face_camera=face_camera)

    for field_name in ("primary_robot", "face_camera", "scene_camera", "lidar"):
        resource_id = str(getattr(selections, field_name, "") or "").strip()
        if not resource_id:
            continue
        if manifest.resource_by_id(resource_id) is None:
            raise ProfileValidationError(
                f"resources.{field_name} references unknown manifest resource "
                f"'{resource_id}'."
            )
    return selections


def _default_resource_id(
    manifest: ProviderManifest,
    *,
    kind: str | None = None,
    capability_id: str | None = None,
) -> str:
    for resource in manifest.resources:
        if kind is not None and resource.kind != kind:
            continue
        if capability_id is not None and capability_id not in resource.capabilities:
            continue
        return resource.id
    return ""


def _primary_robot_resource(
    manifest: ProviderManifest,
    resources: ResourceSelectionsProfile | None,
) -> ProviderResource:
    resource_id = str(getattr(resources, "primary_robot", "") or "").strip()
    if not resource_id:
        resource_id = _default_resource_id(manifest, kind="robot_base")
    resource = manifest.resource_by_id(resource_id)
    if resource is None:
        raise ProfileValidationError(
            f"manifest {manifest.id} does not define primary robot resource "
            f"'{resource_id or '<missing>'}'."
        )
    return resource


def _validate_tool_capabilities(
    enabled_tool_ids: tuple[str, ...],
    *,
    robot_family: str | None,
    manifest: ProviderManifest,
    resources: ResourceSelectionsProfile | None,
) -> None:
    primary_robot = _primary_robot_resource(manifest, resources)
    scene_camera = _selected_resource(
        manifest,
        getattr(resources, "scene_camera", "") if resources is not None else "",
    )
    face_camera = _selected_resource(
        manifest,
        getattr(resources, "face_camera", "") if resources is not None else "",
    )
    for tool_id in enabled_tool_ids:
        for capability_id in required_capability_ids_for_tool_id(
            tool_id,
            robot_family=robot_family,
        ):
            resource = _resource_for_capability(
                capability_id,
                primary_robot=primary_robot,
                scene_camera=scene_camera,
                face_camera=face_camera,
            )
            if resource is None or capability_id not in resource.capabilities:
                raise ProfileValidationError(
                    f"Tool '{tool_id}' requires manifest capability "
                    f"'{capability_id}', but no selected resource provides it."
                )


def _require_selected_capability(
    *,
    capability_id: str,
    resource: ProviderResource | None,
    selector_name: str,
    feature_name: str,
) -> None:
    if resource is None or capability_id not in resource.capabilities:
        raise ProfileValidationError(
            f"{feature_name} requires selected resource {selector_name} "
            f"to provide manifest capability '{capability_id}'."
        )


def _validate_runtime_resource_capabilities(
    *,
    manifest: ProviderManifest,
    resources: ResourceSelectionsProfile,
    face_recognition: FaceRecognitionProfile,
) -> None:
    primary_robot = _primary_robot_resource(manifest, resources)
    face_camera = _selected_resource(manifest, resources.face_camera)

    if face_recognition.enabled:
        _require_selected_capability(
            capability_id="camera.rgb",
            resource=face_camera,
            selector_name="resources.face_camera",
            feature_name="face_recognition.enabled",
        )
    if face_recognition.depth_gate.enabled:
        _require_selected_capability(
            capability_id="camera.rgbd",
            resource=face_camera,
            selector_name="resources.face_camera",
            feature_name="face_recognition.depth_gate.enabled",
        )
    if face_recognition.owner_turn.enabled:
        _require_selected_capability(
            capability_id="camera.intrinsics",
            resource=face_camera,
            selector_name="resources.face_camera",
            feature_name="face_recognition.owner_turn.enabled",
        )
        _require_selected_capability(
            capability_id="transform.lookup",
            resource=primary_robot,
            selector_name="resources.primary_robot",
            feature_name="face_recognition.owner_turn.enabled",
        )


def _selected_resource(
    manifest: ProviderManifest,
    resource_id: str,
) -> ProviderResource | None:
    rendered = str(resource_id or "").strip()
    if not rendered:
        return None
    return manifest.resource_by_id(rendered)


def _resource_for_capability(
    capability_id: str,
    *,
    primary_robot: ProviderResource,
    scene_camera: ProviderResource | None,
    face_camera: ProviderResource | None,
) -> ProviderResource | None:
    if capability_id.startswith("camera."):
        if scene_camera is not None and capability_id in scene_camera.capabilities:
            return scene_camera
        return face_camera
    return primary_robot


def _parse_profile(
    payload: dict[str, Any],
    *,
    profile_path: Path,
    framework_config: dict[str, Any],
) -> ScenarioProfile:
    profile_data = dict(payload)
    name = _pop_optional_str(profile_data, "name", default=profile_path.stem)
    manifest_id = _pop_optional_str(profile_data, "manifest", default=None) or ""
    manifest = _load_manifest_for_profile(manifest_id)
    resources_data = _pop_section(profile_data, "resources")
    resources = _parse_resource_selections(resources_data, manifest=manifest)
    if "robot" in profile_data:
        raise ProfileValidationError(
            "profile.robot is no longer supported; use manifest/resources."
        )
    if "robot_family" in profile_data:
        raise ProfileValidationError(
            "profile.robot_family is no longer supported; use manifest resource family."
        )
    robot = _parse_robot(manifest=manifest, resources=resources)
    robot_family = robot.family
    if robot_family not in SUPPORTED_ROBOT_FAMILIES:
        raise ProfileValidationError(
            "manifest primary robot resource family must be one of: "
            + ", ".join(SUPPORTED_ROBOT_FAMILIES)
        )

    legacy_agent_data = _pop_section(profile_data, "agent")
    if legacy_agent_data:
        raise ProfileValidationError(
            "Argos no longer uses an 'agent' config section. Move prompt_file to realtime.prompt_file."
        )

    navigation_data = _pop_section(profile_data, "navigation")
    tools_data = _pop_section(profile_data, "tools")
    employee_directory_data = _pop_section(profile_data, "employee_directory")
    knowledge_base_data = _pop_list(profile_data, "knowledge_bases", default=[])
    identity_data = _pop_section(profile_data, "identity_store")
    memory_data = _pop_section(profile_data, "memory_store")
    slack_memory_data = _pop_section(profile_data, "slack_memory")
    face_data = _pop_section(profile_data, "face_recognition")
    speaker_data = _pop_section(profile_data, "speaker_recognition")
    realtime_data = _pop_section(profile_data, "realtime")
    engagement_data = _pop_section(profile_data, "engagement")
    startup_data = _pop_section(profile_data, "startup")
    battery_data = _pop_section(profile_data, "battery")
    embodiment_data = _pop_section(profile_data, "embodiment")
    _reject_unknown(profile_data, "profile")

    navigation = _parse_navigation(navigation_data)
    tools = _parse_tools(
        tools_data,
        robot_family=robot_family,
        manifest=manifest,
        resources=resources,
    )
    employee_directory = _parse_employee_directory(employee_directory_data)
    knowledge_bases = tuple(
        _parse_knowledge_base_entry(item, index=i)
        for i, item in enumerate(knowledge_base_data)
    )
    face_recognition = _parse_face_recognition(face_data)
    identity_store = _parse_identity_store(identity_data)
    memory_store = _parse_memory_store(memory_data)
    slack_memory = _parse_slack_memory(slack_memory_data)
    speaker_recognition = _parse_speaker_recognition(speaker_data)
    realtime = _parse_realtime(
        realtime_data,
        face_profile=face_recognition,
    )
    _validate_runtime_resource_capabilities(
        manifest=manifest,
        resources=resources,
        face_recognition=face_recognition,
    )
    tools, employee_directory, face_recognition, realtime = _reconcile_profile_dependencies(
        robot_family=robot_family,
        tools=tools,
        employee_directory=employee_directory,
        face_recognition=face_recognition,
        realtime=realtime,
    )
    engagement = _parse_engagement(engagement_data)
    startup = _parse_startup(startup_data, robot_family=robot_family)
    battery = _parse_battery(battery_data)
    embodiment = _parse_embodiment(embodiment_data, robot_family=robot_family)

    if navigation.startup_patrol_route and not navigation.locations_file:
        raise ProfileValidationError(
            "navigation.startup_patrol_route requires navigation.locations_file."
        )

    return ScenarioProfile(
        name=name,
        source_path=profile_path,
        framework_config=framework_config,
        manifest_id=manifest_id,
        manifest=manifest,
        resources=resources,
        robot=robot,
        robot_family=robot_family,
        navigation=navigation,
        tools=tools,
        employee_directory=employee_directory,
        knowledge_bases=knowledge_bases,
        identity_store=identity_store,
        memory_store=memory_store,
        slack_memory=slack_memory,
        face_recognition=face_recognition,
        speaker_recognition=speaker_recognition,
        realtime=realtime,
        engagement=engagement,
        startup=startup,
        battery=battery,
        embodiment=embodiment,
    )


def _parse_robot(
    *,
    manifest: ProviderManifest,
    resources: ResourceSelectionsProfile,
) -> RobotProfile:
    resource = _primary_robot_resource(manifest, resources)
    provider = manifest.provider_by_id(resource.provider)
    if provider is None:
        raise ProfileValidationError(
            f"manifest {manifest.id} primary robot resource '{resource.id}' "
            f"references unknown provider '{resource.provider}'."
        )
    bridge = ProviderBindingProfile(
        transport=provider.transport,
        key_prefix=provider.key_prefix,
        connect_endpoints=provider.connect_endpoints or None,
        provider_id=provider.id,
        resource_id=resource.id,
    )
    return RobotProfile(
        id=manifest.id,
        family=(resource.family or DEFAULT_ROBOT_FAMILY).strip(),
        display_name=manifest.display_name,
        bridge=bridge,
    )


def _parse_navigation(data: dict[str, Any]) -> NavigationProfile:
    locations_file = _pop_optional_str(data, "locations_file", default=None)
    startup_patrol_route = tuple(
        _coerce_string_list(
            _pop_list(data, "startup_patrol_route", default=[]),
            context="navigation.startup_patrol_route",
        )
    )
    _reject_unknown(data, "navigation")
    return NavigationProfile(
        locations_file=locations_file,
        startup_patrol_route=startup_patrol_route,
    )


def _parse_tools(
    data: dict[str, Any],
    *,
    robot_family: str | None,
    manifest: ProviderManifest,
    resources: ResourceSelectionsProfile | None = None,
) -> ToolsProfile:
    raw_enabled = data.pop("enabled_tool_ids", None)
    if raw_enabled is None:
        raw_enabled = []
    enabled_tool_ids = tuple(
        _coerce_string_list(
            raw_enabled,
            context="tools.enabled_tool_ids",
        )
    )
    _reject_unknown(data, "tools")
    try:
        resolve_builtin_tool_names(enabled_tool_ids, robot_family=robot_family)
    except ValueError as exc:
        raise ProfileValidationError(str(exc)) from exc
    _validate_tool_capabilities(
        enabled_tool_ids,
        robot_family=robot_family,
        manifest=manifest,
        resources=resources,
    )
    return ToolsProfile(enabled_tool_ids=enabled_tool_ids)


def _parse_knowledge_base_entry(item: Any, *, index: int) -> KnowledgeBaseProfile:
    context = f"knowledge_bases[{index}]"
    if not isinstance(item, dict):
        raise ProfileValidationError(f"{context} must be a mapping.")
    data = dict(item)
    kind = _pop_required_str(data, "kind", context=context)
    root_dir = _pop_required_str(data, "root_dir", context=context)
    tool_name = _pop_required_str(data, "tool_name", context=context)
    description = _pop_required_str(data, "description", context=context)
    k = _pop_int(data, "k", default=4)
    _reject_unknown(data, context)
    return KnowledgeBaseProfile(
        kind=kind,
        root_dir=root_dir,
        tool_name=tool_name,
        description=description,
        k=k,
    )


def _parse_employee_directory(data: dict[str, Any]) -> EmployeeDirectoryProfile:
    enabled = _pop_bool(data, "enabled", default=False)
    site_code = (_pop_optional_str(data, "site_code", default="") or "").strip()
    _reject_unknown(data, "employee_directory")
    if enabled and not site_code:
        raise ProfileValidationError(
            "employee_directory.site_code is required when employee_directory.enabled is true."
        )
    return EmployeeDirectoryProfile(
        enabled=enabled,
        site_code=site_code,
    )


def _parse_face_recognition(data: dict[str, Any]) -> FaceRecognitionProfile:
    from argos_src.face_recognition.depth_gate import DepthGateSettings

    proactive_data = _pop_section(data, "proactive_greeting")
    depth_gate_data = _pop_section(data, "depth_gate")
    owner_turn_data = _pop_section(data, "owner_turn")
    preference_extraction_data = _pop_section(data, "preference_extraction")

    proactive = ProactiveGreetingProfile(
        recognized_enabled=_pop_bool(proactive_data, "recognized_enabled", default=True),
        unknown_enabled=_pop_bool(proactive_data, "unknown_enabled", default=True),
        recognized_cooldown_sec=_pop_float(
            proactive_data,
            "recognized_cooldown_sec",
            default=45.0,
        ),
        unknown_cooldown_sec=_pop_float(
            proactive_data,
            "unknown_cooldown_sec",
            default=30.0,
        ),
    )
    _reject_unknown(proactive_data, "face_recognition.proactive_greeting")

    depth_gate = FaceDepthGateProfile(
        enabled=_pop_bool(depth_gate_data, "enabled", default=False),
        sync_slop_sec=_pop_float(depth_gate_data, "sync_slop_sec", default=0.12),
        sync_queue_size=_pop_int(depth_gate_data, "sync_queue_size", default=10),
        capture_timeout_sec=_pop_float(
            depth_gate_data,
            "capture_timeout_sec",
            default=1.5,
        ),
        max_face_depth_m=_pop_float(
            depth_gate_data,
            "max_face_depth_m",
            default=2.0,
        ),
        min_valid_samples=_pop_int(depth_gate_data, "min_valid_samples", default=2),
        patch_size=_pop_int(depth_gate_data, "patch_size", default=3),
        search_radius_px=_pop_int(depth_gate_data, "search_radius_px", default=12),
        max_valid_depth_m=_pop_float(
            depth_gate_data,
            "max_valid_depth_m",
            default=10.0,
        ),
    )
    try:
        DepthGateSettings(
            sync_slop_sec=depth_gate.sync_slop_sec,
            sync_queue_size=depth_gate.sync_queue_size,
            capture_timeout_sec=depth_gate.capture_timeout_sec,
            max_face_depth_m=depth_gate.max_face_depth_m,
            min_valid_samples=depth_gate.min_valid_samples,
            patch_size=depth_gate.patch_size,
            search_radius_px=depth_gate.search_radius_px,
            max_valid_depth_m=depth_gate.max_valid_depth_m,
        )
    except ValueError as exc:
        raise ProfileValidationError(
            f"face_recognition.depth_gate invalid: {exc}"
        ) from exc
    _reject_unknown(depth_gate_data, "face_recognition.depth_gate")

    owner_turn = FaceOwnerTurnProfile(
        enabled=_pop_bool(owner_turn_data, "enabled", default=False),
        camera_yaw_offset_rad=_pop_float(
            owner_turn_data,
            "camera_yaw_offset_rad",
            default=0.0,
        ),
        deadband_deg=_pop_float(owner_turn_data, "deadband_deg", default=3.0),
        turn_gain=_pop_float(owner_turn_data, "turn_gain", default=1.0),
        max_turn_deg=_pop_float(owner_turn_data, "max_turn_deg", default=25.0),
        angular_speed_rad_s=_pop_float(
            owner_turn_data,
            "angular_speed_rad_s",
            default=0.8,
        ),
        command_hz=_pop_float(owner_turn_data, "command_hz", default=50.0),
        delay_after_recording_sec=_pop_float(
            owner_turn_data,
            "delay_after_recording_sec",
            default=0.05,
        ),
        odom_frame=_pop_optional_str(owner_turn_data, "odom_frame", default="odom")
        or "odom",
        robot_frame=_pop_optional_str(
            owner_turn_data,
            "robot_frame",
            default="base_link",
        )
        or "base_link",
        yaw_tolerance_deg=_pop_float(
            owner_turn_data,
            "yaw_tolerance_deg",
            default=1.5,
        ),
        max_duration_sec=_pop_float(
            owner_turn_data,
            "max_duration_sec",
            default=1.5,
        ),
        slow_zone_deg=_pop_float(owner_turn_data, "slow_zone_deg", default=8.0),
        min_angular_speed_rad_s=_pop_float(
            owner_turn_data,
            "min_angular_speed_rad_s",
            default=0.25,
        ),
    )
    if owner_turn.deadband_deg < 0.0:
        raise ProfileValidationError("face_recognition.owner_turn.deadband_deg must be >= 0")
    if owner_turn.turn_gain <= 0.0:
        raise ProfileValidationError("face_recognition.owner_turn.turn_gain must be > 0")
    if owner_turn.max_turn_deg <= 0.0:
        raise ProfileValidationError("face_recognition.owner_turn.max_turn_deg must be > 0")
    if owner_turn.angular_speed_rad_s <= 0.0:
        raise ProfileValidationError(
            "face_recognition.owner_turn.angular_speed_rad_s must be > 0"
        )
    if owner_turn.command_hz <= 0.0:
        raise ProfileValidationError("face_recognition.owner_turn.command_hz must be > 0")
    if owner_turn.delay_after_recording_sec < 0.0:
        raise ProfileValidationError(
            "face_recognition.owner_turn.delay_after_recording_sec must be >= 0"
        )
    if owner_turn.yaw_tolerance_deg < 0.0:
        raise ProfileValidationError(
            "face_recognition.owner_turn.yaw_tolerance_deg must be >= 0"
        )
    if owner_turn.max_duration_sec <= 0.0:
        raise ProfileValidationError(
            "face_recognition.owner_turn.max_duration_sec must be > 0"
        )
    if owner_turn.slow_zone_deg <= 0.0:
        raise ProfileValidationError("face_recognition.owner_turn.slow_zone_deg must be > 0")
    if owner_turn.min_angular_speed_rad_s <= 0.0:
        raise ProfileValidationError(
            "face_recognition.owner_turn.min_angular_speed_rad_s must be > 0"
        )
    if owner_turn.min_angular_speed_rad_s > owner_turn.angular_speed_rad_s:
        raise ProfileValidationError(
            "face_recognition.owner_turn.min_angular_speed_rad_s must be <= angular_speed_rad_s"
        )
    _reject_unknown(owner_turn_data, "face_recognition.owner_turn")

    preference_extraction = PreferenceExtractionProfile(
        enabled=_pop_bool(preference_extraction_data, "enabled", default=False)
    )
    _reject_unknown(
        preference_extraction_data,
        "face_recognition.preference_extraction",
    )

    profile = FaceRecognitionProfile(
        enabled=_pop_bool(data, "enabled", default=True),
        db_path=_pop_optional_str(
            data,
            "db_path",
            default=DEFAULT_FACE_DB_PATH,
        )
        or DEFAULT_FACE_DB_PATH,
        loop_interval_sec=_pop_float(data, "loop_interval_sec", default=1.0),
        recognition_threshold=_pop_float(
            data,
            "recognition_threshold",
            default=0.6,
        ),
        depth_gate=depth_gate,
        owner_turn=owner_turn,
        preference_extraction=preference_extraction,
        proactive_greeting=proactive,
    )
    profile = replace(profile, db_path=str(resolve_repo_path(profile.db_path)))
    _reject_unknown(data, "face_recognition")
    return profile


def _parse_identity_store(data: dict[str, Any]) -> IdentityStoreProfile:
    profile = IdentityStoreProfile(
        db_path=_pop_optional_str(
            data,
            "db_path",
            default=DEFAULT_IDENTITY_DB_PATH,
        )
        or DEFAULT_IDENTITY_DB_PATH,
    )
    profile = replace(profile, db_path=str(resolve_repo_path(profile.db_path)))
    _reject_unknown(data, "identity_store")
    return profile


def _parse_memory_store(data: dict[str, Any]) -> MemoryStoreProfile:
    profile = MemoryStoreProfile(
        db_path=_pop_optional_str(
            data,
            "db_path",
            default=str(DEFAULT_MEMORY_DB_PATH),
        )
        or str(DEFAULT_MEMORY_DB_PATH),
    )
    profile = replace(profile, db_path=str(resolve_repo_path(profile.db_path)))
    _reject_unknown(data, "memory_store")
    return profile


def _parse_slack_memory(data: dict[str, Any]) -> SlackMemoryProfile:
    channel_entries = _pop_list(data, "channels", default=[])
    channels: list[SlackMemoryChannelProfile] = []
    for index, raw in enumerate(channel_entries):
        if not isinstance(raw, dict):
            raise ProfileValidationError(
                f"slack_memory.channels[{index}] must be a mapping."
            )
        channel_data = dict(raw)
        context = f"slack_memory.channels[{index}]"
        name = _pop_required_str(channel_data, "name", context=context).strip()
        if not name:
            raise ProfileValidationError(f"{context}.name is required.")
        max_messages = _pop_int(
            channel_data,
            "max_messages_per_window",
            default=200,
        )
        if max_messages <= 0:
            raise ProfileValidationError(
                f"{context}.max_messages_per_window must be positive."
            )
        channels.append(
            SlackMemoryChannelProfile(
                name=name,
                channel_id=(
                    _pop_optional_str(channel_data, "channel_id", default="") or ""
                ).strip(),
                site_code=(
                    _pop_optional_str(channel_data, "site_code", default="") or ""
                ).strip(),
                person_memory_enabled=_pop_bool(
                    channel_data,
                    "person_memory_enabled",
                    default=True,
                ),
                site_memory_enabled=_pop_bool(
                    channel_data,
                    "site_memory_enabled",
                    default=True,
                ),
                include_threads=_pop_bool(channel_data, "include_threads", default=True),
                max_messages_per_window=max_messages,
            )
        )
        _reject_unknown(channel_data, context)

    poll_interval = _pop_float(data, "poll_interval_sec", default=1800.0)
    if poll_interval <= 0:
        raise ProfileValidationError("slack_memory.poll_interval_sec must be positive.")
    lookback_minutes = _pop_int(data, "lookback_minutes", default=30)
    if lookback_minutes <= 0:
        raise ProfileValidationError("slack_memory.lookback_minutes must be positive.")
    profile = SlackMemoryProfile(
        enabled=_pop_bool(data, "enabled", default=False),
        start_with_agent=_pop_bool(data, "start_with_agent", default=False),
        bot_token_env=(
            _pop_optional_str(data, "bot_token_env", default="SLACK_BOT_TOKEN")
            or "SLACK_BOT_TOKEN"
        ).strip()
        or "SLACK_BOT_TOKEN",
        poll_interval_sec=poll_interval,
        lookback_minutes=lookback_minutes,
        channels=tuple(channels),
    )
    _reject_unknown(data, "slack_memory")
    return profile


def _parse_speaker_recognition(data: dict[str, Any]) -> SpeakerRecognitionProfile:
    enabled = _pop_bool(data, "enabled", default=True)
    try:
        policy = SpeakerRecognitionPolicy(
            backend="speechbrain_ecapa",
            db_path=str(resolve_repo_path(DEFAULT_SPEAKER_DB_PATH)),
            query_min_voiced_sec=_pop_float(data, "query_min_voiced_sec", default=0.8),
            query_match_threshold=_pop_float(data, "query_match_threshold", default=0.60),
            query_margin_threshold=_pop_float(data, "query_margin_threshold", default=0.08),
            reference_update_threshold=_pop_float(
                data,
                "reference_update_threshold",
                default=0.55,
            ),
            enroll_min_voiced_sec=_pop_float(data, "enroll_min_voiced_sec", default=2.0),
            enroll_max_voiced_sec=_pop_float(data, "enroll_max_voiced_sec", default=0.0),
            enroll_min_rms_level=_pop_float(data, "enroll_min_rms_level", default=350.0),
            max_clipped_fraction=_pop_float(data, "max_clipped_fraction", default=0.02),
            explicit_prompt_after_silent_failures=_pop_int(
                data,
                "explicit_prompt_after_silent_failures",
                default=2,
            ),
        )
    except ValueError as exc:
        raise ProfileValidationError(
            f"speaker_recognition invalid: {exc}"
        ) from exc
    _reject_unknown(data, "speaker_recognition")
    return SpeakerRecognitionProfile(enabled=enabled, policy=policy)


def _parse_realtime(
    data: dict[str, Any],
    *,
    face_profile: FaceRecognitionProfile,
) -> RealtimeProfile:
    admission_data = _pop_section(data, "admission")
    admission = RealtimeAdmissionProfile(
        block_during_speaking=_pop_bool(
            admission_data,
            "block_during_speaking",
            default=True,
        ),
        open_on_face_presence=_pop_bool(
            admission_data,
            "open_on_face_presence",
            default=True,
        ),
        open_on_interaction_states=tuple(
            _coerce_string_list(
                _pop_list(
                    admission_data,
                    "open_on_interaction_states",
                    default=["alert", "cooldown"],
                ),
                context="realtime.admission.open_on_interaction_states",
            )
        ),
        open_on_wake_window=_pop_bool(
            admission_data,
            "open_on_wake_window",
            default=True,
        ),
    )
    _reject_unknown(admission_data, "realtime.admission")

    prompt_file = _pop_optional_str(data, "prompt_file", default=None)
    model = _pop_optional_str(data, "model", default="gpt-realtime-1.5") or "gpt-realtime-1.5"
    voice = _pop_optional_str(data, "voice", default="cedar") or "cedar"
    audio_output_speed = _pop_float(data, "audio_output_speed", default=0.9)
    transcription_model = _pop_optional_str(
        data,
        "transcription_model",
        default="gpt-4o-mini-transcribe",
    )
    input_device = _pop_optional_str(
        data,
        "input_device",
        default="pulse",
    )
    output_device = _pop_optional_str(
        data,
        "output_device",
        default="pulse",
    )
    noise_reduction = _pop_optional_str(
        data,
        "noise_reduction",
        default="near_field",
    )
    language = _pop_optional_str(
        data,
        "language",
        default=None,
    )
    vad_threshold = _pop_float(
        data,
        "vad_threshold",
        default=0.8,
    )
    silence_grace_period = _pop_float(
        data,
        "silence_grace_period",
        default=0.3,
    )
    wake_word = _pop_optional_str(data, "wake_word", default="hey puffle") or "hey puffle"
    wake_word_model = _pop_optional_str(data, "wake_word_model", default=None)
    if wake_word_model:
        wake_word_model = resolve_wake_word_model(wake_word_model)
    else:
        wake_word_model = resolve_wake_word_model(wake_word)
    wake_threshold = _pop_float(
        data,
        "wake_threshold",
        default=0.5,
    )
    wake_window_sec = _pop_float(data, "wake_window_sec", default=5.0)
    input_sample_rate = _pop_int(data, "input_sample_rate", default=24000)
    output_sample_rate = _pop_int(data, "output_sample_rate", default=24000)
    input_block_size = _pop_int(data, "input_block_size", default=2400)
    max_output_tokens = _pop_optional_int(data, "max_output_tokens", default=None)
    _reject_unknown(data, "realtime")

    return RealtimeProfile(
        prompt_file=prompt_file,
        model=model,
        voice=voice,
        audio_output_speed=audio_output_speed,
        transcription_model=transcription_model,
        input_device=input_device,
        output_device=output_device,
        noise_reduction=noise_reduction,
        language=language,
        vad_threshold=vad_threshold,
        silence_grace_period=silence_grace_period,
        wake_word=wake_word,
        wake_word_model=wake_word_model,
        wake_threshold=wake_threshold,
        wake_window_sec=wake_window_sec,
        input_sample_rate=input_sample_rate,
        output_sample_rate=output_sample_rate,
        input_block_size=input_block_size,
        max_output_tokens=max_output_tokens,
        admission=admission,
    )


def _reconcile_profile_dependencies(
    *,
    robot_family: str,
    tools: ToolsProfile,
    employee_directory: EmployeeDirectoryProfile,
    face_recognition: FaceRecognitionProfile,
    realtime: RealtimeProfile,
) -> tuple[
    ToolsProfile,
    EmployeeDirectoryProfile,
    FaceRecognitionProfile,
    RealtimeProfile,
]:
    """Collapse dependent knobs so the loaded profile matches runtime behavior."""
    if face_recognition.enabled and employee_directory.enabled:
        return tools, employee_directory, face_recognition, realtime

    effective_tool_ids = list(tools.enabled_tool_ids)
    if not employee_directory.enabled:
        effective_tool_ids = [
            name
            for name in effective_tool_ids
            if resolve_builtin_tool_name(name, robot_family=robot_family)
            != "resolve_employee_identity"
        ]

    if not face_recognition.enabled:
        effective_tool_ids = [
            name
            for name in effective_tool_ids
            if resolve_builtin_tool_name(name, robot_family=robot_family)
            != "enroll_visible_person"
        ]
        proactive = face_recognition.proactive_greeting
        if proactive.recognized_enabled or proactive.unknown_enabled:
            face_recognition = replace(
                face_recognition,
                proactive_greeting=replace(
                    proactive,
                    recognized_enabled=False,
                    unknown_enabled=False,
                ),
            )
        if face_recognition.preference_extraction.enabled:
            face_recognition = replace(
                face_recognition,
                preference_extraction=PreferenceExtractionProfile(enabled=False),
            )
        if realtime.admission.open_on_face_presence:
            realtime = replace(
                realtime,
                admission=replace(
                    realtime.admission,
                    open_on_face_presence=False,
                ),
            )

    effective_tool_ids_tuple = tuple(effective_tool_ids)
    if effective_tool_ids_tuple != tools.enabled_tool_ids:
        tools = replace(tools, enabled_tool_ids=effective_tool_ids_tuple)
    return tools, employee_directory, face_recognition, realtime


def _looks_like_wake_word_model_path(value: str) -> bool:
    candidate = Path(value)
    return (
        candidate.suffix.lower() == ".onnx"
        or candidate.is_absolute()
        or candidate.parent != Path(".")
        or value.startswith(".")
        or os.sep in value
    )


def _find_named_wake_word_models(wake_word: str) -> list[Path]:
    if not ARGOS_MODELS_DIR.exists():
        return []

    target_key = _normalize_wake_word_key(wake_word)
    if not target_key:
        return []

    matches: list[Path] = []
    for model_path in ARGOS_MODELS_DIR.rglob("*.onnx"):
        if _normalize_wake_word_key(model_path.stem) == target_key:
            matches.append(model_path.resolve())
    return sorted(set(matches))


def _normalize_wake_word_key(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else " "
        for character in str(value or "")
    )
    return " ".join(normalized.split())


def _parse_engagement(data: dict[str, Any]) -> EngagementProfile:
    profile = EngagementProfile(
        coalescer_debounce_sec=_pop_float(
            data,
            "coalescer_debounce_sec",
            default=0.4,
        ),
        coalescer_max_wait_sec=_pop_float(
            data,
            "coalescer_max_wait_sec",
            default=2.0,
        ),
        alert_timeout_sec=_pop_float(data, "alert_timeout_sec", default=15.0),
        cooldown_sec=_pop_float(data, "cooldown_sec", default=7.0),
        speaking_timeout_sec=_pop_float(
            data,
            "speaking_timeout_sec",
            default=30.0,
        ),
        startup_patrol_delay_sec=_pop_float(
            data,
            "startup_patrol_delay_sec",
            default=2.0,
        ),
        patrol_next_hop_delay_sec=_pop_float(
            data,
            "patrol_next_hop_delay_sec",
            default=5.0,
        ),
    )
    _reject_unknown(data, "engagement")
    return profile


def _parse_startup(data: dict[str, Any], *, robot_family: str) -> StartupProfile:
    profile = StartupProfile(
        prepare_robot=_pop_bool(
            data,
            "prepare_robot",
            default=(robot_family == "spot"),
        ),
        service_timeout_sec=_pop_float(data, "service_timeout_sec", default=10.0),
        fail_on_prepare_error=_pop_bool(
            data,
            "fail_on_prepare_error",
            default=True,
        ),
    )
    _reject_unknown(data, "startup")
    return profile


def _parse_battery(data: dict[str, Any]) -> BatteryProfile:
    profile = BatteryProfile(
        enabled=_pop_bool(data, "enabled", default=True),
        low_battery_pct=_pop_float(data, "low_battery_pct", default=30.0),
        charging_ready_pct=_pop_float(data, "charging_ready_pct", default=90.0),
    )
    _reject_unknown(data, "battery")
    return profile


def _parse_embodiment(
    data: dict[str, Any],
    *,
    robot_family: str,
) -> EmbodimentProfile:
    gestures_data = _pop_section(data, "gestures")
    gestures = GestureEmbodimentProfile(
        enabled=_pop_bool(gestures_data, "enabled", default=False),
        preset=_pop_optional_str(gestures_data, "preset", default="auto") or "auto",
        tilt_enabled=_pop_bool(gestures_data, "tilt_enabled", default=True),
        nodding_enabled=_pop_bool(gestures_data, "nodding_enabled", default=True),
    )
    try:
        resolve_gesture_preset_name(
            robot_family=robot_family,
            preset=gestures.preset,
        )
    except ValueError as exc:
        raise ProfileValidationError(str(exc)) from exc
    _reject_unknown(gestures_data, "embodiment.gestures")
    _reject_unknown(data, "embodiment")
    return EmbodimentProfile(gestures=gestures)


def _pop_section(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.pop(key, None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProfileValidationError(f"{key} must be a mapping.")
    return dict(value)


def _pop_list(
    mapping: dict[str, Any],
    key: str,
    *,
    default: Optional[list[Any]] = None,
) -> list[Any]:
    value = mapping.pop(key, default)
    if value is None:
        return [] if default is None else list(default)
    if not isinstance(value, list):
        raise ProfileValidationError(f"{key} must be a list.")
    return list(value)


def _pop_optional_str(
    mapping: dict[str, Any],
    key: str,
    *,
    default: Optional[str],
) -> Optional[str]:
    value = mapping.pop(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProfileValidationError(f"{key} must be a string.")
    return value


def _pop_required_str(mapping: dict[str, Any], key: str, *, context: str) -> str:
    value = mapping.pop(key, None)
    if value is None:
        raise ProfileValidationError(f"{context}.{key} is required.")
    if not isinstance(value, str):
        raise ProfileValidationError(f"{context}.{key} must be a string.")
    return value


def _pop_float(mapping: dict[str, Any], key: str, *, default: float) -> float:
    value = mapping.pop(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileValidationError(f"{key} must be a number.")
    return float(value)


def _pop_int(mapping: dict[str, Any], key: str, *, default: int) -> int:
    value = mapping.pop(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileValidationError(f"{key} must be an integer.")
    return int(value)


def _pop_optional_int(
    mapping: dict[str, Any],
    key: str,
    *,
    default: Optional[int],
) -> Optional[int]:
    value = mapping.pop(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileValidationError(f"{key} must be an integer or null.")
    return int(value)


def _pop_bool(mapping: dict[str, Any], key: str, *, default: bool) -> bool:
    value = mapping.pop(key, default)
    if not isinstance(value, bool):
        raise ProfileValidationError(f"{key} must be a boolean.")
    return value


def _coerce_string_list(values: list[Any], *, context: str) -> list[str]:
    out: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ProfileValidationError(f"{context} entries must be strings.")
        stripped = value.strip()
        if stripped:
            out.append(stripped)
    return out


def _reject_unknown(mapping: dict[str, Any], context: str) -> None:
    if not mapping:
        return
    unknown = ", ".join(sorted(mapping))
    raise ProfileValidationError(f"Unknown keys in {context}: {unknown}")
