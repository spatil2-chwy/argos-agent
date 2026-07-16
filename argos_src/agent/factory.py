"""Factory function to create and wire the Argos realtime companion agent."""

from __future__ import annotations

import atexit
import logging
from typing import Optional

from argos_src.agent.gesture_runtime import (
    GESTURE_STATE_IDLE,
    GESTURE_STATE_LISTENING,
    GestureRuntime,
    resolve_gesture_preset_name,
)
from argos_src.agent.agent_runtime import RealtimeRobotAgent
from argos_src.profile_config import (
    DEFAULT_ROBOT_FAMILY,
    ScenarioProfile,
    apply_agent_cli_overrides,
    load_scenario_profile,
    resolve_locations_file,
    resolve_prompt_file,
)
from argos_src.resource_paths import load_system_prompt
from argos_src.runtime.battery_state import BatteryStateCache
from argos_src.provider_api.client import ProviderClient
from argos_src.provider_api.factory import create_provider_client
from argos_src.nav_support.locations import LocationStore, NavigationState
from argos_src.tools import (
    MEMORY_TOOL_NAMES,
    NAVIGATION_TOOL_NAMES,
    build_builtin_tools,
    build_knowledge_tools,
    resolve_builtin_tool_name,
    resolve_builtin_tool_names,
)

from .bridges import FaceEventBridge, PatrolLoopBridge
from .control.coalescer import EventCoalescer
from .control.engagement_runtime import EngagementStateMachine
from .factory_wiring import FactoryRuntimeWireup
from argos_src.observability.state_observer import StructuredStateObserver
from .startup import derive_initial_robot_posture, prepare_robot_for_agent_session


logger = logging.getLogger(__name__)

ROBOT_LABEL_BY_FAMILY = {
    "spot": "Spot",
    DEFAULT_ROBOT_FAMILY: "Go2",
}
NODE_PREFIX_BY_FAMILY = {
    "spot": "spot",
    DEFAULT_ROBOT_FAMILY: "go2_quadruped",
}
STAND_TOOL_BY_FAMILY = {
    "spot": "spot_stand",
    DEFAULT_ROBOT_FAMILY: "go2_balance_stand",
}


def _robot_label(robot_family: str) -> str:
    return ROBOT_LABEL_BY_FAMILY.get(robot_family, robot_family.replace("_", " ").title())


def _node_prefix(robot_family: str) -> str:
    return NODE_PREFIX_BY_FAMILY.get(robot_family, robot_family)


def _stand_tool_name(robot_family: str) -> str:
    return STAND_TOOL_BY_FAMILY.get(robot_family, "move_robot")


def _has_navigation_tools(resolved_tool_names: tuple[str, ...]) -> bool:
    return any(name in NAVIGATION_TOOL_NAMES for name in resolved_tool_names)


def _derive_runtime_flags(
    scenario_profile: ScenarioProfile,
    resolved_tool_names: tuple[str, ...],
) -> dict[str, bool]:
    navigation_enabled = _has_navigation_tools(resolved_tool_names)
    return {
        "navigation_enabled": navigation_enabled,
        "needs_navigation_state": (
            navigation_enabled
            or scenario_profile.face_recognition.enabled
            or scenario_profile.battery.enabled
            or bool(scenario_profile.navigation.startup_patrol_route)
        ),
        "needs_face_runtime": scenario_profile.face_recognition.enabled
        or ("capture_scene" in resolved_tool_names),
        "battery_enabled": bool(scenario_profile.battery.enabled),
        "self_charge_available": bool(scenario_profile.battery.enabled)
        and ("charging_dock" in resolved_tool_names),
    }


def _create_gesture_runtime(
    *,
    scenario_profile: ScenarioProfile,
    robot_client: ProviderClient,
    engagement: EngagementStateMachine,
) -> Optional[GestureRuntime]:
    gesture_profile = scenario_profile.embodiment.gestures
    if not gesture_profile.enabled:
        return None
    preset_name = resolve_gesture_preset_name(
        robot_family=scenario_profile.robot_family,
        preset=gesture_profile.preset,
    )
    if not preset_name:
        logger.info(
            "Embodied gestures enabled but no preset is available for robot_family=%s.",
            scenario_profile.robot_family,
        )
        return None
    enabled_states = []
    if gesture_profile.tilt_enabled:
        enabled_states.append(GESTURE_STATE_IDLE)
    if gesture_profile.nodding_enabled:
        enabled_states.append(GESTURE_STATE_LISTENING)
    if not enabled_states:
        logger.info("Embodied gestures enabled but all gesture states are disabled.")
        return None
    logger.info(
        "Embodied gestures enabled preset=%s robot_family=%s states=%s",
        preset_name,
        scenario_profile.robot_family,
        ",".join(enabled_states),
    )
    return GestureRuntime(
        connector=robot_client,
        engagement=engagement,
        preset_name=preset_name,
        enabled_states=tuple(enabled_states),
    )


def _create_display_runtime(*, scenario_profile: ScenarioProfile):
    if not bool(getattr(scenario_profile.display, "enabled", True)):
        return None
    resource_id = str(
        getattr(scenario_profile.resources, "interaction_display", "") or ""
    ).strip()
    manifest = scenario_profile.manifest
    if not resource_id or manifest is None:
        return None
    resource = manifest.resource_by_id(resource_id)
    if resource is None or not resource.has_capability("display.command"):
        return None
    provider = manifest.provider_by_id(resource.provider)
    if provider is None:
        return None
    from argos_src.display import DisplayRuntime

    display_client = create_provider_client(
        transport=provider.transport,
        key_prefix=provider.key_prefix,
        connect_endpoints=provider.connect_endpoints,
        resource_id=resource.id,
        manifest=manifest,
        auth_token_env=(
            getattr(provider.auth, "token_env", "") if provider.auth is not None else ""
        ),
    )
    return DisplayRuntime(client=display_client, resource_id=resource.id)


def _create_identity_memory_provider_client(*, scenario_profile: ScenarioProfile):
    resource_id = str(
        getattr(scenario_profile.resources, "identity_memory", "") or ""
    ).strip()
    manifest = scenario_profile.manifest
    if not resource_id or manifest is None:
        raise ValueError("identity_memory.enabled requires resources.identity_memory.")
    resource = manifest.resource_by_id(resource_id)
    if resource is None or not resource.has_capability("memory.identity"):
        raise ValueError(
            "identity_memory.enabled requires selected resources.identity_memory "
            "to provide memory.identity."
        )
    provider = manifest.provider_by_id(resource.provider)
    if provider is None:
        raise ValueError(
            f"Memory resource '{resource.id}' references unknown provider "
            f"'{resource.provider}'."
        )
    client = create_provider_client(
        transport=provider.transport,
        key_prefix=provider.key_prefix,
        connect_endpoints=provider.connect_endpoints,
        resource_id=resource.id,
        manifest=manifest,
        auth_token_env=(
            getattr(provider.auth, "token_env", "") if provider.auth is not None else ""
        ),
    )
    return client, resource.id


def _resolve_agent_profile(
    scenario_profile: Optional[ScenarioProfile],
    *,
    map_locations_file: Optional[str],
    startup_patrol_route: Optional[list[str]],
    prompt_file: Optional[str],
) -> ScenarioProfile:
    if scenario_profile is None:
        return apply_agent_cli_overrides(
            load_scenario_profile(),
            map_file=map_locations_file,
            startup_patrol_route=startup_patrol_route,
            prompt_file=prompt_file,
        )
    if (
        map_locations_file is not None
        or startup_patrol_route is not None
        or prompt_file is not None
    ):
        return apply_agent_cli_overrides(
            scenario_profile,
            map_file=map_locations_file,
            startup_patrol_route=startup_patrol_route,
            prompt_file=prompt_file,
        )
    return scenario_profile


def _attach_engagement_wiring(
    engagement: EngagementStateMachine,
    coalescer: EventCoalescer,
) -> None:
    attach_coalescer = getattr(engagement, "attach_coalescer", None)
    if callable(attach_coalescer):
        attach_coalescer(coalescer)
    else:
        engagement._coalescer = coalescer

    attach_battery_low_submitter = getattr(
        engagement,
        "attach_battery_low_submitter",
        None,
    )
    if callable(attach_battery_low_submitter):
        attach_battery_low_submitter(coalescer.submit)
    else:
        engagement._battery_low_submit = lambda text, meta: coalescer.submit(text, meta)


def create_agent(
    scenario_profile: Optional[ScenarioProfile] = None,
    *,
    map_locations_file: Optional[str] = None,
    startup_patrol_route: Optional[list[str]] = None,
    prompt_file: Optional[str] = None,
    raw_data_capture: object | None = None,
) -> RealtimeRobotAgent:
    """Create the profile-driven realtime Argos agent runtime."""
    scenario_profile = _resolve_agent_profile(
        scenario_profile,
        map_locations_file=map_locations_file,
        startup_patrol_route=startup_patrol_route,
        prompt_file=prompt_file,
    )

    robot_family = scenario_profile.robot_family
    node_prefix = _node_prefix(robot_family)
    robot_label = scenario_profile.robot.display_name or _robot_label(robot_family)

    robot_client = create_provider_client(
        transport=scenario_profile.robot.bridge.transport,
        key_prefix=scenario_profile.robot.bridge.key_prefix,
        connect_endpoints=scenario_profile.robot.bridge.connect_endpoints,
        resource_id=scenario_profile.robot.bridge.resource_id,
        manifest=scenario_profile.manifest,
    )
    startup_steps = prepare_robot_for_agent_session(
        robot_client,
        scenario_profile=scenario_profile,
    )
    initial_robot_posture = derive_initial_robot_posture(
        scenario_profile=scenario_profile,
        startup_steps=startup_steps,
    )
    if startup_steps:
        succeeded = sum(1 for step in startup_steps if step.get("success"))
        logger.info(
            "%s startup preparation complete: %s/%s steps succeeded.",
            robot_label,
            succeeded,
            len(startup_steps),
        )

    enabled_tool_ids = scenario_profile.tools.enabled_tool_ids
    if not scenario_profile.face_recognition.enabled:
        enabled_tool_ids = tuple(
            name
            for name in enabled_tool_ids
            if resolve_builtin_tool_name(name, robot_family=robot_family)
            != "enroll_visible_person"
        )
    resolved_tool_names = resolve_builtin_tool_names(
        enabled_tool_ids,
        robot_family=robot_family,
    )
    runtime_flags = _derive_runtime_flags(scenario_profile, resolved_tool_names)
    navigation_enabled = runtime_flags["navigation_enabled"]
    needs_navigation_state = runtime_flags["needs_navigation_state"]
    needs_face_runtime = runtime_flags["needs_face_runtime"]
    self_charge_available = runtime_flags["self_charge_available"]
    memory_tools_enabled = any(name in MEMORY_TOOL_NAMES for name in resolved_tool_names)
    display_runtime = _create_display_runtime(scenario_profile=scenario_profile)

    face_service = None
    speaker_service = None
    preference_extractor = None
    identity_memory_client = None
    memory_provider = None
    memory_context_compiler = None
    adaptive_update_coordinator = None
    if scenario_profile.identity_memory.enabled and (
        needs_face_runtime
        or scenario_profile.identity_memory.record_live_episodes
        or memory_tools_enabled
        or scenario_profile.speaker_recognition.enabled
    ):
        if scenario_profile.identity_memory.backend == "noop":
            from argos_src.identity_memory import NoopIdentityMemoryClient

            identity_memory_client = NoopIdentityMemoryClient()
        else:
            from argos_src.identity_memory import TailwagHttpIdentityMemoryClient

            memory_provider_client, memory_resource_id = (
                _create_identity_memory_provider_client(
                    scenario_profile=scenario_profile,
                )
            )
            identity_memory_client = TailwagHttpIdentityMemoryClient(
                provider_client=memory_provider_client,
                resource_id=memory_resource_id,
                site_code=scenario_profile.identity_memory.site_code,
                place_room_id=scenario_profile.identity_memory.place_room_id,
                retention_class=scenario_profile.identity_memory.retention_class,
                extract_live_turn_memory=(
                    scenario_profile.identity_memory.extract_live_turn_memory
                ),
            )
        memory_provider = identity_memory_client
        memory_context_compiler = identity_memory_client
        try:
            from argos_src.identity_memory import AdaptiveBiometricUpdateCoordinator

            adaptive_update_coordinator = AdaptiveBiometricUpdateCoordinator(
                identity_memory_client,
                logger_=logger,
            )
        except ImportError:
            logger.debug("Adaptive biometric update coordinator unavailable", exc_info=True)

    if needs_face_runtime:
        from argos_src.face_recognition.attention_gate import (
            AttentionGateSettings,
        )
        from argos_src.face_recognition.depth_gate import DepthGateSettings
        from argos_src.face_recognition.face_recognition_service import (
            FaceEnrollmentPolicy,
            FaceRecognitionStabilitySettings,
            FaceRecognitionService,
        )
        attention_gate = scenario_profile.face_recognition.attention_gate
        enrollment_policy = scenario_profile.face_recognition.enrollment_policy
        recognition_stability = scenario_profile.face_recognition.recognition_stability

        face_service = FaceRecognitionService(
            robot_client=robot_client,
            identity_memory_client=identity_memory_client,
            memory_store=identity_memory_client,
            site_code=scenario_profile.identity_memory.site_code,
            camera_resource_id=scenario_profile.resources.face_camera,
            camera_yaw_offset_rad=(
                scenario_profile.face_recognition.owner_turn.camera_yaw_offset_rad
            ),
            display_runtime=display_runtime,
            live_image_enabled=scenario_profile.face_recognition.live_image_enabled,
            depth_gate_settings=(
                DepthGateSettings(
                    sync_slop_sec=scenario_profile.face_recognition.depth_gate.sync_slop_sec,
                    sync_queue_size=scenario_profile.face_recognition.depth_gate.sync_queue_size,
                    capture_timeout_sec=scenario_profile.face_recognition.depth_gate.capture_timeout_sec,
                    max_face_depth_m=scenario_profile.face_recognition.depth_gate.max_face_depth_m,
                    min_valid_samples=scenario_profile.face_recognition.depth_gate.min_valid_samples,
                    patch_size=scenario_profile.face_recognition.depth_gate.patch_size,
                    search_radius_px=scenario_profile.face_recognition.depth_gate.search_radius_px,
                    max_valid_depth_m=scenario_profile.face_recognition.depth_gate.max_valid_depth_m,
                )
                if scenario_profile.face_recognition.depth_gate.enabled
                else None
            ),
            attention_gate_settings=AttentionGateSettings(
                enabled=attention_gate.enabled,
                min_face_area=attention_gate.min_face_area,
                max_abs_yaw_deg=attention_gate.max_abs_yaw_deg,
                max_abs_pitch_deg=attention_gate.max_abs_pitch_deg,
                max_abs_roll_deg=attention_gate.max_abs_roll_deg,
                min_abs_pitch_deg=attention_gate.min_abs_pitch_deg,
            ),
            enrollment_policy=FaceEnrollmentPolicy(
                min_face_area=enrollment_policy.min_face_area,
                min_brightness=enrollment_policy.min_brightness,
                max_brightness=enrollment_policy.max_brightness,
                min_contrast=enrollment_policy.min_contrast,
                min_embedding_similarity=enrollment_policy.min_embedding_similarity,
            ),
            recognition_stability_settings=FaceRecognitionStabilitySettings(
                window_frames=recognition_stability.window_frames,
                min_hits=recognition_stability.min_hits,
            ),
        )
        if (
            scenario_profile.identity_memory.record_live_episodes
            and memory_provider is not None
        ):
            preference_extractor = memory_provider
        if scenario_profile.face_recognition.enabled:
            face_service.start_loop(
                camera_resource_id=scenario_profile.resources.face_camera,
                interval=scenario_profile.face_recognition.loop_interval_sec,
            )

    if scenario_profile.speaker_recognition.enabled:
        from argos_src.speaker_recognition.service import SpeakerRecognitionService

        speaker_service = SpeakerRecognitionService(
            policy=scenario_profile.speaker_recognition.policy,
            identity_memory_client=identity_memory_client,
            adaptive_update_coordinator=adaptive_update_coordinator,
        )
        try:
            speaker_service.prewarm()
        except Exception:
            logger.warning(
                "Speaker backend prewarm failed; live turns will fall back to lazy initialization.",
                exc_info=True,
            )

    navigation_runtime_store = None
    location_store_for_prompt = None
    nav_state = None
    if needs_navigation_state:
        locations_path = resolve_locations_file(scenario_profile.navigation.locations_file)
        navigation_runtime_store = LocationStore(path=locations_path)
        nav_state = NavigationState(navigation_runtime_store)
        if navigation_enabled:
            location_store_for_prompt = navigation_runtime_store
    wiring = FactoryRuntimeWireup(
        robot_client=robot_client,
        nav_state=nav_state,
        format_navigation_event=_format_navigation_event,
    )

    battery_cache = None
    if runtime_flags["battery_enabled"]:
        battery_cache = BatteryStateCache(
            robot_client,
            low_battery_pct=scenario_profile.battery.low_battery_pct,
            charging_ready_pct=scenario_profile.battery.charging_ready_pct,
            self_charge_available=self_charge_available,
            on_charging_ready=wiring.notify_charging_ready,
        )
        wiring.bind_battery_cache(battery_cache)

    tools = build_builtin_tools(
        robot_family=robot_family,
        enabled_tool_ids=enabled_tool_ids,
        robot_client=robot_client,
        face_service=face_service,
        identity_memory_client=identity_memory_client,
        location_store=navigation_runtime_store,
        nav_state=nav_state,
        on_nav_event=wiring.submit_nav_event,
        battery_cache=battery_cache,
        default_camera_resource=scenario_profile.resources.scene_camera,
        display_runtime=display_runtime,
        memory_provider=memory_provider,
    )
    tools.extend(build_knowledge_tools(scenario_profile.knowledge_bases))

    resolved_prompt_file = resolve_prompt_file(scenario_profile.realtime.prompt_file)
    base_system_prompt = load_system_prompt(resolved_prompt_file)

    state_observer = StructuredStateObserver()
    engagement = EngagementStateMachine(
        voice_cmd_publisher=wiring.publish_voice_cmd,
        alert_timeout_sec=scenario_profile.engagement.alert_timeout_sec,
        cooldown_sec=scenario_profile.engagement.cooldown_sec,
        speaking_timeout_sec=scenario_profile.engagement.speaking_timeout_sec,
        on_idle_entered=wiring.on_idle_entered,
        nav_state=nav_state,
        battery_cache=battery_cache,
        self_charge_available=self_charge_available,
        state_observer=state_observer,
    )
    gesture_runtime = _create_gesture_runtime(
        scenario_profile=scenario_profile,
        robot_client=robot_client,
        engagement=engagement,
    )

    agent = RealtimeRobotAgent(
        scenario_profile=scenario_profile,
        robot_client=robot_client,
        tools=tools,
        base_system_prompt=base_system_prompt,
        engagement=engagement,
        coalescer=None,
        face_service=face_service,
        speaker_service=speaker_service,
        identity_memory_client=identity_memory_client,
        adaptive_update_coordinator=adaptive_update_coordinator,
        raw_data_capture=raw_data_capture,
        memory_context_compiler=memory_context_compiler,
        preference_extractor=preference_extractor,
        preference_extraction_enabled=(
            scenario_profile.identity_memory.record_live_episodes
            and preference_extractor is not None
        ),
        location_store=location_store_for_prompt,
        nav_state=nav_state,
        battery_cache=battery_cache,
        gesture_runtime=gesture_runtime,
        display_runtime=display_runtime,
        owner_turn_controller=None,
        initial_robot_posture=initial_robot_posture,
        stand_tool_name=_stand_tool_name(robot_family),
        supports_navigation=navigation_enabled,
        state_observer=state_observer,
    )
    if (
        scenario_profile.face_recognition.enabled
        and scenario_profile.face_recognition.owner_turn.enabled
        and face_service is not None
    ):
        from argos_src.agent.owner_turn import OwnerTurnController, OwnerTurnSettings

        owner_turn = scenario_profile.face_recognition.owner_turn
        agent.owner_turn_controller = OwnerTurnController(
            connector=robot_client,
            face_service=face_service,
            nav_state=nav_state,
            recording_state_provider=agent.is_recording_active,
            settings=OwnerTurnSettings(
                enabled=owner_turn.enabled,
                deadband_deg=owner_turn.deadband_deg,
                turn_gain=owner_turn.turn_gain,
                max_turn_deg=owner_turn.max_turn_deg,
                angular_speed_rad_s=owner_turn.angular_speed_rad_s,
                command_hz=owner_turn.command_hz,
                delay_after_recording_sec=owner_turn.delay_after_recording_sec,
                odom_frame=owner_turn.odom_frame,
                robot_frame=owner_turn.robot_frame,
                yaw_tolerance_deg=owner_turn.yaw_tolerance_deg,
                max_duration_sec=owner_turn.max_duration_sec,
                slow_zone_deg=owner_turn.slow_zone_deg,
                min_angular_speed_rad_s=owner_turn.min_angular_speed_rad_s,
            ),
        )
    wiring.bind_agent(agent)
    attach_recording_state_provider = getattr(
        engagement,
        "attach_recording_state_provider",
        None,
    )
    recording_state_provider = getattr(agent, "is_recording_active", lambda: False)
    if callable(attach_recording_state_provider):
        attach_recording_state_provider(recording_state_provider)
    else:
        engagement._recording_state_provider = recording_state_provider

    if scenario_profile.face_recognition.enabled and face_service is not None:
        subscribe_presence = getattr(face_service, "subscribe_presence", None)
        if callable(subscribe_presence):
            unsubscribe_presence = subscribe_presence(agent.update_face_presence_snapshot)
            atexit.register(unsubscribe_presence)

    coalescer = EventCoalescer(
        agent=agent,
        engagement=engagement,
        debounce_sec=scenario_profile.engagement.coalescer_debounce_sec,
        max_wait_sec=scenario_profile.engagement.coalescer_max_wait_sec,
        state_observer=state_observer,
    )
    agent.coalescer = coalescer
    wiring.bind_coalescer(coalescer)
    _attach_engagement_wiring(engagement, coalescer)

    patrol_bridge = None
    if nav_state is not None:
        patrol_bridge = PatrolLoopBridge(
            robot_client=robot_client,
            nav_state=nav_state,
            battery_cache=battery_cache,
            next_hop_delay_sec=scenario_profile.engagement.patrol_next_hop_delay_sec,
        )
        wiring.bind_patrol_bridge(patrol_bridge)

        from argos_src.tools.unitree_go2.navigation.toolset import (
            process_navigation_event,
        )

        subscribe_navigation = getattr(robot_client, "subscribe_navigation", None)
        if callable(subscribe_navigation):
            unsubscribe_navigation = subscribe_navigation(
                lambda event: process_navigation_event(
                    state=nav_state,
                    event=event,
                    on_nav_event=wiring.submit_nav_event,
                )
            )
            atexit.register(unsubscribe_navigation)

    face_event_bridge = None
    if scenario_profile.face_recognition.enabled and face_service is not None and nav_state is not None:
        face_event_bridge = FaceEventBridge(
            face_service=face_service,
            robot_client=robot_client,
            coalescer=coalescer,
            engagement=engagement,
            nav_state=nav_state,
            presence_callback=None,
            recognized_greet_enabled=scenario_profile.face_recognition.proactive_greeting.recognized_enabled,
            unknown_greet_enabled=scenario_profile.face_recognition.proactive_greeting.unknown_enabled,
            require_attention=scenario_profile.face_recognition.proactive_greeting.require_attention,
            recognized_greet_cooldown_sec=scenario_profile.face_recognition.proactive_greeting.recognized_cooldown_sec,
            unknown_greet_cooldown_sec=scenario_profile.face_recognition.proactive_greeting.unknown_cooldown_sec,
        )
        face_event_bridge.start()

    wiring.maybe_start_startup_patrol(
        startup_patrol_route=list(scenario_profile.navigation.startup_patrol_route),
        navigation_runtime_store=navigation_runtime_store,
        startup_delay_sec=scenario_profile.engagement.startup_patrol_delay_sec,
    )

    atexit.register(agent.shutdown)
    if face_event_bridge is not None:
        atexit.register(face_event_bridge.stop)

    return agent

def _format_navigation_event(event: dict) -> str:
    """Create a compact deterministic nav event text for the LLM."""
    event_type = event.get("event_type", "unknown")
    goal_id = event.get("goal_id", "unknown")
    tool_name = event.get("tool_name", "navigation")
    target = event.get("target_label", "unknown")
    if event_type == "goal_result":
        outcome = event.get("outcome", "unknown")
        status_name = event.get("status_name", "UNKNOWN")
        error_msg = event.get("error_msg", "")
        if error_msg:
            return (
                "NAV_EVENT: "
                f"type=goal_result goal_id={goal_id} tool={tool_name} "
                f"target={target} final=true outcome={outcome} "
                f"status={status_name} error={error_msg}"
            )
        return (
            "NAV_EVENT: "
            f"type=goal_result goal_id={goal_id} tool={tool_name} "
            f"target={target} final=true outcome={outcome} "
            f"status={status_name}"
        )
    if event_type == "waypoint_reached":
        idx = event.get("waypoint_index", "?")
        total = event.get("waypoint_total", "?")
        name = event.get("waypoint_name", f"waypoint_{idx}")
        return (
            "NAV_EVENT: "
            f"type=waypoint_reached goal_id={goal_id} tool={tool_name} "
            f"target={target} final=false index={idx} total={total} "
            f"name={name}"
        )
    return (
        f"NAV_EVENT: type={event_type} goal_id={goal_id} "
        f"tool={tool_name} target={target}"
    )
