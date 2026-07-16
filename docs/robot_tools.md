# Robot Tool And Provider Contract

This document describes the model-visible robot tool contract. Read it with:

- `argos_src/tools/tool_ids.py`
- `argos_src/tools/registry.py`
- `argos_src/tools/unitree_go2/`
- `config/profiles/static_interaction.yaml`
- `config/profiles/cody_interaction.yaml`
- `config/manifests/puffle.yaml`
- `config/manifests/cody.yaml`
- `docs/realtime_turn_flow.md`

The Realtime model sees function schemas. Local Python resolves public tool IDs
to runtime tool names, validates selected provider capabilities, executes the
tool, and inserts the tool result back into Realtime history.

## Profile And Manifest Binding

Profiles select a manifest and resource IDs:

```yaml
manifest: puffle
resources:
  primary_robot: base
  face_camera: arducam_001
  scene_camera: arducam_001
  interaction_display: screen_001
```

The manifest defines providers, resources, resource families, and capabilities.
For the default Puffle profile:

- `base` is the Unitree Go2 robot resource.
- `arducam_001` is the RGB camera resource.
- `screen_001` is the optional display resource.

`profile_config.py` validates that enabled tool IDs have matching selected
resource capabilities before the agent is built.

## Default Enabled Tools

`static_interaction` currently enables posture/action, short motion, visual
capture, face enrollment, and employee identity resolution.

`cody_interaction` also enables `memory.search_semantic`, which exposes a
read-only Tailwag semantic-memory query tool. Navigation tools are available in
code but are not enabled in those interaction profiles unless added to
`tools.enabled_tool_ids`.

| Public tool ID | Runtime tool | Capability | Main side effect |
|---|---|---|---|
| `posture.rest` | `go2_damp` | `posture.command` | Rest/damp posture. |
| `posture.stand` | `go2_balance_stand` | `posture.command` | Stand posture. |
| `posture.stop` | `go2_stop_move` | `posture.command` | Stop posture/action. |
| `posture.sit` | `go2_sit` | `posture.command` | Sit posture. |
| `motion.move_robot` | `move_robot` | `motion.velocity` | Short velocity command, then stop. |
| `vision.capture_scene` | `capture_scene` | `camera.rgb` | Captures a provider camera image for the model. |
| `identity.enroll_visible_person` | `enroll_visible_person` | `camera.rgb` | Runs face enrollment; saves only after quality gates and display acceptance when configured. |
| `identity.resolve_employee_identity` | `resolve_employee_identity` | none | Local employee-directory lookup. |
| `memory.search_semantic` | `search_memory_semantic` | none | Searches Tailwag episodes and durable memory items for the current recognized owner. |
| `embodiment.unitree_go2.hello` | `go2_hello` | `embodiment.action` | Go2 gesture/action. |
| `embodiment.unitree_go2.stretch` | `go2_stretch` | `embodiment.action` | Go2 gesture/action. |
| `embodiment.unitree_go2.content` | `go2_content` | `embodiment.action` | Go2 gesture/action. |
| `embodiment.unitree_go2.bow_down` | `go2_bow_down` | `embodiment.action` | Go2 gesture/action. |
| `embodiment.unitree_go2.look_up` | `go2_look_up` | `embodiment.action` | Go2 gesture/action. |
| `embodiment.unitree_go2.left_tilt` | `go2_left_tilt` | `embodiment.action` | Go2 gesture/action. |
| `embodiment.unitree_go2.right_tilt` | `go2_right_tilt` | `embodiment.action` | Go2 gesture/action. |

Additional Go2 action IDs exist for `damp`, `balance_stand`, `stop_move`,
`sit`, `dance1`, `dance2`, `scrape`, `front_jump`, `front_pounce`, and
`finger_heart`. Use the public `embodiment.unitree_go2.<action>` IDs in
profiles, not runtime tool names, unless you are writing tests around the
registry.

## Spot Tool IDs

The same registry also supports Spot profiles. Spot-specific public IDs include:

| Public tool ID | Runtime tool | Capability |
|---|---|---|
| `spot.system.claim` | `spot_claim` | `posture.command` |
| `spot.system.release` | `spot_release` | `posture.command` |
| `spot.system.power_on` | `spot_power_on` | `posture.command` |
| `spot.system.power_off` | `spot_power_off` | `posture.command` |
| `posture.stand` | `spot_stand` | `posture.command` |
| `posture.stop` | `spot_stop` | `posture.command` |
| `posture.sit` | `spot_sit` | `posture.command` |
| `posture.self_right` | `spot_self_right` | `posture.command` |
| `posture.rollover` | `spot_rollover` | `posture.command` |
| `posture.set_stand_height` | `spot_set_stand_height` | `posture.command` |
| `posture.reset_body_pose` | `spot_reset_body_pose` | `posture.command` |

`motion.move_robot` also resolves to Spot's `move_robot` when the manifest
primary robot family is `spot`.

## Optional Navigation And Docking Tools

These IDs are available when the profile enables them and the selected manifest
resource provides the required capabilities.

| Public tool ID | Runtime tool | Required capability | Notes |
|---|---|---|---|
| `navigation.navigate_to_location` | `navigate_to_location` | `navigation.goal` | Starts an interruptible named-location goal and returns before arrival. |
| `navigation.navigate_to_location_blocking` | `navigate_to_location_blocking` | `navigation.goal` | Waits for final navigation result. |
| `navigation.navigate_relative` | `navigate_relative` | `navigation.goal`, `transform.lookup` | Builds a map-frame goal from current transform. |
| `navigation.follow_waypoints` | `follow_waypoints` | `navigation.goal` | Starts an interruptible waypoint route. |
| `navigation.cancel` | `cancel_navigation` | `navigation.goal` | Cancels active navigation and may save a resumable mission. |
| `navigation.stop_patrol` | `stop_patrol` | `navigation.goal` | Stops patrol and optionally cancels active navigation. |
| `navigation.localize_current_location` | `localize_current_location` | `navigation.goal`, `transform.lookup` | Compares current pose to saved locations without saving or marking state. |
| `navigation.mark_return_point` | `mark_return_point` | `navigation.goal`, `transform.lookup` | Stores a temporary task return point without persisting it. |
| `navigation.navigate_to_return_point_blocking` | `navigate_to_return_point_blocking` | `navigation.goal`, `transform.lookup` | Returns to a temporary return point and waits for arrival. |
| `navigation.save_current_location` | `save_current_location` | `navigation.goal`, `transform.lookup` | Persists the current pose as a named saved location. |
| `dock.charging` | `charging_dock` | `dock.charging` | Uses saved `charge_dock` approach pose and provider docking. |

`save_current_location` persists map resources. Treat it as an operator-controlled
action because overwriting names such as `charge_dock` changes future navigation
and docking behavior. Use `mark_return_point` for temporary "come back here"
mission state instead of writing a saved location.

## Optional Tailwag Memory Tool

`memory.search_semantic` is model-visible only when the active profile enables
it and `identity_memory.enabled` creates a Tailwag-backed identity-memory
client. The runtime tool searches episodes and extracted memory items scoped to
the current resolved `owner_id`.
If no current owner is recognized, the tool returns an error instead of running
a broad unscoped search.

The memory tool is read-only from Argos' point of view. Episode ingestion,
memory extraction, active follow-up handling, archival, and repair belong to
Tailwag. See `identity_memory.md` for the memory contract.

## Motion And Local Bounds

`move_robot` is for short nearby repositioning and expressive movement, not
mapped navigation. Its schema describes safe ranges such as gentle walking,
normal walking, spin speed, and duration. The local tool passes `max_duration=10`
to the provider and always expects the provider client to stop after the command.

The schema descriptions are not a replacement for provider-side motion safety.
Provider implementations should continue to clamp unsafe velocity, duration, and
navigation requests before touching hardware.

## Patrol, Battery, And Charging

Engagement state suppresses patrol while the robot is interacting. Startup
patrol, idle patrol resume, and patrol next hops are runtime-owned navigation
dispatches; they do not require the model to call navigation tools. Navigation
result events may still be surfaced as internal context for operator-visible
status or user-facing updates.

Battery state also gates navigation:

- `battery.low_battery_pct` comes from the profile; `static_interaction` uses `10.0`.
- Below the threshold, general navigation tools return a blocked result.
- Charging can still be allowed through `charging_dock` when configured.
- A charging-ready event can tell the model the robot may stand and resume normal work.

## Owner-Turn Motion

When a spoken turn resolves to a known owner, the runtime can request a short
background physical turn toward that person. This is not a model tool call. It
is driven by `face_recognition.owner_turn`, camera intrinsics, transform lookup,
and the owner-turn controller. Tool calls cancel the owner-turn request so tool
motion and orientation motion do not fight each other.

## Tool Loop Contract

For normal registered tools:

```text
model function_call
  -> local tool execution
  -> function_call_output item
  -> follow-up response.create after pending tool calls finish
```

Unknown tool calls should be treated as contract bugs: they can leave a turn
waiting for tool completion until the runtime watchdog cancels it. Keep profile
tool IDs, runtime schemas, and tests aligned.

## Manual Robot Safety

Run hardware-free tests before live motion. For manual robot checks, confirm:

- clear physical space around the robot
- a working stop path or operator e-stop
- current battery state
- the active profile's enabled tool IDs
- whether navigation or startup patrol is enabled

Do not launch live robot/provider/runtime commands or issue robot motion without
explicit operator approval.
