# Argos Robot Bridge

The Argos agent should talk to robot capabilities, not directly to ROS,
Unitree DDS, RTSP, REST, or serial devices. Zenoh is the default transport for
those capability messages.

Robot provider processes may use transport-specific dependencies. The agent
process uses `argos_src.robot_api.RobotClient` only.

V1 bridge scope:

- Go2 action commands
- short velocity movement
- camera snapshots / RGBD frames
- camera intrinsics and transforms for owner-turn
- battery telemetry
- optional face-presence and voice-command publication for external observers

Recommended ROS-node path:

```text
Argos tools/runtime
  -> RobotClient capability operation
  -> ZenohRobotClient
  -> Argos capability message on Zenoh
  -> robot provider
  -> ROS 2 topic/service/action or SDK call
  -> ROS 2 hardware node / robot SDK
```

Transport rule: do not expose ROS topic names or ROS message types as the
Argos-facing API. Use the capability operation names in `protocol.py`; the
provider decides how to reach the underlying robot subsystem. If
`zenoh-bridge-ros2dds` is useful, keep it behind the provider.

Multi-robot deployments should run one Argos agent per robot and give each
robot a unique bridge key prefix. The capability names stay common; the prefix
selects the robot.

```yaml
robot:
  id: puffle
  family: unitree_go2
  display_name: Puffle
  bridge:
    transport: zenoh
    key_prefix: pair/robots/puffle
    connect_endpoints: []
```

This routes requests under keys like:

```text
pair/robots/puffle/request/{request_id}
pair/robots/puffle/response/{request_id}
pair/robots/puffle/event/{op}
```
