# Argos Complete Setup

This guide describes the ROS-agnostic Argos runtime setup. Argos talks to robot
capabilities over Zenoh. ROS, Unitree SDKs, Nav2, camera drivers, RTSP, REST,
and serial devices belong in the external robot provider process, not in the
Argos agent process.

## Architecture

```text
Argos agent/tools
  -> RobotClient capability API
  -> ZenohRobotClient
  -> Argos capability messages on Zenoh keys
  -> external robot provider
  -> ROS 2 / Nav2 / Go2 SDK / camera stack
  -> robot hardware
```

Argos should not source a ROS workspace. If a capability needs ROS message
packages such as `go2_interfaces`, only the external provider should import
them.

## Runtime Contract

Argos publishes provider-neutral capability requests. The provider translates
those requests to the current robot stack.

Example Argos request:

```json
{
  "id": "uuid",
  "type": "request",
  "op": "go2.action",
  "args": {
    "api_id": 1003,
    "parameter": {},
    "topic": "rt/api/sport/request",
    "priority": 0
  },
  "timeout_ms": 3000,
  "ts": 1780000000.123
}
```

Example provider translation:

```text
go2.action
  -> /webrtc_req
  -> go2_interfaces/msg/WebRtcReq
  -> topic=rt/api/sport/request
```

The bridge protocol lives in `argos_src/robot_bridge/protocol.py`.

## Fresh Machine: Argos Runtime

### 1. Install Base Packages

```bash
sudo apt update
sudo apt install -y \
  git \
  curl \
  python3-dev \
  python3-pip \
  python3-venv \
  libportaudio2 \
  portaudio19-dev \
  ffmpeg
```

### 2. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry --version
```

If Poetry hits desktop keyring issues:

```bash
export POETRY_KEYRING_ENABLED=false
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

### 3. Clone And Install Argos

```bash
export WORKSPACE_DIR="$HOME/go2_quadruped_companion"
git clone git@github.com:Chewy-Inc/go2_quadruped_companion.git "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR/argos_src"
poetry install
source setup_shell.sh
python3 -m pip install --no-deps -r argos_src/face_recognition/requirements.txt
python3 -m pip install eclipse-zenoh
```

`setup_shell.sh` activates the Argos Poetry environment and exposes the repo
root on `PYTHONPATH`. It does not source ROS.

### 4. Configure Argos

```bash
export OPENAI_API_KEY=...
export ARGOS_ROBOT_TRANSPORT=zenoh
export ARGOS_ZENOH_KEY_PREFIX=pair/robots/puffle
```

If the provider is not discovered automatically, configure Zenoh endpoints:

```bash
export ARGOS_ZENOH_CONNECT=tcp/ROBOT_OR_PROVIDER_HOST:7447
```

Optional integrations:

```bash
export SLACK_BOT_TOKEN=...
```

Snowflake-backed employee directory variables are still optional and only needed
when that profile feature is enabled.

### 5. Start Argos

Start the external robot provider first. Then start Argos:

```bash
cd "$WORKSPACE_DIR/argos_src"
source setup_shell.sh
export OPENAI_API_KEY=...
export ARGOS_ROBOT_TRANSPORT=zenoh
python3 run_profile.py --profile static_interaction
```

Argos should not require `source /opt/ros/...` or a Go2 SDK overlay. If it does,
that dependency has leaked back into the agent process.

## Robot Provider Setup

The provider is a separate process handled outside the Argos agent runtime. It
may run on the robot computer, a ROS workstation, or another machine with access
to the robot network.

The provider owns:

- ROS workspace sourcing
- `go2_interfaces` and other robot message packages
- Nav2 actions/services
- camera topics and transforms
- Unitree SDK / CycloneDDS setup
- optional `zenoh-bridge-ros2dds` routing

Provider-side shape:

```text
external provider
  -> receives Argos capability messages on Zenoh
  -> translates to ROS/SDK calls
  -> publishes responses/events back to Argos
```

Use `zenoh-bridge-ros2dds` when the provider needs ROS 2/DDS traffic routed
across machines. Do not make Argos publish ROS-shaped messages directly to that
bridge.

## Fake Transport

`ARGOS_ROBOT_TRANSPORT=fake` exists for tests and isolated unit checks. It is
not a deployment mode and should not be used to validate physical robot I/O.

## Knowledge Bases

Scenario profiles can load built-in knowledge tools with:

```yaml
knowledge_bases:
  - kind: whoami_query
    root_dir: chewy_docs
    tool_name: query_chewy_knowledge
    description: Search Chewy Robotics company documentation.
    k: 4
```

Knowledge tooling is Argos-owned. Build or rebuild a local FAISS knowledge base
with:

```bash
cd ~/rai
python3 -m argos_src.knowledge.build_faiss chewy_docs
```

Existing `generated/index.faiss`, `generated/index.pkl`, and
`generated/vdb_kwargs.json` folders created by the previous external builder
remain compatible.

## Troubleshooting

- If Argos cannot import `zenoh`, install the Python Zenoh package in the Argos
  environment with `python3 -m pip install eclipse-zenoh`.
- If Argos times out on robot actions, confirm the provider is running and using
  the same `ARGOS_ZENOH_KEY_PREFIX`.
- If the provider cannot publish Go2 actions, check ROS sourcing and
  `go2_interfaces` on the provider machine, not the Argos machine.
- If camera snapshots fail, confirm the provider implements
  `camera.latest_image` or `camera.latest_rgbd`.
- If owner-turn fails, confirm the provider implements `tf.transform` and
  `motion.velocity_sample`.
- If OpenAI Realtime fails, confirm `OPENAI_API_KEY`, microphone/speaker device
  names in the profile, and `logs/latency.log`.

## Regression Checks

Run focused Argos tests from the repo root:

```bash
python3 -B -m pytest \
  tests/argos_src/robot_api/test_zenoh_robot_client.py \
  tests/argos_src/agent/test_agent_runtime.py \
  tests/argos_src/agent/test_orchestrator.py \
  tests/argos_src/agent/test_bridges.py \
  tests/argos_src/agent/test_factory_gestures.py \
  tests/argos_src/agent/test_gesture_runtime.py \
  tests/argos_src/agent/test_owner_turn.py \
  tests/argos_src/face_recognition/test_face_recognition_service.py \
  tests/argos_src/test_argos_profile_config.py
```

The key acceptance criterion is that the Argos runtime imports and starts
without ROS, while physical robot I/O goes through the Zenoh provider.
