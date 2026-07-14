# Argos Complete Setup

This guide describes the ROS-agnostic Argos runtime setup. Argos talks to
provider/resource capabilities over Zenoh. ROS, Unitree SDKs, Nav2, camera
drivers, RTSP, REST, and serial devices belong in the external provider
process, not in the Argos agent process.

## Architecture

```text
Argos agent/tools
  -> provider-backed capability client
  -> provider API transport
  -> Argos provider/resource messages on Zenoh keys
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

The bridge protocol lives in `argos_src/provider_api/wire.py`.

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
export WORKSPACE_DIR="$HOME/argos-agent"
git clone https://github.com/spatil2-chwy/argos-agent.git "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"
poetry install
source setup_shell.sh
python3 -m pip install --no-deps -r argos_src/face_recognition/requirements.txt
```

`setup_shell.sh` activates the Argos Poetry environment and exposes the repo
root on `PYTHONPATH`. It does not source ROS.

Tailwag-backed memory now runs as an HTTP provider. Keep the sibling
`../tailwag-memory` checkout available to run the Tailwag service, but Argos no
longer installs it as a Python package.

### 4. Configure Argos

```bash
export OPENAI_API_KEY=...
```

The selected profile loads `config/manifests/puffle.yaml`, which supplies the
provider transport, key prefix, and resource IDs. If the Zenoh provider is not
discovered automatically, configure endpoints:

```bash
export ARGOS_ZENOH_CONNECT=tcp/ROBOT_OR_PROVIDER_HOST:7447
```

Optional integrations:

```bash
export SLACK_BOT_TOKEN=...
```

Tailwag-backed memory requires Tailwag runtime configuration when
`identity_memory.enabled: true`:

```bash
export TAILWAG_API_BEARER_TOKEN=...
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=tailwag-memory
export OPENAI_API_KEY=...
export TAILWAG_EMBEDDING_MODEL=text-embedding-3-small
export TAILWAG_EMBEDDING_DIMENSION=64
export TAILWAG_SYNTHESIS_MODEL=gpt-5.5
```

The bundled manifests expect Tailwag on `http://localhost:8000` at
`/argos/providers/memory/resources/memory/...`.

`SLACK_BOT_TOKEN` is only required when Tailwag Slack polling is enabled. Keep
the token in the environment and put only `bot_token_env: SLACK_BOT_TOKEN` in
the profile YAML.

For Puffle's local browser screen, run the display control server separately at
`http://localhost:4173`. The `puffle` manifest selects the HTTP-backed
`screen_001` resource through provider `puffle-go2-display`.
Set `display.enabled: false` in a profile when running without the screen.

Snowflake-backed employee directory variables are still optional and only needed
when that profile feature is enabled.

### 5. Start Argos

Start the external robot provider first. Then start Argos:

```bash
cd "$WORKSPACE_DIR"
source setup_shell.sh
export OPENAI_API_KEY=...
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
- camera streams and transforms
- Unitree SDK / CycloneDDS setup
- optional `zenoh-bridge-ros2dds` routing

Provider-side shape:

```text
external provider
  -> receives Argos provider/resource capability messages on Zenoh
  -> translates to ROS/SDK calls
  -> publishes responses/events back to Argos
```

Use `zenoh-bridge-ros2dds` when the provider needs ROS 2/DDS traffic routed
across machines. Do not make Argos publish ROS-shaped messages directly to that
bridge.

## Fake Transport

`ARGOS_PROVIDER_TRANSPORT=fake` exists for tests and isolated unit checks. It is
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
cd ~/argos-agent
python3 -m argos_src.knowledge.build_faiss chewy_docs
```

Existing `generated/index.faiss`, `generated/index.pkl`, and
`generated/vdb_kwargs.json` folders remain compatible.

## Troubleshooting

- If Argos times out on robot actions, confirm the provider is running and using
  the same manifest key prefix and resource IDs as the selected profile.
- If the provider cannot publish Go2 actions, check ROS sourcing and
  `go2_interfaces` on the provider machine, not the Argos machine.
- If camera snapshots fail, confirm the selected camera resource provides
  `camera.rgb` or `camera.rgbd`.
- If owner-turn fails, confirm the selected robot and camera resources provide
  `transform.lookup`, `camera.intrinsics`, and the needed motion capability.
- If the Puffle screen does not update, confirm the local display server is
  running at `http://localhost:4173` and that the selected profile includes
  `display.enabled: true`.
- If OpenAI Realtime fails, confirm `OPENAI_API_KEY`, microphone/speaker device
  names in the profile, and `logs/latency.log`.

## Regression Checks

Run focused Argos tests from the repo root:

```bash
python3 -B -m pytest \
  tests/argos_src/provider_api/test_zenoh_provider_client.py \
  tests/argos_src/agent/test_agent_runtime.py \
  tests/argos_src/agent/control/test_engagement_coalescer.py \
  tests/argos_src/agent/test_bridges.py \
  tests/argos_src/agent/test_factory_gestures.py \
  tests/argos_src/agent/test_gesture_runtime.py \
  tests/argos_src/agent/test_owner_turn.py \
  tests/argos_src/face_recognition/test_face_recognition_service.py \
  tests/argos_src/test_argos_profile_config.py
```

The key acceptance criterion is that the Argos runtime imports and starts
without ROS, while physical robot I/O goes through the Zenoh provider.
