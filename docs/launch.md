# Launch and Testing

The supported runtime is:

- robot bring-up in one terminal
- one Argos realtime agent process in another terminal
- direct audio in and audio out through the OpenAI Realtime session

The supported profile for launch is `static_interaction`.

## What Runs

`python3 run_profile.py --profile static_interaction`

That starts one process which owns:

- microphone capture
- speaker playback
- the OpenAI Realtime websocket session
- local wake/face/cooldown admission logic
- engagement state publishing
- tool calling
- transcript side-channel for preference extraction
- optional interaction display updates for Puffle's browser screen

## Prerequisites

Prepare the Argos environment:

```bash
cd ~/argos-agent
poetry install
source setup_shell.sh
python3 -m pip install --no-deps -r argos_src/face_recognition/requirements.txt
```

You also need:

- `OPENAI_API_KEY` exported in the shell that will run Argos
- the dedicated Argos Poetry environment installed with `cd ~/argos-agent && poetry install`
- working microphone and speaker devices on the host machine
- outbound HTTPS connectivity to the Tailwag API Gateway
- `TAILWAG_API_BEARER_TOKEN` injected from AWS Secrets Manager secret
  `aaggarwal1-tailwag/api-bearer-token` without committing it

`setup_shell.sh` is the canonical Argos activation path. It activates the
Argos Poetry environment and exposes the repo root for `argos_src` imports. Prefer
that over ad-hoc `pip install ...` commands.

For the default `static_interaction` profile, install the face model add-on too:

```bash
python3 -m pip install --no-deps -r argos_src/face_recognition/requirements.txt
```

That extra step is still manual because `facenet-pytorch 2.6.0` advertises a
`torch<2.3` constraint while Argos uses a newer torch/torchvision stack.
Install it without dependency resolution so Poetry keeps the Argos CUDA
packages intact.

If you have not prepared the environment yet:

```bash
cd ~/argos-agent
poetry install
source setup_shell.sh
python3 -m pip install --no-deps -r argos_src/face_recognition/requirements.txt
```

## Recommended Launch

Terminal 1: external robot provider

Start the provider process that owns ROS, SDK, camera, navigation, or other
robot-specific dependencies. Argos only talks to that provider through the
manifest-selected provider/resource route.

Terminal 2: Argos realtime runtime

```bash
cd ~/argos-agent
source setup_shell.sh
export OPENAI_API_KEY=...
export TAILWAG_API_BEARER_TOKEN=...
python3 run_profile.py --profile static_interaction
```

For POC data-quality sessions, add raw artifact capture:

```bash
python3 run_profile.py --profile static_interaction --save-raw-data
```

Artifacts are written under `data_collection/raw_sessions/` by default, which is
git-ignored. Use `--raw-data-dir /path/to/dir` to put them somewhere else.

The profile selects `manifest: puffle` and resource IDs such as `base`,
`arducam_001`, and `screen_001`. When the manifest leaves Zenoh endpoints empty,
set `ARGOS_ZENOH_CONNECT=tcp/ROBOT_OR_PROVIDER_HOST:7447` if discovery is not
enough in your network.

Optional display: for Puffle's screen, start the local browser display server at
`http://localhost:4173` before enrollment review or visual state testing. Argos
talks to the selected `resources.interaction_display` screen resource. Set
`display.enabled: false` in the profile when running without the screen.

If startup succeeds, you should see the runtime print:

- the selected profile
- the realtime model and voice
- input and output device names
- a message saying the realtime agent is running

## Simplest Mental Model

Use `run_profile.py` for the Argos realtime runtime:

```bash
python3 run_profile.py --profile static_interaction
```

## First Bring-Up Checklist

1. Start the robot.
2. Start `run_profile.py`.
3. Stand in front of the camera.
4. Say the wake word or speak while attention-gated face admission is active.
5. Confirm the robot speaks back through the speaker.
6. Confirm face presence is updating in the Argos logs/provider events.
7. Confirm the logs show `recording_started`, `response_create`, and `playback_completed` for a normal turn.
8. If using the Puffle screen, confirm idle shows the happy face, recording shows `Recording...`, thinking shows `Thinking...`, and assistant speech streams subtitles.

## Behavioral Baseline

Before any structural cleanup, these are the baseline behaviors worth protecting:

1. Normal wake-word turn reaches `recording_started`, `response_create`, and `playback_completed`.
2. Attention-gated face presence can open passive listening when the robot is idle.
3. A tool call completes and the follow-up assistant response still plays.
4. Speaking over the robot interrupts playback cleanly.
5. A recognized-speaker conversation still flushes preference extraction on idle or shutdown.

## Common Launch Variants

Override wake word:

```bash
python3 run_profile.py --profile static_interaction --wake-word "hey mycroft"
```

Override wake thresholds:

```bash
python3 run_profile.py --profile static_interaction \
  --wake-threshold 0.6 \
  --wake-window-sec 6.0 \
  --silence-grace-period 0.8
```

Override map file and startup patrol route:

```bash
python3 run_profile.py --profile static_interaction \
  --map-file lab.json \
  --patrol-route "truck_loading,home"
```

Swap prompt file:

```bash
python3 run_profile.py --profile static_interaction \
  --prompt-file static_interaction_prompt.md
```

## Audio Device Notes

The supported live launcher does not currently expose `--input-device` or `--output-device` flags.

If you need to change audio devices, edit the retained profile:

[static_interaction.yaml](../config/profiles/static_interaction.yaml)

The relevant fields are:

- `realtime.input_device`
- `realtime.output_device`
- `realtime.input_sample_rate`
- `realtime.output_sample_rate`
- `realtime.input_block_size`

## Realtime API Sanity Check

There is not currently a checked-in standalone Argos-only realtime harness under `argos_src/`.

For a supported sanity path, start the normal runtime and use the latency logs in
parallel:

```bash
cd ~/argos-agent
export OPENAI_API_KEY=...
python3 run_profile.py --profile static_interaction
python3 -m argos_src.observability.latency_tail --follow --component realtime
```

That is still the best way to separate whether a failure is:

- the robot stack
- the realtime API session
- or your local microphone/speaker setup

## Face Enrollment

Face enrollment happens through the live `enroll_visible_person` tool during a
Argos interaction, after the person confirms their identity and consent.
When the `interaction_display` resource is configured, the tool shows a blocking
face-capture preview on the Puffle screen and saves only after Accept.

## Identity and Voice References

Identities, face references, voice references, Slack ingestion, and durable
memory are owned by Tailwag. Use approved Tailwag-side administrative tooling
when you need to inspect or delete a person completely; no Tailwag checkout or
process is needed on the Argos host.

For controlled lab collection, use the local-first biometric commands instead of
the live conversational enrollment tool:

```bash
python3 -m scripts.labs.biometric_enrollment_lab capture "Jane Doe"
python3 -m scripts.labs.biometric_enrollment_lab list
python3 -m scripts.labs.biometric_enrollment_lab push
python3 -m scripts.labs.biometric_enrollment_lab cleanup
```

Follow the complete safety, capture, push, retry, verification, and retention
procedure in [biometric_enrollment_lab.md](biometric_enrollment_lab.md).

`capture` makes no Tailwag/identity-memory request, although configured camera
and display providers may use their own endpoints. `push` requires a selected
identity-memory resource with both `memory.identity` and `memory.biometrics`,
and the matching Tailwag face-existence endpoint must be deployed first. It
resolves exactly one employee-directory record from the captured email prefix,
verifies the official name, validates an existing Person when present, checks existing face and voice references, and fails closed if
either check is unavailable or malformed. Before embeddings are sent,
the operator must confirm subject consent and type the canonical name. Push uploads only
missing aggregate vectors, journals each modality independently, and never
deletes local files. `--provider-transport fake` is not a push dry-run.
`cleanup` permanently removes a selected intact, upload-complete local bundle;
unencrypted incomplete or corrupt captures require administrator review.


## Camera Preview

Use the external robot provider's camera preview/debug tool. Argos itself does
not open ROS image topics; it asks the provider for decoded frames through the
robot client.

## Manual Smoke Tests

Run the targeted regression tests before live robot checks. For any smoke test
that can move the robot, confirm clear space, battery state, a working stop path,
and operator approval. For destructive identity/biometric cleanup, use Tailwag
tooling.

After bring-up, these are the highest-value manual checks:

1. Wake-word turn from idle: say the wake word and ask a short question.
2. Attention-gated face turn: stand in view, face the camera, and speak without the wake word.
3. Internal-event turn: trigger a non-motion provider event, or a nav/battery event only with operator approval.
4. Tool call: ask for something that should call a known non-navigation tool, like visual inspection; use motion/trick tools only with operator approval.
5. Interruption: speak while the robot is talking and confirm playback stops cleanly.
6. Memory ingestion: have a short recognized-speaker conversation, then inspect the resulting episode/person memory with Tailwag tooling.

## Targeted Regression Tests

The fastest repo-local checks for the current Argos runtime are:

```bash
python3 -B -m pytest \
  tests/argos_src/agent/test_agent_runtime.py \
  tests/argos_src/agent/control/test_engagement_coalescer.py \
  tests/argos_src/face_recognition/test_face_recognition_service.py \
  tests/argos_src/test_argos_profile_config.py
```

Employee-directory resolution is covered through
`tests/argos_src/tools/unitree_go2/vision/test_resolve_employee_identity_tool.py`
and the tests in the separate Tailwag repository.

## Logs and Observability

Latency logs are written to `logs/latency.log`.

Useful helpers:

```bash
python3 -m argos_src.observability.latency_tail --follow
python3 -m argos_src.observability.latency_report
```

The current runtime emits realtime-oriented events such as:

- `speech_end`
- `audio_commit`
- `response_create`
- `first_audio_latency_s`

See [observability.md](observability.md) for details.

Two important runtime notes:

- Playback and engagement state stay inside the Argos runtime.
- turn-scoped dynamic instructions are attached on `response.create`; they are not inserted into conversation history.

## Troubleshooting

If the agent starts but you get no speech:

- confirm `OPENAI_API_KEY` is exported in the same shell
- confirm the microphone and speaker devices in `static_interaction.yaml`
- watch `logs/latency.log` while reproducing the problem
- check whether face presence or wake-word admission is actually opening the mic

If the robot hears you but never replies:

- watch `logs/latency.log`
- confirm the websocket session starts successfully
- confirm the tool call path is not failing repeatedly
- confirm the logs show `Realtime response created` and then either `playback_completed` or a clear cancellation reason
- if the next question seems to trigger an earlier answer, look for a stuck turn that never reached `response.done` or never started playback
- if replies get stale after a speaker handoff, confirm `inference_scope_selected` and `response_create` show the expected selected item ids

If face-triggered interaction is not happening:

- confirm `resources.face_camera` points at a manifest camera resource with `camera.rgb`
- confirm the face loop is running
- confirm face-presence provider events are reaching Argos


## Knowledge Bases

Scenario profiles can load one or more built-in knowledge tools through the `knowledge_bases:` list in the selected YAML profile.

Current supported kind:

- `whoami_query`

Each knowledge base is rooted at a directory like this:

```text
my_kb/
├── documentation/
├── images/
├── urdfs/
└── generated/
    ├── index.faiss
    ├── index.pkl
    ├── info.json
    └── vdb_kwargs.json
```

Build one knowledge base:

```bash
cd ~/argos-agent
python3 -m argos_src.knowledge.build_faiss chewy_docs
```

This builds `generated/index.faiss`, `generated/index.pkl`, and
`generated/vdb_kwargs.json` from files under `documentation/` and `urdfs/`.
Knowledge bases with existing generated files do not need to be rebuilt.

Use it from a Go2 profile:

```yaml
knowledge_bases:
  - kind: whoami_query
    root_dir: clinic_kb
    tool_name: query_clinic_knowledge
    description: Search the clinic knowledge base for policies and procedures.
    k: 4
```

If you do not want any knowledge-base tools for a profile, either omit `knowledge_bases:` entirely or set:

```yaml
knowledge_bases: []
```


## Tests

Useful targeted runs:

```bash
python3 -B -m pytest tests/argos_src
python3 -B -m pytest tests/argos_src/agent/test_agent_runtime.py
python3 -B -m pytest tests/argos_src/test_argos_profile_config.py
```

For the realtime rewrite itself, the most important validation is still manual robot/audio smoke testing.
