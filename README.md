# Argos Realtime Voice Agent

Argos companion stack for Unitree Go2 with a profile-driven realtime speech runtime, direct audio input/output, tool calling, navigation, charging, and face-recognition-backed context.

## Run

```bash
cd ~/argos-agent
poetry install
source setup_shell.sh
python3 -m pip install --no-deps -r argos_src/face_recognition/requirements.txt
python3 run_profile.py --profile static_interaction
```

Canonical operator guide: `docs/launch.md`. Full documentation map:
`docs/README.md`.

## Canonical Runtime Paths

```text
.
├── run_profile.py
├── pyproject.toml
├── poetry.lock
├── config/
│   ├── manifests/
│   └── profiles/
├── resources/
│   ├── nav_locations/
│   ├── prompts/
│   └── wake_words/
├── scripts/
│   └── labs/
├── var/              # ignored local runtime state
├── docs/
└── argos_src/
    ├── agent/
    ├── capabilities/
    ├── display/
    ├── face_recognition/
    ├── identity_memory/
    ├── integrations/
    ├── media/
    ├── provider_api/
    ├── runtime/
    └── tools/
        ├── common/
        └── unitree_go2/
```

## Docs

- Docs index: `docs/README.md`
- Operator launch/runbook: `docs/launch.md`
- Runtime architecture and turn flow: `docs/architecture.md`, `docs/realtime_turn_flow.md`
- Prompting and history: `docs/prompting_and_history.md`
- Robot tools/provider contract: `docs/robot_tools.md`
- Identity and memory: `docs/face_recognition.md`, `docs/speaker_recognition.md`
- Display, observability, setup: `docs/interaction_display.md`, `docs/observability.md`, `docs/complete_setup.md`
