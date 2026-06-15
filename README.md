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

Canonical operator guide: `docs/launch.md`

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
    ├── identity/
    │   └── embeddings/
    ├── integrations/
    ├── media/
    ├── memory/
    ├── provider_api/
    ├── runtime/
    └── tools/
        ├── common/
        └── unitree_go2/
```

## Docs

- Launch and testing: `docs/launch.md`
- Architecture: `docs/architecture.md`
- Voice/runtime notes: `docs/voice.md`
- Puffle interaction display: `docs/interaction_display.md`
- Face recognition: `docs/face_recognition.md`
- Speaker recognition: `docs/speaker_recognition.md`
- Identity store: `docs/identity_store.md`
- Memory store: `docs/memory_store.md`
- Employee directory: `docs/employee_directory.md`
- Observability: `docs/observability.md`
