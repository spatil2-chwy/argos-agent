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
├── docs/
└── argos_src/
    ├── config/profiles/
    ├── runtime/
    ├── observability/
    ├── agent/
    ├── nav_support/
    │   └── locations.py
    └── tools/
        ├── common/
        └── unitree_go2/
```

## Docs

- Launch and testing: `docs/launch.md`
- Architecture: `docs/architecture.md`
- Voice/runtime notes: `docs/voice.md`
- Face recognition: `docs/face_recognition.md`
- Employee directory: `docs/employee_directory.md`
- Observability: `docs/observability.md`
