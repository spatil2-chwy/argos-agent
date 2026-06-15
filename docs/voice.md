# Argos Voice Interaction

Talk to your Go2 robot with the Argos realtime speech runtime: direct microphone audio in, direct speaker audio out, same-session transcription for observability/memory, and realtime tool calling in one persistent session.

Use [launch.md](/home/spatil2/argos-agent/docs/launch.md) for setup and bring-up.
Use [observability.md](/home/spatil2/argos-agent/docs/observability.md) for latency debugging.
Use [speaker_recognition.md](/home/spatil2/argos-agent/docs/speaker_recognition.md) for speaker ownership and voice enrollment details.
Use [interaction_display.md](/home/spatil2/argos-agent/docs/interaction_display.md) for Puffle screen state and subtitles.

## Example

```
You: "Hello Puffle!"
Robot: "Hi there."

You: "Can you check that for me?"
Robot: "Sure, let me take a look."
```

## Latency Logging

The realtime runtime prints timestamped lines so you can measure end-to-end latency.
Sample output:

```
ts=2026-04-24 13:23:45.100 | component=realtime | event=speech_end
ts=2026-04-24 13:23:45.151 | component=realtime | event=audio_commit | req_id=rt-abc123
ts=2026-04-24 13:23:45.152 | component=realtime | event=response_create | req_id=rt-abc123
ts=2026-04-24 13:23:45.487 | component=realtime | metric=first_audio_latency_s | req_id=rt-abc123 | duration_s=0.387
```

**What each diff measures:**

| Step | Log line | What it times |
|------|----------|---------------|
| 1 | `event=speech_end` | Local end-of-speech detection |
| 2 | `event=audio_commit` | Time to commit buffered mic audio into the realtime session |
| 3 | `event=response_create` | When the model turn was requested |
| 4 | `metric=first_audio_latency_s` | **Felt latency** from speech end to first speaker audio |

## Core Files

```
.
├── run_profile.py              # Single-process launcher
└── argos_src/
    ├── agent/agent_runtime.py      # Persistent realtime session + audio/tool loop
    ├── agent/agent_audio.py        # Audio capture, commit, and playback callbacks
    ├── agent/agent_events/         # Realtime server-event parsing + routing helpers
    ├── display/runtime.py          # Optional Puffle screen facade
    └── runtime/audio_admission.py  # Local wake/face/cooldown admission logic
```
