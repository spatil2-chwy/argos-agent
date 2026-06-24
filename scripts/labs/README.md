# Argos Helper Scripts

These scripts are small lab tools for testing Argos functionality without starting
the full realtime agent.

They still use the real Argos service code, profile config, provider camera
resources, face embedding model, and face/identity stores. They are meant for diagnosis and
parameter tuning, not for normal user-facing registration.

Run from the repo root:

```bash
cd ~/argos-agent
source setup_shell.sh
poetry run python -m scripts.labs.face_registration_lab --help
poetry run python -m scripts.labs.face_recognition_lab --help
poetry run python -m scripts.labs.face_capture_lab --help
poetry run python -m scripts.labs.audio_detection_lab --help
poetry run python -m scripts.labs.speaker_recognition_lab --help
poetry run python -m scripts.labs.rapidfuzz_employee_lab --help
poetry run python -m scripts.labs.openai_say_lab --help
```

## Structured perception labs + eval

Use these when you want repeatable artifacts plus a label file for quantitative
evaluation. Labs write runs under `var/labs/...`; eval writes reports under
`var/eval/perception/...`.

Face enrollment/detection/recognition/depth/attention capture:

```bash
poetry run python -m scripts.labs.face_capture_lab --mode enrollment --frames 10 --interval-sec 1
poetry run python -m scripts.labs.face_capture_lab --mode recognition --frames 10 --interval-sec 1
poetry run python -m scripts.labs.face_capture_lab --mode attention --frames 20 --interval-sec 1
poetry run python -m scripts.labs.face_capture_lab --mode depth --frames 10 --interval-sec 1
poetry run python -m scripts.labs.face_capture_lab --mode all --frames 10 --interval-sec 1
```

For sparse attention stills, use raw per-frame attention scoring so production
temporal smoothing does not suppress otherwise attentive head-pose frames:

```bash
poetry run python -m scripts.labs.face_capture_lab --mode attention --frames 40 --interval-sec 0.5 --attention-eval-raw
```

Audio detection capture:

```bash
poetry run python -m scripts.labs.audio_detection_lab --clips 10
poetry run python -m scripts.labs.audio_detection_lab --audio-file /path/to/clip.wav
```

After capture, edit the generated `labels.todo.jsonl` and fill only the label
fields you know. Then run eval:

```bash
poetry run python -m scripts.eval.perception_eval --run-dir var/labs/face/enrollment/<run_id>
poetry run python -m scripts.eval.perception_eval --run-dir var/labs/audio/detection/<run_id>
```

Eval produces:

- `eval_report.md`
- `eval_report.json`
- `metrics.csv`
- `failures.csv`
- `threshold_sweeps.csv`

Ground truth is manual for v1: raw camera/audio artifacts do not contain recall
labels by themselves. The lab predicts; you label; eval computes metrics.

Registration quality dry run:

```bash
poetry run python -m scripts.labs.face_registration_lab --frames 5
```

Dry-run and enrollment previews are saved under `scripts/labs/face_preview` by
default. Use `--preview-dir /path/to/dir` to choose another folder.

With depth enabled, each diagnostic frame waits until a synced RGBD pair arrives.
Use `--max-frame-wait-sec 10` only if you want the helper to give up instead of
waiting indefinitely.

Current registration tuning defaults match the production agent:
- `min_face_area=5000`
- `min_sharpness=12`
- `min_brightness=35`
- `min_contrast=15.5`
- `max_nose_center_offset=0.1`

Use `--details` when you want the full diagnostic dump.

Actual enrollment:

```bash
poetry run python -m scripts.labs.face_registration_lab --name "Jane Doe" --enroll
```

Recognition once:

```bash
poetry run python -m scripts.labs.face_recognition_lab --once
```

Recognition loop:

```bash
poetry run python -m scripts.labs.face_recognition_lab --loop --interval 0.5
```

Speaker enrollment to a temporary lab DB:

```bash
poetry run python -m scripts.labs.speaker_recognition_lab enroll --person-id person_me --clips 3
```

Speaker recognition against that lab DB:

```bash
poetry run python -m scripts.labs.speaker_recognition_lab recognize --clips 1
```

Each speaker attempt also saves a JSON report with:
- the effective profile/policy config used for that run
- raw vs trimmed clip stats
- VAD frame counts and frame-RMS summaries
- explicit diagnostics such as trim fallback, VAD mismatch, quiet clips, and borderline matches

List the temporary lab references:

```bash
poetry run python -m scripts.labs.speaker_recognition_lab list
```

Employee-directory registration probe from the microphone:

```bash
# Agent-style registration probe: wait for "Listening...", say your name once,
# print the transcript, the actual Realtime tool-call args, and the employee-directory
# match result, then exit.
poetry run python -m scripts.labs.rapidfuzz_employee_lab --sites bos1,bos3

# Same as above, but keep listening until Ctrl+C.
poetry run python -m scripts.labs.rapidfuzz_employee_lab --sites bos1,bos3 --loop

# If Snowflake stores Latin-script names and multilingual ASR is returning a
# native-script transcript, force English transcription for this lab run.
poetry run python -m scripts.labs.rapidfuzz_employee_lab --sites bos1,bos3 --language en
```

One-off OpenAI speech without starting the realtime agent:

```bash
# Playback defaults to the local pipewire sounddevice output.
poetry run python -m scripts.labs.openai_say_lab "Hello from Puffle." --play

# Save an mp3 instead of playing locally.
poetry run python -m scripts.labs.openai_say_lab "Back in five minutes." --format mp3

# Tune the delivery.
poetry run python -m scripts.labs.openai_say_lab \
  "The lab speaker is ready." \
  --voice marin \
  --instructions "Sound warm, concise, and a little excited." \
  --play
```
