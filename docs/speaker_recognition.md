# Argos Speaker Recognition and Voice Enrollment

Read this with:

- `argos_src/agent/agent_audio.py`
- `argos_src/agent/agent_runtime.py`
- `argos_src/agent/agent_tools.py`
- `argos_src/speaker_recognition/service.py`
- `argos_src/speaker_recognition/policy.py`
- `argos_src/speaker_recognition/backend.py`
- `argos_src/identity/embeddings/speaker_store.py`
- `argos_src/speaker_recognition/manage_voice.py`

This document explains the Argos speaker path:

1. how audio turns become speaker-owned turns
2. how post-registration voice enrollment works
3. what preprocessing happens before ECAPA embeddings
4. how audio ownership and strict face ownership interact
5. how to inspect saved voice references and delete a whole identity

## Mental Model

Speaker recognition is not a separate always-on diarization service.

It is a turn-owned helper around the normal realtime audio path:

```text
microphone audio
    -> resample to 16 kHz mono
    -> local VAD + wake/admission
    -> commit one audio turn
    -> keep a speaker-specific 16 kHz turn clip
    -> ECAPA embedding
    -> compare against saved voice references
    -> combine with face visibility
    -> freeze owner_id onto the turn
```

Voice enrollment is also turn-owned:

```text
successful face enrollment
    -> arm pending voice enrollment
    -> next clean spoken turn from that person
    -> quality gate
    -> ECAPA embedding
    -> save one reusable voice reference
```

## Main Components

| File | Responsibility |
|---|---|
| `argos_src/agent/agent_audio.py` | Captures mic audio, resamples to 16 kHz, buffers turn audio, and triggers speaker resolution at audio commit. |
| `argos_src/agent/agent_runtime.py` | Owns pending voice-enrollment state, audio ownership logs, and post-turn voice enrollment. |
| `argos_src/agent/agent_tools.py` | Arms pending voice enrollment after `enroll_visible_person` succeeds. |
| `argos_src/speaker_recognition/service.py` | Main orchestration layer for query embedding lookup and voice reference storage. |
| `argos_src/speaker_recognition/policy.py` | Audio preprocessing, clip stats, quality gates, and owner-resolution rules. |
| `argos_src/speaker_recognition/backend.py` | SpeechBrain ECAPA backend wrapper. |
| `argos_src/identity/embeddings/speaker_store.py` | Persistent ChromaDB storage for one voice reference embedding per `person_id`. |
| `argos_src/identity/store.py` | Shared identity store used after speaker ownership resolves. |
| `argos_src/memory/store.py` | Source-aware social/context memory store keyed by `person_id` or site. |
| `argos_src/speaker_recognition/manage_voice.py` | CLI for listing and showing saved voice references. |

## Query Flow

### 1. Audio capture and turn creation

`RealtimeAgentAudioMixin._capture_callback()` resamples the live mic stream to
`16 kHz` mono for local VAD and speaker processing.

When local end-of-speech fires, `_commit_audio_turn()`:

- builds the `QueuedTurn`
- freezes the strict `primary_face_person_id` and visible face ids from speech start
- prepares a speaker-specific 16 kHz clip for speaker recognition
- blocks on `SpeakerRecognitionService` to resolve the turn owner before creating
  the response

### 2. Speaker preprocessing

The current speaker preprocessing is intentionally simple:

- keep the original turn audio for the realtime API
- build a speaker-specific clip from the locally buffered `16 kHz` PCM
- currently pass the full committed clip to speaker recognition

`SpeakerRecognitionService.trim_turn_audio()` can trim with a supplied VAD, but
the live commit path passes `vad=None` because the capture VAD is not reused from
the commit thread. In that path, duration and RMS gates apply to the full
speaker clip, not to a separately packed voiced-only clip.

This is not a full diarization pipeline and does not do:

- source separation
- aggressive denoising
- chunk averaging at query time
- multi-speaker segmentation

### 3. Query quality gate

For query ownership, Argos only enforces a minimum speaker-clip duration:

- `query_min_voiced_sec`

The default profile uses `0.8s`.

Short queries like "hi" or "hello" are therefore still allowed when they clear
that duration floor.

### 4. Owner resolution

After embedding and scoring, the runtime combines audio evidence with the face scene.

Possible outcomes:

- `audio`
  Strong audio match wins.
- `audio_face_agree`
  Audio matches the current visible primary face strongly enough.
- `face`
  Audio evidence is weak, but the frozen face scene has exactly one usable
  recognized face.
- `unknown`
  Audio is weak and there is no strict single-face owner candidate.

This is why a later turn can move between:

- strong audio ownership
- strict single-face ownership
- fully unresolved ownership

depending on clip quality and face visibility.

## Voice Enrollment Flow

### 1. Enrollment is armed by face registration

When `enroll_visible_person` succeeds, `agent_tools.py` arms a pending voice
enrollment target for that `person_id`.

That does not save voice immediately. It only tells the runtime:

"the next clean spoken turn from this person can be used to save a reusable voice reference."

### 2. Enrollment happens on a later spoken turn

`agent_runtime.py` attempts voice enrollment only after a real spoken turn completes.

The turn must have:

- one clean recognized face scene
- turn audio available
- a pending voice enrollment target with no saved reference yet

### 3. Enrollment quality gates

The enrollment path is stricter than the query path.

The current default gates are:

- minimum voiced duration: `2.0s`
- maximum voiced duration: uncapped by default
- minimum RMS level: `350.0`
- maximum clipped fraction: `0.02`

If the clip passes, Argos saves exactly one normalized voice reference embedding
for that person.

## Storage Model

The speaker embedding store keeps one reusable reference embedding per `person_id`.

Metadata saved alongside it includes:

- `model_name`
- `created_at`
- `query_duration_s`
- `rms_level`
- `clip_count`
- `total_voiced_sec`
- `mean_rms_level`

Important detail:

- saving a new reference for the same `person_id` updates the stored centroid
- repeated consistent enrollment clips are averaged together

## Current Preprocessing Choices

The current path is intentionally conservative and small:

- required `16 kHz` mono speaker clip
- full committed speaker clip in the live runtime
- no transcript dependency
- no hard max duration by default

What is not in the current code:

- speaker embedding smoothing across several query turns
- extra local normalization beyond the RMS/clipping checks
- full diarization or source separation

## Operational Notes

### Backend prewarm and first-use model load

`factory.py` calls `SpeakerRecognitionService.prewarm()` during startup when
speaker recognition is enabled. If the backend exposes a prewarm hook and the
model is cached, this moves model load cost out of the first live turn. If
prewarm fails or the model is not cached yet, the first live enrollment or query
can still pay for:

- model load
- first inference warmup
- Hugging Face model download if not cached

After that, later queries are much cheaper.

### Strict face ownership still matters

Speaker recognition does not replace face recognition.

When audio confidence is weak, exactly one usable recognized face in the frozen
speech-start scene is what allows face ownership instead of dropping to
`unknown`.

## Managing Saved Voice References

List saved voice references:

```bash
cd ~/argos-agent
source setup_shell.sh
python3 -m argos_src.speaker_recognition.manage_voice --list
```

Show one saved reference:

```bash
python3 -m argos_src.speaker_recognition.manage_voice --show "Your Name"
```

Delete one person and all linked embeddings:

```bash
python3 -m argos_src.identity.manage_identity --delete "Your Name"
python3 -m argos_src.identity.manage_identity --delete person_your_name_20260505_123456 -y
```

Both management CLIs accept either:

- the stored `person_id`, or
- the human name from the identity store

## Useful Logs

The main runtime lines to watch are:

- `Speaker recognition has no enrolled voice references yet`
- `Speaker resolution initial ...`
- `Voice enrollment attempting ...`
- `Voice enrollment saved ...`
- `Voice enrollment skipped ...`

Those tell you whether the current problem is:

- no saved voice reference yet
- weak query ownership
- enrollment quality rejection
- or a successful save followed by later audio matching
