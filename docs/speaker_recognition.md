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
    -> use raw 16 kHz turn audio for speaker use
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
    -> empty/clipping guard
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
| `argos_src/speaker_recognition/policy.py` | Clip stats, minimal safety gates, and owner-resolution rules. |
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
- uses raw locally buffered `16 kHz` PCM for speaker recognition
- blocks on `SpeakerRecognitionService` to resolve the turn owner before creating
  the response

### 2. Speaker preprocessing

The current production speaker path intentionally does not preprocess the
speaker clip beyond using locally buffered `16 kHz` mono PCM:

- keep the original turn audio for the realtime API
- use the same raw locally buffered `16 kHz` PCM for ECAPA speaker embedding
- do not VAD-trim, denoise, normalize, or duration-gate before embedding

This is not a full diarization pipeline and does not do:

- VAD-based voiced-region packing
- source separation
- aggressive denoising
- chunk averaging at query time
- multi-speaker segmentation

This default came from `argos-perception-eval` cross validation on collected
robot audio: raw ECAPA embeddings with four-clip person centroids produced
44/45 correct rankings. A conservative `top_score >= 0.5` and `margin >= 0.3`
rule had no false accepts but rejected many correct clips, so the runtime uses
`top_score >= 0.4` and `margin >= 0.2` as the starting policy.

### 3. Query gate

For query ownership, Argos now allows any captured speaker turn to be embedded.
The default profile sets:

- `query_min_voiced_sec: 0.0`
- `query_match_threshold: 0.40`
- `query_margin_threshold: 0.20`

`resolve_owner_id()` requires the top audio match to clear both the score
threshold and the margin threshold. If audio is ambiguous but the face scene has
a strict owner candidate, ownership falls back to face rather than accepting the
ambiguous audio match.

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

### 3. Enrollment safety gates

The enrollment path now avoids arbitrary duration and loudness gates. The
current default gates are:

- reject empty audio
- maximum voiced duration: uncapped
- minimum RMS level: disabled by default (`0.0`)
- maximum clipped fraction: `0.02`

If the clip passes, Argos saves or updates one normalized voice reference
centroid for that person.

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
- raw captured turn audio for speaker embedding
- no transcript dependency
- no duration or RMS gate by default

What is not in the current code:

- speaker embedding smoothing across several query turns
- extra local normalization
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
