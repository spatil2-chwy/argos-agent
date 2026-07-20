# Argos Speaker Recognition and Voice Enrollment

Read this with:

- `argos_src/agent/control/audio_runtime.py`
- `argos_src/agent/agent_runtime.py`
- `argos_src/agent/control/tool_runtime.py`
- `argos_src/speaker_recognition/service.py`
- `argos_src/speaker_recognition/policy.py`
- `argos_src/speaker_recognition/backend.py`
- `argos_src/identity_memory/tailwag_http.py`

This document explains the Argos speaker path:

1. how audio turns become speaker-owned turns
2. how post-registration voice enrollment works
3. what preprocessing happens before ECAPA embeddings
4. how audio ownership and strict face ownership interact
5. where saved voice references live

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
| `argos_src/agent/control/audio_runtime.py` | Captures mic audio, resamples to 16 kHz, buffers turn audio, and triggers speaker resolution at audio commit. |
| `argos_src/agent/agent_runtime.py` | Owns pending voice-enrollment state, audio ownership logs, and post-turn voice enrollment helpers. |
| `argos_src/agent/control/tool_runtime.py` | Arms pending voice enrollment after `enroll_visible_person` succeeds. |
| `argos_src/speaker_recognition/service.py` | Main orchestration layer for query embedding lookup and voice reference storage. |
| `argos_src/speaker_recognition/policy.py` | Clip stats, minimal safety gates, and owner-resolution rules. |
| `argos_src/speaker_recognition/backend.py` | SpeechBrain ECAPA backend wrapper. |
| `argos_src/identity_memory/tailwag_http.py` | Calls Tailwag for voice search, voice enrollment, owner resolution, and prompt context. |

## Query Flow

### 1. Audio capture and turn creation

`AudioRuntime._capture_callback()` resamples the live mic stream to
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

This raw-audio default came from `argos-perception-eval` cross validation on
collected robot audio. Current production thresholding is owned by Tailwag, not
by local Argos profile keys.

### 3. Query scoring

For query ownership, Argos embeds any captured speaker turn with audio and calls
Tailwag `search_voice(..., limit=2)`.

Tailwag owns:

- voice thresholds
- margin policy
- consent and archived-person filtering
- whether a voice candidate is recognized or rejected

### 4. Owner resolution

After voice search, Argos passes Tailwag:

- the recognized voice candidate, if any
- top score, runner-up score, margin, status, and reason
- the strict primary face candidate frozen at speech start
- visible face candidates frozen at speech start

Tailwag `resolve_turn_owner()` combines that evidence and returns the final
owner result. If Tailwag is unavailable or errors, Argos falls back to strict
face-only ownership.

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

Resolved outcomes are projected into the authoritative
`[PERSON SPEAKING TO YOU — IDENTITY RESOLVED]` prompt block. The block labels
`audio` as a trusted voice match, `face` as a trusted face match, and
`audio_face_agree` as a trusted face and audio match.

This is why a later turn can move between:

- strong audio ownership
- strict single-face ownership
- fully unresolved ownership

depending on clip quality and face visibility.

## Voice Enrollment Flow

### 1. Enrollment is armed by face registration

When `enroll_visible_person` succeeds, `ToolRuntime` arms a pending voice
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

The current enrollment gates are:

- reject empty audio
- maximum clipped fraction: `0.02`
- Tailwag may accept, merge, or reject repeated references

Argos does not locally compare new voice embeddings against existing references.
If the clip passes local gates, Argos sends the normalized embedding to Tailwag.

## Storage Model

Durable voice references live in `tailwag-memory`, reached through
`identity_memory_client` and `argos_src/identity_memory/tailwag_http.py`.
Argos does not maintain a separate in-repo speaker-reference database.

When `SpeakerRecognitionService.try_store_reference()` enrolls a voice
reference, it sends:

- `person_id`
- normalized ECAPA `embedding`
- `consent_status: consented`

Metadata sent alongside it includes:

- `query_duration_s`
- `rms_level`
- `clipped_fraction`
- `attempt_kind`

Important detail:

- whether repeated enrollment updates are accepted, averaged, or rejected is
  owned by Tailwag's biometric store
- Argos still rejects empty clips and clips above `max_clipped_fraction` before
  sending an enrollment request

Adaptive voice updates are separate from first voice enrollment. If a person has
no voice reference yet, Argos skips adaptive voice updates so the pending
post-face-enrollment voice capture remains the first durable sample. Once a voice
reference exists, Argos may offer a turn's ECAPA embedding to Tailwag only when
face and voice agree on the owner. Voice-only and face-only ownership do not
self-train the voice reference.

Tailwag owns the similarity threshold, running average, sample count, completion
state, and consent/status filtering for those updates.

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

Inspect and manage durable voice references with Tailwag tooling from
`tailwag-memory`. There is no current repo-local voice-management command.

## Useful Logs

The main runtime lines to watch are:

- `Speaker recognition has no identity-memory client; using strict face ownership.`
- `Speaker resolution initial ...`
- `Voice enrollment armed ...`
- `Voice enrollment attempting ...`
- `Voice enrollment audio stats ...`
- `Voice enrollment saved ...`
- `Voice enrollment skipped ...`

Those tell you whether the current problem is:

- Tailwag identity-memory integration is unavailable
- weak query ownership
- enrollment quality rejection
- or a successful save followed by later audio matching
