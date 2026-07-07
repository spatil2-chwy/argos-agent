# Attention Gate

This document explains how Argos decides whether a visible face is attending to
the robot, and how that decision opens passive microphone admission.

Read this with:

- `argos_src/face_recognition/attention_gate/gate.py`
- `argos_src/face_recognition/face_recognition_service.py`
- `argos_src/face_recognition/presence_cache.py`
- `argos_src/runtime/audio_admission.py`
- `argos_src/agent/control/audio_runtime.py`
- `config/profiles/static_interaction.yaml`

## What It Is

The attention gate is a lightweight filter on top of face detection. It answers:

```text
Is any visible person probably addressing the robot right now?
```

It is not identity recognition, speaker recognition, or gaze tracking. It uses
the existing face detector plus a head-pose model, then publishes a compact
`attention_status` that audio admission can consume.

In the static mounted-camera profile, passive mic admission is configured to open
on attention, not merely on face presence:

```yaml
realtime:
  admission:
    open_on_face_presence: false
    open_on_attention_presence: true
```

That means a person can be visible in the wide RealSense view without opening
the mic unless their face also passes the attention gate.

## End-to-End Flow

The runtime loop is:

```text
camera frame
  -> MTCNN face detection
  -> optional depth filtering
  -> FaceNet embedding / identity match
  -> SixDRepNet head pose on each usable face crop
  -> attention gate thresholds
  -> FacePresenceSnapshot(attention_status=...)
  -> FacePresenceGate local cache
  -> resolve_record_admission(...)
  -> microphone may start recording if VAD also sees speech
```

The attention gate runs after a face is already considered usable for the face
recognition scene. Enrollment has separate, stricter quality rules.

## Per-Face Decision

For each detected face, `FaceAttentionGate.evaluate(...)` does this:

1. Check that the gate is enabled.
2. Compute the detected face bbox area.
3. Reject very small faces before running head pose.
4. Run SixDRepNet on the face crop to estimate yaw, pitch, and roll.
5. Compare the pose against fixed yaw/pitch/roll limits.

The result is stored on the face as a `FaceAttentionObservation`:

```text
attentive
confidence
reason
yaw_deg / pitch_deg / roll_deg
raw_attentive
raw_confidence
```

`confidence` and `raw_confidence` are compatibility fields. They are binary:
`1.0` when the attention decision is true and `0.0` otherwise.

Important reasons:

| Reason | Meaning |
|---|---|
| `attentive` | Face passed pose and size checks. |
| `face_too_small` | Face bbox is below the configured minimum. |
| `head_pose_outside_threshold` | Yaw, pitch, or roll exceeded the configured limit. |
| `sixdrepnet_unavailable` | The head-pose model could not be initialized. |

## Size And Pose

The gate uses one absolute bbox-area threshold and one fixed pose threshold set.
It does not vary pose limits by distance or reject faces based on how far their
center is from the camera optical axis.

```yaml
attention_gate:
  min_face_area: 1300
  max_abs_yaw_deg: 20.0
  max_abs_pitch_deg: 18.0
  max_abs_roll_deg: 90.0
  min_abs_pitch_deg: 0.0
```

The intent is:

- reject very small crops before spending time on head pose
- keep yaw and pitch strict enough to avoid obvious side conversations
- keep roll permissive so head tilt rarely blocks otherwise plausible attention

## Scene-Level Attention

After all faces are evaluated, scene analysis counts attentive faces and selects
targets:

- `attention_status = "attentive"` when at least one face is attentive
- `attention_status = "inattentive"` when faces are visible but none are attentive
- `attention_status = "none"` when no faces are visible

The presence snapshot also carries counts and primary-attention metadata:

```text
attention_count
attentive_recognized_count
attentive_unknown_count
primary_attention_kind
primary_attention_name
primary_attention_person_id
attention_confidence
```

`primary_attention_person_id` is only populated when there is exactly one
attentive recognized face. Unknown attentive people can still open passive mic
admission, but they do not become a known identity owner.

## Microphone Admission

Attention is an admission signal, not an audio stop signal.

When recording is not active, the audio callback calls
`resolve_record_admission(...)`. In the static profile, admission opens if:

- the robot is not blocked by speaking/playback guard
- focused navigation policy allows listening
- `open_on_attention_presence` is true
- the latest face snapshot says `attention_status == "attentive"`

Admission opening alone does not start a turn. Voice still has to be detected by
local VAD:

```text
attention present + VAD positive -> recording can start
attention present + silence -> no turn starts
```

Once recording has started, loss of attention does not immediately stop the
recording. Active capture ignores admission and continues appending audio until
local VAD has been silent for `silence_grace_period`. This is deliberate:

- people move their head while speaking
- attention estimates can flicker frame to frame
- cutting off active speech would be worse than letting VAD finish the turn

So the rule is:

```text
attention controls whether passive recording may start
VAD/silence controls when active recording ends
```

Wake word and selected interaction states can also open admission depending on
profile settings. In the static interaction profile, `alert` can open admission
by state, while `cooldown` does not open the mic by itself; a cooldown
follow-up still needs attention or wake word.

## Why This Design

For a mounted RealSense camera, this is the most logical conservative heuristic
given the current stack:

- face detection tells us a person is visible
- head pose tells us whether their face is broadly oriented toward the robot
- VAD prevents silent visual attention from becoming a turn

The design deliberately avoids pretending that head pose is exact gaze. It only
answers whether the person is plausibly addressing the robot. That matches the
robot interaction use case better than trying to classify eye contact precisely
from a small wide-angle face crop.

Depth is useful for separate face-depth gating, but the attention gate itself no
longer uses depth or bbox ratio to alter pose thresholds.

The main remaining limitation is calibration. The best thresholds depend on:

- camera resolution and field of view
- camera mounting height
- robot height relative to a standing person
- the distance where humans naturally talk to the robot
- lighting and face detector stability

For real deployment tuning, inspect attention logs with:

```text
reason, yaw, pitch, roll, bbox area, depth_m
```

Then tune in this order:

1. `min_face_area` if valid distant faces are rejected.
2. `max_abs_yaw_deg` if side conversations falsely open admission.
3. `max_abs_pitch_deg` if users facing the robot while looking up or down are rejected.
4. `max_abs_roll_deg` if tilted heads are incorrectly blocked or admitted.
