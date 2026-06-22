# Face Recognition, Enrollment, and Identity Flow

Read this with:

- `argos_src/face_recognition/face_recognition_service.py`
- `argos_src/face_recognition/pipeline.py`
- `argos_src/face_recognition/depth_gate.py`
- `argos_src/face_recognition/scene_analysis.py`
- `argos_src/face_recognition/presence_cache.py`
- `argos_src/tools/unitree_go2/vision/enroll_visible_person.py`
- `argos_src/tools/unitree_go2/vision/resolve_employee_identity.py`
- `argos_src/employee_directory/service.py`
- `argos_src/agent/agent_audio.py`
- `argos_src/agent/agent_runtime.py`
- `argos_src/agent/runtime_context.py`
- `argos_src/agent/agent_tools.py`

This document covers the current Argos face path:

1. live face enrollment
2. enrollment preprocessing and failure messages
3. continuous recognition preprocessing
4. multi-face scenes
5. face identity vs audio speaker ownership

For the dedicated attention-gate flow, thresholds, and mic-admission behavior,
see `docs/attention_gate.md`.
6. config knobs and threshold policy

## Short Answer

`primary_face_person_id` is the only face-derived owner candidate. It is set
only when the current depth-gated scene has exactly one usable recognized face.
If no recognized face, an unknown face, or multiple usable faces are present, it
is `None`.

The final spoken-turn owner is resolved later by speaker recognition in `argos_src/speaker_recognition/policy.py`, using:

- `primary_face_person_id` from the camera scene
- `audio_speaker_id` from voice matching
- visible face ids frozen at speech start

So the mental model is:

```text
strict single face -> primary_face_person_id
voice match        -> audio_speaker_id
owner policy       -> owner_id for the turn
```

## Main Components

| File | Responsibility |
|---|---|
| `face_recognition_service.py` | Background recognition loop, one-shot recognition, live enrollment, enrollment quality policy. |
| `pipeline.py` | MTCNN face detection and FaceNet embedding extraction. |
| `depth_gate.py` | Optional aligned-depth validation for detected faces. |
| `scene_analysis.py` | Applies strict single-face primary-owner selection and summarizes the social scene. |
| `presence_cache.py` | Stores current visible people, strict primary face identity, and `/go2/face_presence` snapshot. |
| `enroll_visible_person.py` | LLM tool wrapper for safe live enrollment. The LLM passes only name and optional username. |
| `resolve_employee_identity.py` | LLM tool for employee-directory lookup before enrollment. |
| `employee_directory/service.py` | Loads Snowflake rows and locally rehydrates the verified profile during enrollment. |
| `agent_tools.py` | Arms voice enrollment after face enrollment succeeds. |

## Live Enrollment Flow

Enrollment is not a CLI flow. It happens through the `enroll_visible_person`
tool during an Argos interaction.

The supported flow is:

```text
unknown visible person
    -> person confirms official first + last name
    -> resolve_employee_identity
    -> if identity is clear, enroll_visible_person(official_name, username?)
    -> FaceRecognitionService.enroll_visible_person()
    -> short camera burst
    -> quality and consistency gates
    -> if display.enabled is true and a screen resource is selected, show face-capture preview
    -> wait for Accept / Reject
    -> save averaged face embedding under person_id
    -> seed live presence cache
    -> agent_tools arms pending voice enrollment
```

The LLM still calls only `enroll_visible_person`. It does not call a display
tool. The review UI is handled inside the enrollment tool path.

The LLM-facing enrollment schema is intentionally small:

```python
class _EnrollVisiblePersonInput(BaseModel):
    official_name: str
    username: str | None = None
```

`enroll_visible_person.py` locally rehydrates employee metadata before calling the
face service:

```python
employee_profile = lookup(username=username or "", official_name=official_name)
result = self.face_service.enroll_visible_person(
    official_name=official_name,
    username=username or "",
    employee_profile=employee_profile,
    camera_resource_id=self.default_camera_resource,
    display_runtime=self.display_runtime,  # present only when configured
)
```

That means the LLM no longer has to pass title, manager, cost center, job family,
tenure, or similar fields back into the tool. It should pass the verified username
when `resolve_employee_identity` returned one. The rest is loaded locally from
`EmployeeDirectoryService.get_verified_profile()`.

## Enrollment Preprocessing

Enrollment is deliberately stricter than recognition because the saved embedding
becomes the long-term reference for that person.

`FaceRecognitionService.enroll_visible_person()` captures a short burst:

- `ENROLLMENT_BURST_FRAMES = 5`
- `ENROLLMENT_REQUIRED_STABLE_FRAMES = 3`
- `ENROLLMENT_BURST_SLEEP_SEC = 0.1`

For each burst frame:

1. Capture color, or synced color + depth when `depth_gate.enabled: true`.
2. Detect faces in `pipeline.py` with MTCNN.
3. Drop detections below `MIN_FACE_DETECTION_CONFIDENCE = 0.9`.
4. Optionally depth-gate detections in `depth_gate.py`.
5. Extract one FaceNet embedding per surviving face.
6. Select one enrollment face, rejecting significant extra faces.
7. Run enrollment quality checks.
8. Reject if this face already matches a known person.
9. Keep accepted frame embeddings for burst consistency.

The diagnostic preparation object is:

```python
@dataclass(frozen=True)
class FacePreparationResult:
    faces: list[dict[str, Any]]
    reason: str = ""
    detected_count: int = 0
    rejected_count: int = 0
```

`reason` is used when the face service detected something but could not produce a
usable embedding. Current reasons are:

- `no_detection`
- `depth_rejected`
- `no_embedding`

After the burst passes quality and consistency checks, the service prepares a
candidate object containing the averaged embedding, durable metadata, reference
face, image shape, and preview image. The preview image is a padded crop derived
from the selected reference face bbox, so the review screen shows the face region
used for embedding with a little surrounding context. The candidate is not saved
yet.

When `display.enabled` is true and `resources.interaction_display` selects a
screen resource, Argos encodes the preview image as a `data:image/png;base64,...` URL
and sends a blocking `face_capture_preview` command to the Puffle display. Only
an Accept response commits the candidate to the face store. Reject, timeout, or
display-unavailable responses return a failed tool result and do not save
anything.

## Enrollment Quality Policy

The enrollment quality gates live in `FaceEnrollmentPolicy` in
`face_recognition_service.py`:

```python
@dataclass(frozen=True)
class FaceEnrollmentPolicy:
    min_face_area: int = 5000
    min_sharpness: float = 12.0
    min_brightness: float = 35.0
    max_brightness: float = 220.0
    min_contrast: float = 15.5
    max_eye_tilt: float = 0.25
    max_nose_center_offset: float = 0.10
    min_embedding_similarity: float = 0.70
```

`_assess_enrollment_face_quality()` can reject a frame for:

| Reason | Meaning | User guidance |
|---|---|---|
| `face_too_small` | Face bbox area is below `5000` px. | Come closer. |
| `face_clipped` | Face touches image boundary. | Center whole face in view. |
| `missing_landmarks` | Required eye, nose, or mouth landmarks are missing. | Face the camera directly. |
| `side_face` | Eye tilt or nose offset suggests a profile/angled face. | Face the camera directly. |
| `too_blurry` | Gradient-based sharpness below `12.0`. | Hold still. |
| `too_dark` | Mean crop brightness below `35.0`. | Move to better light. |
| `too_bright` | Mean crop brightness above `220.0`. | Move away from bright light. |
| `low_contrast` | Crop contrast below `15.5`. | Move to better light. |
| `embedding_inconsistent` | Fewer than 3 accepted burst embeddings are mutually similar enough. | Hold still and face the camera. |

The final saved face reference is the average of normalized embeddings from
consistent accepted frames. This is useful because one noisy frame should not
become the person's permanent reference.

## Enrollment Exit Points and LLM Tool Payloads

The model does not receive raw images, depth samples, embeddings, or internal
metrics. It receives the JSON string returned by `tool_response_json()` from
`enroll_visible_person.py`.

Every enrollment response has:

```json
{
  "success": false,
  "status": "...",
  "message": "..."
}
```

Every failure includes `failure_reason`. Some responses also include
`recognized_name` or `person_id`.

| Status | Failure reason | When it happens | What the LLM sees |
|---|---|---|---|
| `error` | `missing_name` | Name was missing before camera work. | `"I still need your name before I can save a new face."` |
| `error` | `capture_failed` | No color frame, or no synced RGBD pair when depth is enabled. | `"I couldn't get a clear camera view right now. Please try again in a moment."` |
| `retry_single_face` | `multiple_faces` | More than one significant face survived preprocessing. | `"I can still see more than one face..."` |
| `retry_quality` | `depth_rejected` | Face was detected, but depth samples were invalid or farther than allowed. | Message asks for a closer face view within about two meters. |
| `retry_quality` | `no_embedding` | Face survived detection but embedding extraction failed. | Message asks the person to face the camera and hold still. |
| `retry_quality` | `no_detection` | No face reached usable detection during the burst. | Generic stable-face guidance. |
| `retry_quality` | `unstable_face` | A single face was intermittently seen, but not enough stable accepted frames were collected and no more specific reason was available. | Generic stable-face guidance. |
| `retry_quality` | quality reason | Blur, lighting, contrast, clipped face, missing landmarks, side face, or too-small face. | The matching guidance from `_quality_response_for_reason()`. |
| `retry_quality` | `embedding_inconsistent` | At least some frames passed, but not enough embeddings agreed. | `"Hold still and face me directly for a second."` |
| `retry_already_known` | `already_known` | Accepted face already matches an enrolled person. | Includes `recognized_name`. |
| `display_unavailable` | `display_unavailable` or `preview_encoding_failed` | A display review was required but the screen was unavailable or the preview could not be encoded. | Says the screen is not ready and asks to try again. |
| `review_timeout` | `review_timeout` | The display preview was shown, but no Accept/Reject response arrived in time. | Says the review timed out and asks whether to retry. |
| `user_rejected_preview` | `user_rejected_preview` | The person rejected the face-capture preview. | Confirms the capture was not saved and asks whether to retry. |
| `enrolled` | none | Enrollment succeeded. | Includes `person_id` and `next_step_hint`. |

On success, the tool returns:

```json
{
  "success": true,
  "status": "enrolled",
  "message": "You're all set, NAME. I'll remember you next time.",
  "person_id": "person_...",
  "next_step_hint": "Now continue with one short social follow-up..."
}
```

Then `agent_tools.py` sees the successful `person_id` and calls
`_arm_pending_voice_enrollment(person_id)`.

## Recognition Preprocessing

Recognition is intentionally lighter than enrollment. It needs to work every
`face_recognition.loop_interval_sec` seconds without blocking conversation.

The background loop in `_loop_tick()` does:

```text
capture frame
    -> prepare faces
    -> recognize each embedding
    -> analyze scene
    -> update FacePresenceCache
```

Recognition preparation uses the same lower-level detection, optional depth gate,
and embedding extraction as enrollment:

```python
detected_faces = self.detect_faces(image)
gated_faces, rejected_count = filter_detections_by_depth(...)
embedding = self.extract_face_embedding(image, detection)
```

What recognition does not do:

- no blur gate
- no brightness gate
- no contrast gate
- no frontal-face enrollment policy
- no 5-frame burst consistency check

That split is intentional. Enrollment should save only good references.
Recognition should still recognize a real person when they move, turn slightly, or
stand in imperfect office lighting.

If a recognition tick has no usable faces, the loop logs the preparation reason
and lets the presence cache expire after `CACHE_EXPIRE_SEC`. The LLM is not sent
a failure message for normal background recognition misses. It simply gets no
fresh recognized-person context after the cache expires.

The cache also enforces expiry on read. If the face loop has not cleared stale
state yet, calls such as `get_cached_persons()` and
`get_primary_face_person_id()` still return empty/`None` once the cache window
has expired.

The one-shot `recognize_faces()` API does return `failure_reason` for no usable
faces. That is useful for debugging, but it is not the normal conversational path.

## Multiple Faces

Enrollment and recognition handle multiple faces differently.

Enrollment wants exactly one visible person:

- one face is accepted
- duplicate/ghost detections overlapping the primary can be ignored
- tiny, weak extra detections can be ignored
- significant extra faces return `status="retry_single_face"` and
  `failure_reason="multiple_faces"`

Recognition supports multiple faces:

- each usable face is embedded
- each embedding is matched independently
- recognized people become `PersonContext` rows
- unrecognized usable faces increment `unknown_count`
- `primary_face_person_id` is set only for exactly one usable recognized face

`scene_analysis.py` keeps ownership strict:

- exactly one usable recognized face -> `primary_face_person_id`
- zero usable recognized faces -> `None`
- more than one usable face, recognized or unknown -> `None`

## Face Target vs Turn Owner

At recording start, `agent_audio.py` snapshots the strict face id and the
visible face id set from the same moment:

```python
self._current_primary_face_person_id = self._get_current_primary_face_person_id()
self._current_visible_face_person_ids = self._get_current_visible_face_person_ids()
```

`_current_primary_face_person_id` is the only face-derived owner candidate.
`_current_visible_face_person_ids` is used to corroborate voice matches against
the same frozen scene.

At audio commit, the queued turn carries:

- `primary_face_person_id`
- `audio_speaker_id`
- `owner_id`
- `owner_source`
- `owner_confidence`

`primary_face_person_id` means "the one recognized face owner candidate when the
turn started." If the scene is ambiguous, it is `None`.

`owner_id` means "the resolved owner of this spoken turn." This is the id used
for live-chat preference extraction and `MemoryStore` ownership.

The canonical face-derived name in turn state is `primary_face_person_id`.

## Config Knobs

Main config lives in:

`config/profiles/static_interaction.yaml`

```yaml
face_recognition:
  enabled: true
  loop_interval_sec: 0.3
  recognition_threshold: 0.6
  depth_gate:
    enabled: true
    sync_slop_sec: 0.12
    sync_queue_size: 10
    capture_timeout_sec: 1.5
    max_face_depth_m: 2.0
    min_valid_samples: 2
    patch_size: 3
    search_radius_px: 12
    max_valid_depth_m: 10.0
  attention_gate:
    enabled: true
    min_face_area: 700
    min_face_area_ratio: 0.00035
    max_abs_yaw_deg: 25.0
    max_abs_pitch_deg: 22.0
    max_abs_roll_deg: 35.0
    distant_max_abs_yaw_deg: 18.0
    distant_max_abs_pitch_deg: 32.0
    distant_max_abs_roll_deg: 28.0
    near_face_area_ratio: 0.035
    distant_face_area_ratio: 0.010
    near_depth_m: 0.8
    distant_depth_m: 2.0
    max_center_offset_ratio: 0.70
    min_confidence: 0.55
    smoothing_window_sec: 1.0
    min_attentive_observations: 2
    hold_sec: 0.8
  proactive_greeting:
    require_attention: true
```

Depth gate settings:

| Setting | Meaning | Keep it? |
|---|---|---|
| `sync_slop_sec: 0.12` | Max timestamp mismatch allowed between color and depth frames. | Yes if depth gate is enabled. |
| `sync_queue_size: 10` | Number of color/depth messages buffered while looking for a synchronized pair. | Yes if depth gate is enabled. |
| `capture_timeout_sec: 1.5` | Max wait for a synced RGBD pair. | Yes if depth gate is enabled. |
| `max_face_depth_m: 2.0` | Face median depth must be at or below this distance. | Tune based on desired interaction range. |
| `min_valid_samples: 2` | At least this many landmark/center points need valid depth. | Keep unless the camera often has sparse depth. |
| `patch_size: 3` | Median sample window around each landmark or face center. Must be odd. | Keep. |
| `search_radius_px: 12` | Nearby-pixel search radius when the exact landmark pixel has invalid depth. | Keep or raise slightly if depth holes are common. |
| `max_valid_depth_m: 10.0` | Raw depth sanity cap before computing face depth. This is not the face range gate. | Keep. |

If the depth camera is reliable, these settings are valuable because they reject
posters, far background faces, and stale visual detections. If real nearby people
are often rejected, tune in this order:

1. raise `max_face_depth_m` from `2.0` to `2.5` or `3.0`
2. raise `search_radius_px` if depth holes are common
3. lower `min_valid_samples` only if the aligned depth image is consistently sparse
4. disable `depth_gate.enabled` only if depth sync is unreliable in the deployment

Attention gate settings:

| Setting | Meaning | Keep it? |
|---|---|---|
| `enabled: true` | Runs a lightweight head-pose gate after usable face detection. | Yes for attention-gated admission. |
| `min_face_area: 700` | Absolute minimum detected face bbox area before head-pose scoring. | Lower for mounted wide-view cameras; keep enrollment stricter. |
| `min_face_area_ratio: 0.00035` | Resolution-scaled face area floor, combined with `min_face_area`. | Tune with camera resolution. |
| `max_abs_yaw_deg: 25.0` | Near-face left/right head angle limit. | Tune for close interaction. |
| `max_abs_pitch_deg: 22.0` | Near-face up/down head angle limit. | Tune with camera mounting height. |
| `max_abs_roll_deg: 35.0` | Near-face head tilt limit. | Usually keep. |
| `distant_max_abs_yaw_deg: 18.0` | Far/small-face left/right head angle limit. | Tighten if far side conversations falsely open admission. |
| `distant_max_abs_pitch_deg: 32.0` | Far/small-face up/down head angle limit. | Mounted cameras often need this higher because users look down toward the robot. |
| `distant_max_abs_roll_deg: 28.0` | Far/small-face head tilt limit. | Tune if tilted distant heads falsely count as attention. |
| `near_face_area_ratio: 0.035` | Face area ratio treated as near when depth is unavailable. | Tune from live bbox logs. |
| `distant_face_area_ratio: 0.010` | Face area ratio treated as distant when depth is unavailable. | Tune from live bbox logs. |
| `near_depth_m: 0.8` | Depth treated as near when `depth_m` exists on the face. | Keep unless the camera is mounted unusually close. |
| `distant_depth_m: 2.0` | Depth treated as distant when `depth_m` exists on the face. | Match the natural standing interaction distance. |
| `max_center_offset_ratio: 0.70` | Rejects faces far from the optical center for attention. | Mounted wide cameras should be looser than webcams. |
| `min_confidence: 0.55` | Confidence floor reported at the configured pose/center acceptance boundary. | Usually keep. |
| `smoothing_window_sec: 1.0` | Rolling window used to reduce flicker. | Usually keep. |
| `min_attentive_observations: 2` | Number of positive observations needed in the window. | Lower only if latency is too high. |
| `hold_sec: 0.8` | Keeps attention briefly after a positive window. | Tune for conversational continuity. |

The attention gate uses 6DRepNet on the existing MTCNN face crops. It does not
replace FaceNet and does not add a second face detector. Pose limits are
interpolated between near and distant settings using face `depth_m` when present,
or face bbox area ratio when depth is unavailable. This lets a mounted RealSense
camera treat a person standing around two meters back differently from a close
webcam-like face crop. If `sixdrepnet` is not
installed or the model cannot initialize, attention returns
`sixdrepnet_unavailable` and passive attention admission remains closed. The
presence snapshot keeps the old face fields and adds:

- `attention_status`
- `attention_count`
- `attentive_recognized_count`
- `attentive_unknown_count`
- `primary_attention_person_id`
- `primary_attention_name`
- `attention_confidence`

For passive listening, prefer:

```yaml
realtime:
  admission:
    open_on_face_presence: false
    open_on_attention_presence: true
```

## Thresholds Not Yet in YAML

Some thresholds are intentionally centralized in code but not yet profile-driven:

- `FaceEnrollmentPolicy` for enrollment quality
- `MIN_FACE_DETECTION_CONFIDENCE = 0.9` in `face_recognition/constants.py`
- presence cache expiry from `CACHE_EXPIRE_SEC` in `models.py`
- speaker ownership thresholds under `speaker_recognition:` in YAML

The clean direction is to keep the policy grouped and explicit. Moving every value
to YAML is useful once field testing shows which ones need runtime tuning.

## Is Preprocessing Overkill?

For enrollment, the current checks are reasonable. They protect the database from
bad permanent references:

- one person only
- close enough
- whole face visible
- frontal enough for landmarks
- not badly blurred
- not badly exposed
- enough contrast
- several consistent embeddings

For recognition, using all of those checks would be too harsh. The current code
does not do that, which is good.

If enrollment feels too strict during real testing, the first candidates to soften
are:

- `min_contrast`, because office lighting can make good faces look low contrast
- `min_brightness` and `max_brightness`, because camera auto-exposure varies
- `max_face_depth_m`, because two meters may be too short for natural approach

The checks I would keep strict are:

- `multiple_faces`
- `side_face`, especially `max_nose_center_offset`
- `face_clipped`
- `missing_landmarks`
- `embedding_inconsistent`

Those are the ones most likely to save the wrong identity or produce a weak
long-term reference.

## Useful Tests

Relevant focused tests:

```bash
poetry run pytest \
  tests/argos_src/test_employee_directory_service.py \
  tests/argos_src/face_recognition/test_face_recognition_service.py \
  tests/argos_src/tools/unitree_go2/vision/test_enroll_visible_person_tool.py \
  tests/argos_src/speaker_recognition/test_policy.py \
  tests/argos_src/speaker_recognition/test_service.py \
  tests/argos_src/agent/test_agent_runtime.py
```

## Useful Commands

Run standalone registration diagnostics without saving:

```bash
cd ~/argos-agent
source setup_shell.sh
poetry run python -m scripts.labs.face_registration_lab --frames 5
```

Try different preprocessing thresholds:

```bash
poetry run python -m scripts.labs.face_registration_lab \
  --min-contrast 12 \
  --min-brightness 30 \
  --max-face-depth-m 2.5
```

Actually save an enrolled face from the helper:

```bash
poetry run python -m scripts.labs.face_registration_lab \
  --name "Your Name" \
  --enroll
```

Run standalone recognition once:

```bash
poetry run python -m scripts.labs.face_recognition_lab --once
```

Run recognition in a loop and include enrollment-quality metrics for each face:

```bash
poetry run python -m scripts.labs.face_recognition_lab \
  --loop \
  --interval 0.5 \
  --include-enrollment-quality
```

Start the supported Argos profile:

```bash
cd ~/argos-agent
python3 run_profile.py --profile static_interaction
```

Manage identities:

```bash
cd ~/argos-agent
python3 -m argos_src.identity.manage_identity --list
python3 -m argos_src.identity.manage_identity --show "Your Name"
python3 -m argos_src.identity.manage_identity --delete "Your Name"
```

`argos_src.identity.manage_identity --delete` removes the identity row plus linked face
and speaker embeddings.
