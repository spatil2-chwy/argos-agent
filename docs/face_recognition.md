# Face Recognition, Enrollment, and Identity Flow

Read this with:

- `argos_src/face_recognition/face_recognition_service.py`
- `argos_src/face_recognition/pipeline.py`
- `argos_src/face_recognition/depth_gate.py`
- `argos_src/face_recognition/scene_analysis.py`
- `argos_src/face_recognition/presence_cache.py`
- `argos_src/tools/unitree_go2/vision/enroll_visible_person.py`
- `argos_src/tools/unitree_go2/vision/resolve_employee_identity.py`
- `argos_src/identity_memory/tailwag_http.py`
- `argos_src/agent/control/audio_runtime.py`
- `argos_src/agent/agent_runtime.py`
- `argos_src/agent/runtime_context.py`
- `argos_src/agent/control/tool_runtime.py`

This document covers the current Argos face path:

1. live face enrollment
2. enrollment preprocessing and failure messages
3. continuous recognition preprocessing
4. multi-face scenes
5. face identity vs audio speaker ownership
6. config knobs and threshold policy

For the dedicated attention-gate flow, thresholds, and mic-admission behavior,
see `docs/attention_gate.md`.

## Short Answer

`primary_face_person_id` is the only face-derived owner candidate. It is set
only when the current depth-gated scene has exactly one usable recognized face.
If no recognized face, an unknown face, or multiple usable faces are present, it
is `None`.

The face loop also records recognition evidence for the strongest visible face
attempt: match status/reason, candidate person/name, similarity score,
threshold, runner-up score, margin, and margin threshold. That evidence is
diagnostic only; it explains why face identity did or did not contribute to the
turn owner, but it does not loosen the strict `primary_face_person_id` policy.

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
| `identity_memory/tailwag_http.py` | Calls Tailwag for directory lookup, biometric search/enrollment, encounters, and prompt context. |
| `control/tool_runtime.py` | Arms voice enrollment after face enrollment succeeds. |

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
    -> ToolRuntime arms pending voice enrollment
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

That means the LLM does not receive or pass manager, cost center, job family, or
similar internal org fields through the tool surface. It should pass the verified
username when `resolve_employee_identity` returned one. The rest is loaded locally
from `EmployeeDirectoryService.get_verified_profile()`.

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

## Enrollment-Only Quality Policy

These checks are still active for face enrollment. They are not part of the
continuous recognition or attention gate path. The enrollment quality gates live
in `FaceEnrollmentPolicy` in `face_recognition_service.py`:

```python
@dataclass(frozen=True)
class FaceEnrollmentPolicy:
    min_face_area: int = 1300
    min_brightness: float = 35.0
    max_brightness: float = 220.0
    min_contrast: float = 15.5
    min_embedding_similarity: float = 0.60
```

`_assess_enrollment_face_quality()` can reject an enrollment frame for:

| Reason | Meaning | User guidance |
|---|---|---|
| `face_too_small` | Face bbox area is below `1300` px. | Come closer. |
| `face_clipped` | Face touches image boundary. | Center whole face in view. |
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

When `failure_reason="embedding_inconsistent"`, the response also includes
`enrollment_diagnostics` so operators can see how far the burst missed the
policy:

```json
{
  "accepted_frame_count": 5,
  "consistent_frame_count": 1,
  "required_stable_frames": 3,
  "min_embedding_similarity": 0.6,
  "best_failed_similarity": 0.58,
  "best_failed_shortfall": 0.12,
  "similarities_to_reference": [1.0, 0.58, 0.42]
}
```

The dashboard exposes the same signal as flattened `tool_enrollment_*` fields on
the `enroll_visible_person` tool-result row, including consistent/required frame
counts, threshold, best failed similarity, shortfall, and reference similarities.

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

Then `ToolRuntime` sees the successful `person_id` and calls
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
detected_faces = [face for face in detected_faces if bbox_area >= min_face_area]
gated_faces, rejected_count = filter_detections_by_depth(...)
embedding = self.extract_face_embedding(image, detection)
```

Argos normalizes provider camera frames to OpenCV-style BGR arrays before they
reach the face pipeline. `pipeline.py` converts those BGR arrays to RGB PIL
images immediately before MTCNN and FaceNet embedding. Raw RGB provider payloads
must be tagged as `rgb8` or `color_space: rgb` so the transport can convert them
to internal BGR first. Raw data capture writes RGB JPEG artifacts for operator
inspection, but recognition and enrollment continue to use the normalized BGR
frame internally.

The minimum recognition face area currently follows `FaceEnrollmentPolicy.min_face_area`
(`1300` px in the static interaction profile). This keeps tiny distant faces from
becoming recognized-person context while still allowing fisheye captures where
valid nearby faces are smaller than RealSense crops.

Recognition sends FaceNet vectors to Tailwag. Tailwag owns biometric reference
search, thresholds, margin policy, consent filtering, and archived-person
filtering, then returns the accepted/rejected diagnostics that Argos logs and
shows on the dashboard.

For adaptive reference updates, the face loop only caches the most recent
accepted face embedding for each recognized person in process memory. It does not
write or update biometric storage directly. After speaker resolution, Argos
offers that cached face observation to Tailwag only when the final owner source
is `audio_face_agree`. Tailwag decides whether the observation is close enough to
the current `FaceReference` aggregate and whether the reference has already
reached its target sample count.

What recognition does not do:

- no blur gate
- no brightness gate
- no contrast gate
- no frontal-face enrollment policy
- no 5-frame burst consistency check
- no prompt-visible biometric update notes

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
- extra detections below `FaceEnrollmentPolicy.min_face_area` can be ignored
- tiny, weak extra detections can be ignored
- significant extra faces return `status="retry_single_face"` and
  `failure_reason="multiple_faces"`

Recognition supports multiple faces:

- each usable face is embedded
- each embedding is matched independently
- recognized people become `PersonContext` rows
- unrecognized usable faces increment `unknown_count`
- `primary_face_person_id` is set only for exactly one usable recognized face

The presence snapshot also tracks consecutive processed face-loop frames with
unknown faces:

- `unknown_stability_frames`
- `attentive_unknown_stability_frames`

These counters reset when unknown faces are no longer visible. Proactive unknown
or mixed face greetings wait until the relevant counter reaches
`recognition_stability.window_frames`, so a known person who briefly misses the
recognition threshold is not immediately greeted as unknown.

`scene_analysis.py` keeps ownership strict:

- exactly one usable recognized face -> `primary_face_person_id`
- zero usable recognized faces -> `None`
- more than one usable face, recognized or unknown -> `None`

## Face Target vs Turn Owner

At recording start, `AudioRuntime` snapshots the strict face id and the
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
for Tailwag realtime episode participants and person-context lookup.

Turn prompt person context is also tied to `owner_id`. When `owner_id` is not
resolved, the runtime does not include visible recognized people as
person-specific prompt context. When `owner_id` is resolved, the prompt may show
that person under `[PERSON SPEAKING TO YOU]` and list other visible people only
as lightweight names under `[OTHER PEOPLE IN VIEW]`.

The canonical face-derived name in turn state is `primary_face_person_id`.

## Config Knobs

Main config lives in:

`config/profiles/static_interaction.yaml`

```yaml
face_recognition:
  enabled: true
  loop_interval_sec: 0.3
  depth_gate:
    enabled: false
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
    min_face_area: 1300
    max_abs_yaw_deg: 20.0
    max_abs_pitch_deg: 18.0
    max_abs_roll_deg: 90.0
    min_abs_pitch_deg: 0.0
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
| `min_face_area: 1300` | Absolute minimum detected face bbox area before head-pose scoring. | Lower only if valid distant faces are rejected. |
| `max_abs_yaw_deg: 20.0` | Left/right head angle limit. | Tighten if side conversations falsely open admission. |
| `max_abs_pitch_deg: 18.0` | Up/down head angle limit. | Tune with camera mounting height. |
| `max_abs_roll_deg: 90.0` | Head tilt limit. | Wide by default so roll rarely blocks attention. |
| `min_abs_pitch_deg: 0.0` | Minimum up/down head angle. | Keep low unless the camera geometry needs a pitch band. |

The attention gate uses 6DRepNet on the existing MTCNN face crops. It does not
replace FaceNet and does not add a second face detector. It uses one absolute
bbox-area threshold and one fixed yaw/pitch/roll threshold set; it does not vary
pose limits by distance or optical-axis offset. If `sixdrepnet` is not
installed or the model cannot initialize, attention returns
`sixdrepnet_unavailable` and passive attention admission remains closed. The
presence snapshot includes:

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

## Thresholds And Profile Knobs

Most field-tuned face policy lives in the profile. A few thresholds remain
centralized in code:

- `face_recognition.enrollment_policy` for enrollment quality
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
- `face_clipped`
- `embedding_inconsistent`

Those are the ones most likely to save the wrong identity or produce a weak
long-term reference.

## Useful Tests

Relevant focused tests:

```bash
poetry run pytest \
  tests/argos_src/face_recognition/test_face_recognition_service.py \
  tests/argos_src/face_recognition/test_attention_gate.py \
  tests/argos_src/face_recognition/test_enrollment_display_review.py \
  tests/argos_src/tools/unitree_go2/vision/test_enroll_visible_person_tool.py \
  tests/argos_src/tools/unitree_go2/vision/test_resolve_employee_identity_tool.py \
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

Manage durable identities, face references, and voice references with Tailwag
tooling from `tailwag-memory`.
