# Biometric Enrollment Lab Runbook

This runbook covers the local-first face and voice enrollment workflow in
`scripts/labs/biometric_enrollment_lab.py`.

The workflow has four separate commands:

```text
capture -> list -> push -> cleanup
```

- `capture` uses the robot provider camera, the host microphone, and optionally
  the interaction display. It does not make a Tailwag identity-memory request,
  though configured providers and first-use model downloads may use the
  network.
- `list` reads local bundle state only.
- `push` resolves the employee through Tailwag, checks existing references,
  performs global biometric conflict searches, and uploads missing aggregate
  embeddings after operator approval.
- `cleanup` permanently deletes an upload-complete local bundle.

## Safety and privacy rules

1. Obtain the subject's consent before starting capture.
2. Capture only one person at a time. Keep other faces out of the camera view
   and other voices out of the recording area.
3. Do not run `run_profile.py` during teleoperation or biometric capture.
4. Move the robot only with the external robot provider's approved teleop
   process. Argos and Tailwag do not contain the operator teleop launcher.
5. Disable provider-side navigation, autonomy, and patrol behaviors before
   positioning or capture.
6. Stop the teleop writer, command zero velocity through its documented stop
   path, and verify that the robot is stationary before capture.
7. Keep the external provider running after teleop stops because capture still
   needs its configured camera resource.
8. Treat the bundle directory as sensitive. It contains unencrypted face
   images, WAV files, and biometric embeddings.
9. Do not manually edit bundle manifests, upload state, embeddings, or captured
   artifacts.

## Prerequisites

Before the lab session, confirm:

- The intended Argos revision is installed on the robot host.
- The matching Tailwag API revision is deployed before any `push`. It must
  provide both face- and voice-reference existence routes.
- The employee directory contains exactly one record for the subject's email
  prefix and site. A pre-existing Tailwag `Person` node is not required.
- The selected profile points to the correct site and resources. The default
  `static_interaction` profile uses site `BOS3`, camera `arducam_001`, display
  `screen_001`, and the Puffle manifest.
- The external robot provider is running and serving the configured camera.
- The host microphone is working with the selected profile settings.
- The interaction-display server is running if the display will be used.
  Otherwise, use `--no-display`.
- The host has sufficient protected disk space for the raw capture bundles.
- The FaceNet and speaker-recognition model dependencies and model caches are
  available.
- For `push`, the host has outbound HTTPS connectivity and
  `TAILWAG_API_BEARER_TOKEN` is available without placing it in shell history or
  the repository.

Prepare the Argos environment:

```bash
cd ~/argos-agent
poetry install
source setup_shell.sh
python3 -m pip install --no-deps -r argos_src/face_recognition/requirements.txt
python3 -m scripts.labs.biometric_enrollment_lab --help
```

The face requirement install is normally a one-time host setup. Do not reinstall
dependencies immediately before every participant unless the environment
actually changed. The OpenAI API key is not required for this lab command. The
first FaceNet or speaker-model initialization may require network access if its
model cache has not already been populated.

## Step 1: verify the participant's directory identity

Record the participant's:

- exact official first and last name
- email ID prefix, such as `jdoe` before the `@` sign
- site code

The email prefix is required and is passed through `--username`:

```bash
python3 -m scripts.labs.biometric_enrollment_lab \
  capture "Jane Doe" \
  --username jdoe \
  --site-code BOS3
```

The operator does not need a Tailwag Person ID. Capture stores the official name
and prefix in the bundle. During push, Tailwag requires exactly one directory
record for that prefix and site and verifies that its official name matches. It
then reuses the linked or unique email-matched Person, or creates
`person_<lowercase_prefix>` for a directory-only employee.

If the captured name or prefix is wrong, start a new capture. Identity values
passed to later `list`, `push`, or `cleanup` commands do not amend the bundle.

## Step 2: position and prepare the robot

1. Use the external provider's teleop tool to move the robot to the participant.
2. Stop teleop and verify zero velocity.
3. Keep the provider process running for camera access.
4. Make sure `run_profile.py` is stopped.
5. Place only the participant in the camera view.
6. Use even lighting and keep the entire face visible.
7. Make the room quiet enough for five clean voice recordings.
8. Confirm the correct microphone and camera are selected.

The default profile records the host microphone through PipeWire at 24 kHz with
2,400-frame blocks. If Zenoh discovery cannot find the provider, set
`ARGOS_ZENOH_CONNECT=tcp/ROBOT_OR_PROVIDER_HOST:7447` to the approved endpoint.
The interaction display is normally served at `http://localhost:4173`.

If the provider-side teleop command is unknown, locate it in the provider
checkout or robot workstation:

```bash
cd /path/to/provider-checkout
rg -n -i \
  'teleop|teleop_twist_keyboard|joy(_teleop)?|/cmd_vel|Twist|velocity_for_duration' .
```

Do not substitute `argos_src/tools/unitree_go2/locomotion/move_robot.py`; that is
an agent-facing bounded motion tool, not an operator teleop program.

## Step 3: capture face samples

Run the capture command from Step 1. The lab:

1. Creates a UUID-named local bundle.
2. Shows the bundle path in the terminal.
3. Requests permission to start the photo phase.
4. Counts down before each photo.
5. Saves only samples that pass single-face and quality checks.
6. Continues until it has five accepted photos or reaches the attempt limit.

The subject should follow the displayed/terminal guidance and make small changes
between photos: straight on, slightly left, slightly right, a natural smile, and
slightly closer while keeping the whole face visible.

Rejected attempts are reported and their sample artifacts are discarded.

## Step 4: capture voice samples

After face capture succeeds, the lab:

1. Requests permission to start the voice phase.
2. Shows a countdown and one sentence at a time.
3. Starts recording only after the countdown.
4. Saves only clips that pass the local audio enrollment checks.
5. Continues until it has five accepted clips or reaches the attempt limit.

The participant should read the displayed sentence. Only one person should
speak. Wait for `Start speaking now` before speaking and remain silent while the
display says `Submitting audio`.

## Step 5: confirm local capture completion

After both modalities pass consistency checks, the command prints:

- the bundle UUID
- the bundle directory
- accepted face and voice counts
- consistency diagnostics
- `Capture ready for approved push`

The default bundle location is:

```text
data_collection/.biometric_enrollment_bundles/<bundle_uuid>/
```

Capture has not contacted Tailwag or uploaded anything at this point.

If capture is interrupted or fails before finalization, its bundle remains in
`capture=collecting` state. It cannot be resumed, pushed, or removed by normal
`cleanup`. Start a new capture for the participant and leave the incomplete
bundle for administrator review and approved exact-path removal.

## Step 6: list and review local bundles

```bash
python3 -m scripts.labs.biometric_enrollment_lab list
```

Each valid bundle shows:

- captured person name
- creation time
- `capture=collecting` or `capture=complete`
- face upload state
- voice upload state
- bundle UUID

Upload states are:

- `pending`: not uploaded yet
- `failed`: a previous push attempt failed and can be retried
- `uploaded`: this bundle uploaded the reference
- `skipped`: Tailwag already had an active reference

An `INVALID local biometric bundle` message means integrity/schema validation
failed. Do not push, edit, or automatically delete that path.

## Step 7: prepare for push

Push is a real remote write. `--provider-transport fake` is not a push dry-run.

Before running it:

1. Confirm the matching Tailwag API revision is deployed and healthy.
2. Confirm the intended employee-directory data is loaded.
3. Export the bearer token in the current shell using the approved secret
   retrieval process. For an interactive session, avoid typing it into shell
   history:

   ```bash
   read -rsp "Tailwag bearer token: " TAILWAG_API_BEARER_TOKEN
   echo
   export TAILWAG_API_BEARER_TOKEN
   ```

4. Confirm outbound HTTPS access to the memory provider endpoint in the selected
   manifest.
5. Confirm the participant has consented to upload.

The camera, microphone, display, teleop, and Argos realtime runtime are not
needed for `push`.

## Step 8: push the selected bundle

```bash
python3 -m scripts.labs.biometric_enrollment_lab push
```

The command:

1. Lists only complete bundles with pending or failed work.
2. Prompts for the bundle number.
3. Resolves the captured name/email prefix to a verified employee-directory record.
4. Validates an existing Person when the directory resolver returns one.
5. Checks whether face and voice references already exist for that identity.
6. If anything is missing, prints the canonical name, canonical person ID, and
   missing modalities, then asks:

   ```text
   Confirm subject consent; type 'Jane Doe' to approve upload:
   ```

7. Binds all retries of the bundle to that canonical identity.
8. Globally searches every missing aggregate for a strong match to another
   person before the first write.
9. During enrollment, Tailwag reuses the existing Person or creates the
   canonical Person for an employee who has only a directory record.
10. Uploads only the missing aggregate embeddings.
11. Journals face and voice results independently.

Existence checks run face then voice. After approval and bundle binding, conflict
searches run face then voice; if both pass, enrollment runs face then voice. This
ordering is why a failed voice write can leave face durable for a safe retry.

Type the displayed canonical name exactly. Any other answer cancels before
conflict search or enrollment.

If both references already exist, no embedding is sent and no upload approval
prompt appears; the local modalities are marked `skipped`.

## Step 9: handle retries safely

If `push` fails:

1. Read the error and do not alter the bundle files.
2. Run `list` and note whether face or voice is `failed`, `uploaded`, or
   `skipped`.
3. Correct the external cause.
4. Run the same `push` command again and select the same bundle.

Retries remain bound to the same canonical person. Tailwag existence checks are
repeated for unfinished modalities, so a remote write that succeeded before a
timeout or local journal failure is detected and marked `skipped` instead of
being uploaded again.

One modality can be durable while the second fails. Do not recapture solely
because of a partial push; retry the same bundle.

Stop and investigate rather than repeatedly retrying when:

- directory resolution is ambiguous or returns the wrong employee
- the global conflict check reports another person
- the canonical name shown by `push` is not the participant
- the bundle is reported invalid

## Step 10: verify completion

Run:

```bash
python3 -m scripts.labs.biometric_enrollment_lab list
```

The selected bundle is locally complete when both modalities are `uploaded` or
`skipped`.

For a remote, read-only verification, use the canonical person ID printed by
`push` when upload work was needed. If both references already existed, obtain
the ID through the approved Tailwag administration path. Call the deployed
face- and voice-reference existence endpoints; both responses should be `true`.
The exact base URL comes from the selected Argos manifest:

```bash
export TAILWAG_BASE_URL=https://example.execute-api.us-east-2.amazonaws.com
export PERSON_ID=person_jdoe

curl -fsS -X POST \
  -H "Authorization: Bearer $TAILWAG_API_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"person_id\":\"$PERSON_ID\"}" \
  "$TAILWAG_BASE_URL/argos/providers/memory/resources/memory/request/biometrics_face_references_exists"

curl -fsS -X POST \
  -H "Authorization: Bearer $TAILWAG_API_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"person_id\":\"$PERSON_ID\"}" \
  "$TAILWAG_BASE_URL/argos/providers/memory/resources/memory/request/biometrics_voice_references_exists"
```

Expected response keys:

```json
{"has_face_reference": true}
{"has_voice_reference": true}
```

Do not place the bearer token directly in the command or documentation. After
verification, clear the temporary shell variables:

```bash
unset TAILWAG_API_BEARER_TOKEN TAILWAG_BASE_URL PERSON_ID
```

## Step 11: clean up local biometric data

Only after verification:

```bash
python3 -m scripts.labs.biometric_enrollment_lab cleanup
```

The command:

1. Lists only intact bundles whose face and voice states are `uploaded` or
   `skipped`.
2. Prompts for the bundle number.
3. Requires the exact confirmation:

   ```text
   DELETE Jane Doe
   ```

4. Permanently deletes the entire local bundle.

Cleanup does not delete anything from Tailwag. It does not retain a local
receipt. Record any required audit information through the approved process
before cleanup.

## Common failures

| Symptom | Meaning and response |
|---|---|
| No eligible bundle for push | Capture is incomplete/invalid, or both modalities are already terminal. Run `list`. |
| Email prefix lookup or name verification fails | Confirm the prefix, official name, and site, then start a new capture. |
| Display prompt unavailable | Start the configured display server or recapture with `--no-display`. |
| Repeated photo rejection | Ensure one face, better lighting, whole face in frame, and a stable camera view. |
| Repeated voice rejection | Confirm the input device, reduce noise, speak after the countdown, and avoid clipping. |
| Attempt limit reached | Start a new capture, optionally with higher `--max-photo-attempts` or `--max-voice-attempts`. |
| Unauthorized or invalid existence response | Verify the bearer token and that the matching Tailwag API revision is deployed. |
| Tailwag request times out | Check network/service health, then retry the same bundle. Enrollment retries recheck existence first. |
| Global conflict identifies another person | Stop. Verify the participant and directory identity before doing anything else. |
| One modality uploaded and the other failed | Fix the external error and retry the same bundle. |
| Bundle integrity verification failed | Preserve it for administrator review; do not push or normal-clean it. |

## Useful capture options

Use terminal prompts instead of the interaction display:

```bash
python3 -m scripts.labs.biometric_enrollment_lab \
  capture "Jane Doe" --username jdoe --site-code BOS3 --no-display
```

Increase attempt limits without reducing the required accepted samples:

```bash
python3 -m scripts.labs.biometric_enrollment_lab \
  capture "Jane Doe" \
  --username jdoe \
  --site-code BOS3 \
  --max-photo-attempts 20 \
  --max-voice-attempts 15
```

Auto-start the two capture phases:

```bash
python3 -m scripts.labs.biometric_enrollment_lab \
  capture "Jane Doe" --username jdoe --site-code BOS3 --yes
```

`--yes` does not approve a Tailwag push.

Use the same non-default root for every command:

```bash
python3 -m scripts.labs.biometric_enrollment_lab \
  capture "Jane Doe" --username jdoe --site-code BOS3 \
  --output-root /approved/protected/path

python3 -m scripts.labs.biometric_enrollment_lab \
  list --output-root /approved/protected/path

python3 -m scripts.labs.biometric_enrollment_lab \
  push --output-root /approved/protected/path

python3 -m scripts.labs.biometric_enrollment_lab \
  cleanup --output-root /approved/protected/path
```

Do not lower quality thresholds or the five-sample minimum during normal lab
operation. Threshold overrides are diagnostic controls and should be changed
only under an approved test plan.
