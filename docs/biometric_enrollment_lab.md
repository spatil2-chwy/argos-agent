# Biometric Enrollment Lab

Use this sheet to capture one participant's face and voice locally, then upload
the approved aggregate references to Tailwag.

The normal sequence is:

```text
capture -> list -> push -> verify -> optional cleanup
```

- `capture` writes a protected local bundle. It does not contact Tailwag.
- `list` reads local bundle state only.
- `push` is a production write. It resolves the employee-directory record,
  checks for conflicts and existing references, and uploads only missing face or
  voice aggregates after confirmation.
- `verify` uses read-only Tailwag existence checks.
- `cleanup` permanently deletes the selected local bundle; it does not delete
  anything from Tailwag.

## Before the lab

Confirm all of the following:

- The participant has consented to face and voice enrollment.
- You have their exact official first and last name and email ID prefix (the
  part before `@`). You do not need a Person ID.
- The email prefix and site match exactly one employee-directory record.
- Only the participant is visible and only the participant speaks during
  capture.
- The robot provider is running and serving `arducam_001`.
- The host microphone works, and the interaction display is available at
  `http://localhost:4173`.
- `run_profile.py` is not running.
- The matching Tailwag API is deployed and healthy.
- The host has the FaceNet and speaker-model dependencies and caches already
  installed.

Local bundles contain unencrypted photos, audio, and embeddings. Keep the output
directory protected, and never manually edit bundle files or upload state.

## 1. Prepare the robot shell

```bash
cd ~/code/argos-agent

git status --short
git switch main
git pull --ff-only origin main

source setup_shell.sh

unset LAB_LOCAL_PROFILE LAB_LOCAL_MANIFEST
unset LAB_TAILWAG_BASE_URL LAB_REQUEST_BASE LAB_ARGOS_OUTPUT_ROOT

export ENROLLMENT_PROFILE='config/profiles/cody_interaction.yaml'
export ENROLLMENT_SITE_CODE='BOS3'
export ENROLLMENT_OUTPUT_ROOT="$HOME/.local/share/argos-biometric-enrollment"
export TAILWAG_BASE_URL='https://a9vhnyd929.execute-api.us-east-2.amazonaws.com'
export TAILWAG_REQUEST_BASE="${TAILWAG_BASE_URL}/argos/providers/memory/resources/memory/request"

mkdir -p "$ENROLLMENT_OUTPUT_ROOT"
chmod 700 "$ENROLLMENT_OUTPUT_ROOT"

curl -fsS "$TAILWAG_BASE_URL/health" | jq .
```

Stop if `git status --short` reports local changes. The health response must show
`"status": "ok"` before continuing.

Environment installation is normally one-time. If this robot has not been
prepared, run `poetry install` and install
`argos_src/face_recognition/requirements.txt` with `--no-deps` before the lab,
not between participants.

## 2. Enter the participant identity

```bash
read -rp 'Official first and last name: ' ENROLLMENT_OFFICIAL_NAME
read -rp 'Employee email prefix: ' ENROLLMENT_USERNAME

printf 'Name: %s\n' "$ENROLLMENT_OFFICIAL_NAME"
printf 'Email prefix: %s\n' "$ENROLLMENT_USERNAME"
printf 'Site: %s\n' "$ENROLLMENT_SITE_CODE"
```

Check the spelling carefully. If the name or prefix is wrong after capture,
start a new capture; later commands do not amend bundle identity.

## 3. Capture face and voice samples

```bash
python3 -m scripts.labs.biometric_enrollment_lab \
  capture "$ENROLLMENT_OFFICIAL_NAME" \
  --username "$ENROLLMENT_USERNAME" \
  --site-code "$ENROLLMENT_SITE_CODE" \
  --profile "$ENROLLMENT_PROFILE" \
  --output-root "$ENROLLMENT_OUTPUT_ROOT"
```

Follow the terminal and display prompts. Capture accepts five face samples and
five voice clips, then checks consistency before finalizing the bundle. The
successful command ends with `Capture ready for approved push`.

If capture stops early or fails consistency, start a new capture. Preserve the
incomplete `capture=collecting` bundle for review; do not edit it or try to push
it.

## 4. Review the local bundle

```bash
python3 -m scripts.labs.biometric_enrollment_lab \
  list \
  --output-root "$ENROLLMENT_OUTPUT_ROOT"
```

Select only the intended participant's newest `capture=complete` bundle. Before
push, both modalities normally show `pending`.

Upload states are:

- `pending`: not uploaded
- `failed`: the same bundle can be retried
- `uploaded`: this bundle uploaded the reference
- `skipped`: Tailwag already had an active reference

Do not push a bundle reported as invalid.

## 5. Load the production token and push

Load the production bearer token through the approved secret process without
putting it in shell history:

```bash
read -rsp 'Production Tailwag bearer token: ' TAILWAG_API_BEARER_TOKEN
echo
export TAILWAG_API_BEARER_TOKEN
```

Verify authenticated access:

```bash
curl -fsS \
  -H "Authorization: Bearer $TAILWAG_API_BEARER_TOKEN" \
  "$TAILWAG_BASE_URL/argos/providers/memory/resources/memory/health" |
  jq .
```

Then push:

```bash
python3 -m scripts.labs.biometric_enrollment_lab \
  push \
  --profile "$ENROLLMENT_PROFILE" \
  --output-root "$ENROLLMENT_OUTPUT_ROOT"
```

The command resolves the email prefix and site to exactly one directory record,
verifies the official name, prints the canonical Person ID and missing
modalities, and asks for consent confirmation. Confirm only when the displayed
canonical identity is the participant. Type the displayed official name exactly.

A Person node may be created for a directory-only employee. Only missing face or
voice reference nodes are written. If a conflict names another person, or the
canonical identity is wrong, stop without confirming.

## 6. Verify completion without creating nodes

First confirm local state:

```bash
python3 -m scripts.labs.biometric_enrollment_lab \
  list \
  --output-root "$ENROLLMENT_OUTPUT_ROOT"
```

Both modalities must be `uploaded` or `skipped`. Enter the canonical Person ID
printed by `push`:

```bash
read -rp 'Canonical Person ID: ' PERSON_ID
```

Verify face and voice references:

```bash
curl -fsS -X POST \
  -H "Authorization: Bearer $TAILWAG_API_BEARER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"person_id\":\"$PERSON_ID\"}" \
  "$TAILWAG_REQUEST_BASE/biometrics_face_references_exists" |
  jq .

curl -fsS -X POST \
  -H "Authorization: Bearer $TAILWAG_API_BEARER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"person_id\":\"$PERSON_ID\"}" \
  "$TAILWAG_REQUEST_BASE/biometrics_voice_references_exists" |
  jq .
```

Expected responses:

```json
{"has_face_reference": true}
{"has_voice_reference": true}
```

These existence checks are read-only. Do not use the normal realtime runtime as
a no-write verification test: live conversations can create Episode nodes and
can submit adaptive biometric updates.

## 7. Finish

Clear secrets and participant variables:

```bash
unset TAILWAG_API_BEARER_TOKEN PERSON_ID
unset ENROLLMENT_OFFICIAL_NAME ENROLLMENT_USERNAME
```

Keep the completed local bundle until its retention requirement is satisfied.
When deletion is explicitly approved, remove it through the lab command:

```bash
python3 -m scripts.labs.biometric_enrollment_lab \
  cleanup \
  --output-root "$ENROLLMENT_OUTPUT_ROOT"
```

`cleanup` prompts for the bundle and requires `DELETE <official name>`. It
permanently deletes only that local bundle.

## Stop and retry rules

- `401 Unauthorized`: reload the current production bearer token.
- `URL rejected: No host part`: rerun the `TAILWAG_BASE_URL` and
  `TAILWAG_REQUEST_BASE` exports from Step 1.
- Display unavailable: start the configured display server, then start a new
  capture.
- Inconsistent samples or an exhausted attempt limit: start a new capture; do
  not lower normal quality thresholds.
- Partial push: fix the external error and retry the same bundle. The retry
  rechecks existing references and uploads only unfinished modalities.
- Directory mismatch, wrong canonical name, biometric conflict, or invalid
  bundle: stop and investigate. Do not repeatedly retry or manually edit files.
