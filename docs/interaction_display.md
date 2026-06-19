# Puffle Interaction Display Resource

The Puffle display is an optional local browser surface controlled through the
provider/resource system. It is not an LLM tool. The runtime owns normal display
state, and tools can use it for blocking human review flows.

## Resource Configuration

The default Puffle manifest defines:

```yaml
providers:
  - id: puffle-go2-display
    transport: http
    key_prefix: argos/providers/puffle-go2-display
    connect_endpoints:
      - http://localhost:4173

resources:
  - id: screen_001
    kind: display
    hardware: puffle_screen
    provider: puffle-go2-display
    capabilities:
      - display.command
      - display.interaction
```

The default `static_interaction` profile selects:

```yaml
display:
  enabled: true

resources:
  interaction_display: screen_001
```

`display.enabled` defaults to `true`. If it is true and the selected manifest
contains a resource with `display.command`, Argos defaults
`resources.interaction_display` to that resource when the profile omits it.

To run on a robot or workstation with no screen, disable display explicitly:

```yaml
display:
  enabled: false
```

When disabled, Argos does not create `DisplayRuntime`, does not send display
commands, and face enrollment uses the non-review path. If a manifest has no
display resource, display updates are also no-ops.

## HTTP Contract

The HTTP provider transport maps display operations to the local display server:

| Operation | Endpoint | Purpose |
|---|---|---|
| `display.command` | `POST /argos/providers/puffle-go2-display/resources/screen_001/display` | Send face, subtitle, clear/reset, message, countdown, Rive, or preview commands. |
| `display.health` | `GET /argos/providers/puffle-go2-display/resources/screen_001/health` | Check whether the display control server is reachable. |
| `display.image` | `POST /argos/providers/puffle-go2-display/resources/screen_001/image` | Show or clear the small live camera image panel. |
| `display.state` | `GET /argos/providers/puffle-go2-display/resources/screen_001/state` | Read current display state. |
| `display.await_response` | `GET /argos/providers/puffle-go2-display/resources/screen_001/response` polling | Wait for an interactive response matching `requestId`. |

The display server is expected at:

```text
http://localhost:4173
```

The base URL stays short; Argos appends the provider/resource namespace from the
manifest. For the default Puffle manifest, all display control routes live under:

```text
/argos/providers/puffle-go2-display/resources/screen_001
```

## Runtime Behavior

`DisplayRuntime` is the only high-level display API used by the agent. It
deduplicates repeated faces and isolates normal display failures from the
conversation path.

Current state mapping:

| Runtime state | Display command |
|---|---|
| idle | face `happy` |
| mic admission / alert | face `think` |
| recording | face `think` plus subtitle `Recording...` |
| audio committed / waiting for model | centered message `Thinking...` |
| assistant speaking | face `excited` |
| assistant transcript deltas | subtitle updates |

Cooldown is an internal engagement state and normally displays as idle/happy.
Recording is the only state mode that sends a status subtitle; assistant
transcript subtitles still stream from the spoken response.

Display updates are queued through a background worker in `RealtimeRobotAgent`.
HTTP calls do not run inside the microphone callback.

## Face Enrollment Review

`enroll_visible_person` remains one agent-visible tool call. The LLM does not
need to call a display tool.

When the display is configured, face enrollment does:

```text
capture and validate burst
    -> prepare candidate embedding and padded reference-face preview
    -> send face_capture_preview to screen_001
    -> wait for Accept / Reject
    -> save only after Accept
```

Reject, timeout, or display-unavailable responses do not save the face.
The preview image is cropped from the same reference face bbox used by the
enrollment candidate, with padding for a more natural confirmation view.

The preview command sent to the display is:

```json
{
  "type": "face_capture_preview",
  "requestId": "enroll-...",
  "imageUrl": "data:image/png;base64,...",
  "title": "Face Capture Preview",
  "acceptLabel": "Accept",
  "rejectLabel": "Reject"
}
```

The browser posts or stores the response for the namespaced response endpoint;
Argos polls
`/argos/providers/puffle-go2-display/resources/screen_001/response`
until it sees a matching `requestId`.

## Tests

Focused coverage lives in:

```text
tests/argos_src/provider_api/test_http_provider_client.py
tests/argos_src/display/test_runtime.py
tests/argos_src/face_recognition/test_enrollment_display_review.py
```

Run with:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest \
  tests/argos_src/provider_api/test_http_provider_client.py \
  tests/argos_src/display/test_runtime.py \
  tests/argos_src/face_recognition/test_enrollment_display_review.py
```
