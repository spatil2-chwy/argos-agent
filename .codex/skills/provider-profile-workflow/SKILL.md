---
name: provider-profile-workflow
description: Guide changes to provider API models, HTTP or Zenoh transports, fake provider behavior, manifests, profile YAML, resource capabilities, enabled tool IDs, or config loading.
---

# Provider Profile Workflow

Use this workflow when changing how profiles describe hardware/resources or how Argos talks to robot/display providers.

## Load Context

Read the smallest relevant subset:

- `config/profiles/static_interaction.yaml`
- `config/manifests/puffle.yaml`
- `docs/architecture.md`
- `docs/launch.md`
- `argos_src/provider_api/`
- `argos_src/profile_config.py`

## Preserve

- Resource IDs and capability names remain stable unless intentionally migrated.
- Profile fields stay backward compatible with tests and launch docs.
- Fake provider behavior mirrors the contract used by the runtime.
- Transport-neutral dataclasses stay plain and predictable.
- Tool enablement in profiles remains aligned with registered tool IDs.

## Work

1. Separate schema/config changes from runtime behavior changes.
2. Update profile, manifest, and docs together when the operator-facing contract changes.
3. Keep test fixtures and fake provider semantics in sync.
4. If parallel agents are requested, use `provider-contract-guardian` and `test-runner`.

## Targeted Tests

```bash
python3 -B -m pytest tests/argos_src/provider_api
python3 -B -m pytest tests/argos_src/test_argos_profile_config.py
python3 -B -m pytest tests/argos_src/tools/test_tool_ids.py
```
