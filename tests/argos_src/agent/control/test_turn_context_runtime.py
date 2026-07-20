from __future__ import annotations

from types import SimpleNamespace

from argos_src.agent.control.turn_context_runtime import TurnContextRuntime


def test_identity_person_normalizes_string_directory_profile_lines():
    profile = SimpleNamespace(
        display_name="Alex",
        interaction_count=2,
        metadata={},
        directory_profile_lines=(
            "['Title: Robotics Software Engineer I Co-op', "
            "'Manager: Brian Waite']"
        ),
    )
    host = SimpleNamespace(
        identity_memory_client=SimpleNamespace(
            person_profile=lambda _person_id: profile,
        ),
        memory_context_compiler=None,
        logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
    )

    person = TurnContextRuntime(host).identity_person("person-1")

    assert person is not None
    assert person.directory_profile_lines == (
        "Title: Robotics Software Engineer I Co-op",
        "Manager: Brian Waite",
    )
