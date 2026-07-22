from __future__ import annotations

import pytest

from argos_src.identity_memory.normalization import normalize_directory_profile_lines


OBSERVED_DIRECTORY_PROFILE = (
    "['Title: Robotics Software Engineer I Co-op', 'Manager: Brian Waite', "
    "'Tenure: ...', 'Function: Administration']"
)
EXPECTED_DIRECTORY_LINES = (
    "Title: Robotics Software Engineer I Co-op",
    "Manager: Brian Waite",
    "Tenure: ...",
    "Function: Administration",
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([" Title: Engineer ", "Manager: Alex"], ("Title: Engineer", "Manager: Alex")),
        (("Title: Engineer", " Manager: Alex "), ("Title: Engineer", "Manager: Alex")),
        (" Title: Engineer ", ("Title: Engineer",)),
        (OBSERVED_DIRECTORY_PROFILE, EXPECTED_DIRECTORY_LINES),
        (None, ()),
        ("", ()),
        ("[]", ()),
        (123, ()),
        ({"Title": "Engineer"}, ()),
        (["Title: Engineer", {"Manager": "Alex"}, None], ("Title: Engineer",)),
    ],
)
def test_normalize_directory_profile_lines(value, expected):
    assert normalize_directory_profile_lines(value) == expected


def test_normalize_directory_profile_lines_preserves_unparseable_string():
    assert normalize_directory_profile_lines("[Title: Engineer]") == (
        "[Title: Engineer]",
    )
