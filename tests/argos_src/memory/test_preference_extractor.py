from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
import warnings
from pathlib import Path

from argos_src.agent.preference_types import PreferenceSegmentTurn


def _load_preference_extractor_module(monkeypatch):
    pydantic_mod = types.ModuleType("pydantic")

    class _FieldSpec:
        def __init__(self, default=..., default_factory=None, **kwargs):
            self.default = default
            self.default_factory = default_factory

    def _field(default=..., default_factory=None, **kwargs):
        return _FieldSpec(default=default, default_factory=default_factory, **kwargs)

    class _BaseModel:
        def __init__(self, **kwargs):
            annotations = {}
            for cls in reversed(type(self).__mro__):
                annotations.update(getattr(cls, "__annotations__", {}))
            for name in annotations:
                if name in kwargs:
                    value = kwargs[name]
                else:
                    default = getattr(type(self), name, None)
                    if isinstance(default, _FieldSpec):
                        if default.default_factory is not None:
                            value = default.default_factory()
                        elif default.default is not ...:
                            value = default.default
                        else:
                            value = None
                    else:
                        value = default
                setattr(self, name, value)

        @classmethod
        def model_validate(cls, value):
            return cls(**dict(value or {}))

        def model_dump(self, mode=None):
            def _dump(value):
                if isinstance(value, _BaseModel):
                    return value.model_dump(mode=mode)
                if isinstance(value, list):
                    return [_dump(item) for item in value]
                if isinstance(value, dict):
                    return {key: _dump(item) for key, item in value.items()}
                return value

            annotations = {}
            for cls in reversed(type(self).__mro__):
                annotations.update(getattr(cls, "__annotations__", {}))
            return {name: _dump(getattr(self, name)) for name in annotations}

        def model_dump_json(self):
            return json.dumps(self.model_dump())

    pydantic_mod.BaseModel = _BaseModel
    pydantic_mod.Field = _field
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_mod)

    messages_mod = types.ModuleType("langchain_core.messages")
    messages_mod.HumanMessage = object
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages_mod)

    init_mod = types.ModuleType("argos_src.llm")
    init_mod.get_llm_model_direct = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "argos_src.llm", init_mod)

    observability_mod = types.ModuleType("argos_src.observability.observability")
    observability_mod.LatencyLogger = object
    observability_mod.perf_now = lambda: 0.0
    monkeypatch.setitem(
        sys.modules,
        "argos_src.observability.observability",
        observability_mod,
    )

    pricing_mod = types.ModuleType("argos_src.observability.pricing")
    pricing_mod.estimate_text_generation_cost = lambda *args, **kwargs: {}
    monkeypatch.setitem(
        sys.modules,
        "argos_src.observability.pricing",
        pricing_mod,
    )

    db_mod = types.ModuleType("argos_src.face_recognition.store")
    db_mod.FaceRecognitionStore = object
    monkeypatch.setitem(sys.modules, "argos_src.face_recognition.store", db_mod)

    module_name = "test_argos_memory_live_chat_module"
    module_path = (
        Path(__file__).resolve().parents[3]
        / "argos_src/memory/live_chat.py"
    )
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_operations_payload_accepts_valid_operations(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)

    should_update, operations = module._parse_operations_payload(
        {
            "update": True,
            "ops": [
                {
                    "op": "create",
                    "kind": "preference",
                    "key": "preferred_language",
                    "summary": "preferred language: Spanish",
                    "value": {"field": "preferred_language", "value": "Spanish"},
                },
                {
                    "op": "create",
                    "kind": "preference",
                    "key": "likes_playful_greetings",
                    "summary": "likes: playful greetings",
                },
                {
                    "op": "create",
                    "kind": "note",
                    "key": "robot_brain_work",
                    "summary": "User works on robot social memory.",
                },
                {
                    "op": "create",
                    "kind": "followup",
                    "key": "luna_recovery",
                    "summary": "Luna is recovering from surgery.",
                    "due_at": "2026-05-16T00:00:00+00:00",
                    "expires_at": "2026-05-20T00:00:00+00:00",
                },
            ],
        },
        fallback_turn_ids=["rt-1"],
    )

    assert should_update is True
    assert operations["ops"][0]["key"] == "preferred_language"
    assert operations["ops"][1]["key"] == "likes_playful_greetings"
    assert operations["ops"][2]["key"] == "robot_brain_work"
    assert operations["ops"][3]["key"] == "luna_recovery"


def test_extraction_prompt_examples_use_full_response_shape(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)
    prompt = module.EXTRACTION_PROMPT.format(
        current_date="2026-05-19",
        current_time="13:45 EDT",
        candidate_memories="[]",
        conversation="User: hi",
    )
    examples = re.findall(r"```json\n(.*?)\n```", prompt, flags=re.DOTALL)

    assert examples
    for example in examples:
        payload = json.loads(example)
        assert set(payload) == {"update", "ops"}
        assert payload["update"] is True
        assert isinstance(payload["ops"], list)
        assert payload["ops"]


def test_extraction_prompt_requires_archiving_answered_followups(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)

    assert "agent asked about it and the user answered" in module.EXTRACTION_PROMPT
    assert "archive the followup, then create or update the durable memory" in (
        module.EXTRACTION_PROMPT
    )
    assert "Do not rely on `expires_at` to clean this up later." in (
        module.EXTRACTION_PROMPT
    )


def test_segment_to_conversation_omits_turn_ids(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)
    segment = module.PreferenceSegment(
        segment_id="rt-1",
        person_id="person-1",
        turns=(
            PreferenceSegmentTurn(
                turn_id="rt-1",
                person_id="person-1",
                user_text="my dog is Mochi",
                assistant_text="How old is Mochi?",
            ),
        ),
    )

    rendered = module._segment_to_conversation(segment)

    assert rendered == "User: my dog is Mochi\nAssistant: How old is Mochi?"
    assert "TURN_ID" not in rendered


def test_parse_operations_payload_rejects_invalid_operations_safely(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)

    should_update, operations = module._parse_operations_payload(
        {
            "update": True,
            "ops": [
                {"op": "create", "kind": "unknown", "summary": "bad"},
                {"op": "create", "kind": "preference", "summary": ""},
                {"op": "update", "summary": "missing memory id"},
                {"op": "archive", "memory_id": ""},
                {"op": "delete", "memory_id": "mem_1"},
            ],
        },
        fallback_turn_ids=["rt-1"],
    )

    assert should_update is False
    assert operations == {"ops": []}


def test_parse_operations_payload_accepts_profile_memory_without_regex_gate(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)

    should_update, operations = module._parse_operations_payload(
        {
            "update": True,
            "ops": [
                {
                    "op": "create",
                    "kind": "preference",
                    "key": "preferred_name",
                    "summary": "preferred name: Sakshi Patil",
                    "value": {"field": "preferred_name", "value": "Sakshi Patil"},
                },
            ],
        },
        fallback_turn_ids=["rt-1"],
        conversation="User: My parents are visiting the U.S. for the first time.",
    )

    assert should_update is True
    assert operations["ops"][0]["key"] == "preferred_name"


def test_parse_operations_payload_accepts_relative_time_note_without_regex_gate(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)

    should_update, operations = module._parse_operations_payload(
        {
            "update": True,
            "ops": [
                {
                    "op": "create",
                    "kind": "note",
                    "key": "parents_cape_cod_weekend",
                    "summary": (
                        "User's parents are visiting the U.S. for the first time "
                        "and they plan to go to Cape Cod this weekend."
                    ),
                    "expires_at": "2026-06-01T00:00:00+00:00",
                },
            ],
        },
        fallback_turn_ids=["rt-1"],
        conversation=(
            "User: My parents are visiting the U.S. for the first time, "
            "and we're going to Cape Cod this weekend."
        ),
    )

    assert should_update is True
    assert operations["ops"][0]["kind"] == "note"
    assert "this weekend" in operations["ops"][0]["summary"]


def test_parse_operations_payload_accepts_dated_followup_for_short_term_plan(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)

    should_update, operations = module._parse_operations_payload(
        {
            "update": True,
            "ops": [
                {
                    "op": "create",
                    "kind": "followup",
                    "key": "parents_cape_cod_trip",
                    "summary": (
                        "Cape Cod trip with their parents planned for "
                        "the weekend of 2026-05-16."
                    ),
                    "due_at": "2026-05-18T09:00:00-04:00",
                    "expires_at": "2026-05-22T23:59:00-04:00",
                },
            ],
        },
        fallback_turn_ids=["rt-1"],
        conversation=(
            "User: My parents are visiting the U.S. for the first time, "
            "and we're going to Cape Cod this weekend."
        ),
    )

    assert should_update is True
    assert operations["ops"][0]["kind"] == "followup"
    assert operations["ops"][0]["expires_at"] == "2026-05-22T23:59:00-04:00"


def test_parse_operations_payload_rejects_followup_without_expiry(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)

    should_update, operations = module._parse_operations_payload(
        {
            "update": True,
            "ops": [
                {
                    "op": "create",
                    "kind": "followup",
                    "key": "parents_cape_cod_trip",
                    "summary": (
                        "Cape Cod trip with their parents planned for "
                        "the weekend of 2026-05-16."
                    ),
                    "due_at": "2026-05-18T09:00:00-04:00",
                },
            ],
        },
        fallback_turn_ids=["rt-1"],
    )

    assert should_update is False
    assert operations == {"ops": []}


def test_structured_response_to_payload_uses_parsed_model_and_usage(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)
    parsed = module.PreferenceExtractionOutput(
        update=True,
        ops=[
            module.MemoryOperationOutput(
                op="create",
                kind="preference",
                key="preferred_language",
                summary="preferred language: Spanish",
            )
        ],
    )

    class Raw:
        usage_metadata = {"input_tokens": 10, "output_tokens": 4}

    payload, usage = module._structured_response_to_payload(
        {"parsed": parsed, "raw": Raw(), "parsing_error": None}
    )

    assert payload["update"] is True
    assert payload["ops"][0]["key"] == "preferred_language"
    assert usage == {"input_tokens": 10, "output_tokens": 4}


def test_structured_response_to_payload_raises_on_parsing_error(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)

    try:
        module._structured_response_to_payload(
            {"parsed": None, "raw": object(), "parsing_error": ValueError("bad schema")}
        )
    except ValueError as exc:
        assert "structured output parsing failed" in str(exc)
    else:
        raise AssertionError("Expected ValueError for structured output parsing failure")


def test_invoke_structured_llm_suppresses_known_parsed_serializer_warning(monkeypatch):
    module = _load_preference_extractor_module(monkeypatch)

    class _HumanMessage:
        def __init__(self, content):
            self.content = content

    sys.modules["langchain_core.messages"].HumanMessage = _HumanMessage

    class _StructuredLLM:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            warnings.warn(
                "Pydantic serializer warnings:\n"
                "  PydanticSerializationUnexpectedValue(Expected `none` - "
                "serialized value may not be as expected [field_name='parsed'])",
                UserWarning,
            )
            return {"parsed": {"update": False}}

    structured_llm = _StructuredLLM()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = module._invoke_structured_llm(structured_llm, "hello")

    assert result == {"parsed": {"update": False}}
    assert len(structured_llm.calls) == 1
    assert structured_llm.calls[0][0].content == "hello"
    assert caught == []
