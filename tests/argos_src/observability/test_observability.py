from __future__ import annotations

from argos_src.observability.observability import LatencyLogger


def test_latency_logger_can_omit_fields_from_console_only(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    log_path = tmp_path / "latency.log"
    monkeypatch.setenv("GO2_LATENCY_LOG_PATH", str(log_path))
    monkeypatch.setenv("GO2_LATENCY_CONSOLE", "1")

    logger = LatencyLogger("realtime")
    logger.emit(
        event="response_create",
        req_id="rt-1",
        model_dynamic_context_b64="bloody-numbers",
        model_dynamic_context_chars=14,
        _console_omit_fields=("model_dynamic_context_b64",),
    )

    file_line = log_path.read_text(encoding="utf-8")
    console_line = capsys.readouterr().out

    assert "model_dynamic_context_b64=bloody-numbers" in file_line
    assert "model_dynamic_context_chars=14" in file_line
    assert "model_dynamic_context_b64" not in console_line
    assert "model_dynamic_context_chars=14" in console_line
