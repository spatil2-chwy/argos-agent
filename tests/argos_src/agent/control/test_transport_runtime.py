from __future__ import annotations

from types import SimpleNamespace
import threading

from argos_src.agent.control.transport_runtime import TransportRuntime


def test_transport_stringifies_structured_tool_output_as_json() -> None:
    assert TransportRuntime.stringify_tool_output({"ok": True}) == '{"ok": true}'


def test_transport_closed_socket_stops_runtime() -> None:
    class _ClosedSocket:
        def send(self, _payload):
            raise RuntimeError("socket is already closed.")

    host = SimpleNamespace(
        _ws=_ClosedSocket(),
        _ws_lock=threading.Lock(),
        _stop_event=threading.Event(),
        logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
    )

    TransportRuntime(host).send_event({"type": "response.create"})

    assert host._stop_event.is_set()
    assert host._ws is None
