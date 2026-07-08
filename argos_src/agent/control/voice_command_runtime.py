"""Local voice-command arbitration for the realtime control runtime."""

from __future__ import annotations

from collections import deque
import time
from typing import Any


class VoiceCommandRuntime:
    """Suppress self-published commands and route operator voice intents."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def note_local_voice_command(self, command: str, *, ttl_sec: float = 1.5) -> None:
        rendered = str(command or "").strip().lower()
        if not rendered:
            return
        host = self._host
        expires_at = time.time() + max(0.1, ttl_sec)
        with host._turn_lock:
            self._prune_ignored_locked(now_s=time.time())
            host._ignored_voice_commands.append((rendered, expires_at))

    def should_ignore(self, command: str) -> bool:
        rendered = str(command or "").strip().lower()
        if not rendered:
            return False
        host = self._host
        now_s = time.time()
        with host._turn_lock:
            self._prune_ignored_locked(now_s=now_s)
            pending = deque()
            ignored = False
            while host._ignored_voice_commands:
                candidate, expires_at = host._ignored_voice_commands.popleft()
                if not ignored and candidate == rendered and expires_at > now_s:
                    ignored = True
                    break
                pending.append((candidate, expires_at))
            while pending:
                host._ignored_voice_commands.appendleft(pending.pop())
        return ignored

    def handle_message(self, msg: Any) -> None:
        raw_command = getattr(msg, "data", msg)
        command = str(raw_command or "").strip().lower()
        if not command:
            return
        host = self._host
        if self.should_ignore(command):
            host.logger.debug("Ignoring self-published voice command=%s", command)
            return
        if command == "stop":
            host.interrupt_current_response(reason="voice_command")

    def _prune_ignored_locked(self, *, now_s: float) -> None:
        commands = self._host._ignored_voice_commands
        while commands and commands[0][1] <= now_s:
            commands.popleft()
