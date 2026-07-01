"""Audio capture, commit, and playback helpers for the Argos agent runtime."""

from __future__ import annotations

import audioop
import base64
from collections import deque
import queue
import threading
import time
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import sounddevice as sd

from argos_src.agent.realtime_turns import AUDIO_CHANNELS, QueuedTurn
from argos_src.observability.observability import perf_now
from argos_src.runtime.audio_admission import resolve_record_admission
from argos_src.speaker_recognition.policy import clip_stats

AUDIO_DTYPE = "int16"
VAD_SAMPLE_RATE = 16000
PLAYBACK_ECHO_SUPPRESSION_SEC = 0.8
RECORDING_PREROLL_SEC = 0.35
RECORDING_START_CONFIRMATION_BLOCKS = 2


class RealtimeAgentAudioMixin:
    def _ensure_preroll_buffer_locked(self) -> deque[tuple[float, bytes, bytes]]:
        buffer = getattr(self, "_recording_preroll_chunks", None)
        if buffer is None:
            buffer = deque()
            self._recording_preroll_chunks = buffer
        return buffer

    def _remember_preroll_chunk_locked(
        self,
        *,
        now_s: float,
        raw_chunk: bytes,
        audio_16k_pcm16: bytes,
    ) -> None:
        window_s = RECORDING_PREROLL_SEC
        buffer = self._ensure_preroll_buffer_locked()
        if window_s <= 0.0:
            buffer.clear()
            return
        buffer.append((now_s, raw_chunk, audio_16k_pcm16))
        cutoff = now_s - window_s
        while buffer and buffer[0][0] < cutoff:
            buffer.popleft()

    def _take_preroll_chunks_locked(self, *, now_s: float) -> list[tuple[bytes, bytes]]:
        window_s = RECORDING_PREROLL_SEC
        buffer = self._ensure_preroll_buffer_locked()
        cutoff = now_s - window_s
        chunks = [
            (raw_chunk, audio_16k_pcm16)
            for chunk_at, raw_chunk, audio_16k_pcm16 in buffer
            if chunk_at >= cutoff
        ]
        buffer.clear()
        return chunks

    def _set_recording_gesture_async(self, active: bool) -> None:
        gesture_runtime = getattr(self, "gesture_runtime", None)
        if gesture_runtime is None:
            return
        gesture_queue = getattr(self, "_recording_gesture_queue", None)
        if gesture_queue is None:
            gesture_queue = queue.Queue()
            self._recording_gesture_queue = gesture_queue
        gesture_lock = getattr(self, "_recording_gesture_lock", None)
        if gesture_lock is None:
            gesture_lock = threading.Lock()
            self._recording_gesture_lock = gesture_lock
        with gesture_lock:
            worker = getattr(self, "_recording_gesture_thread", None)
            if worker is None or not worker.is_alive():
                worker = threading.Thread(
                    target=self._recording_gesture_worker_loop,
                    daemon=True,
                )
                self._recording_gesture_thread = worker
                worker.start()
        gesture_queue.put(bool(active))

    def _recording_gesture_worker_loop(self) -> None:
        gesture_queue = getattr(self, "_recording_gesture_queue", None)
        if gesture_queue is None:
            return
        while not self._stop_event.is_set():
            try:
                active = gesture_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                gesture_runtime = getattr(self, "gesture_runtime", None)
                if gesture_runtime is not None:
                    gesture_runtime.set_recording_active(bool(active))
            except Exception:
                self.logger.exception(
                    "Failed to %s listening gesture",
                    "enable" if active else "finalize",
                )
            finally:
                gesture_queue.task_done()

    def _speaker_audio_debug_payload(
        self,
        *,
        raw_audio_pcm16: bytes,
        trimmed_audio_pcm16: bytes,
        capture_vad_positive_blocks: int,
    ) -> dict[str, float | int]:
        raw_waveform = np.frombuffer(raw_audio_pcm16 or b"", dtype=np.int16).copy()
        trimmed_waveform = np.frombuffer(trimmed_audio_pcm16 or b"", dtype=np.int16).copy()
        raw_stats = clip_stats(raw_waveform)
        trimmed_stats = clip_stats(trimmed_waveform)
        kept_ratio = (
            float(len(trimmed_audio_pcm16 or b"")) / float(len(raw_audio_pcm16 or b""))
            if raw_audio_pcm16
            else 0.0
        )
        return {
            "capture_vad_positive_blocks": int(capture_vad_positive_blocks),
            "vad_window_samples": int(getattr(self._vad, "window_size", 0) or 0),
            "raw_duration_s": round(float(raw_stats.duration_s), 3),
            "trimmed_duration_s": round(float(trimmed_stats.duration_s), 3),
            "kept_ratio": round(float(kept_ratio), 4),
            "raw_rms_level": round(float(raw_stats.rms_level), 1),
            "trimmed_rms_level": round(float(trimmed_stats.rms_level), 1),
            "raw_clipped_fraction": round(float(raw_stats.clipped_fraction), 6),
            "trimmed_clipped_fraction": round(float(trimmed_stats.clipped_fraction), 6),
        }

    def _log_speaker_audio_preprocessing(
        self,
        *,
        req_id: str,
        payload: dict[str, float | int],
    ) -> None:
        self.logger.info(
            "Speaker audio preprocessing req_id=%s raw_duration_s=%.3f "
            "trimmed_duration_s=%.3f kept_ratio=%.4f raw_rms=%.1f trimmed_rms=%.1f "
            "raw_clipped=%.6f trimmed_clipped=%.6f capture_vad_positive_blocks=%s "
            "vad_window_samples=%s",
            req_id,
            float(payload.get("raw_duration_s", 0.0) or 0.0),
            float(payload.get("trimmed_duration_s", 0.0) or 0.0),
            float(payload.get("kept_ratio", 0.0) or 0.0),
            float(payload.get("raw_rms_level", 0.0) or 0.0),
            float(payload.get("trimmed_rms_level", 0.0) or 0.0),
            float(payload.get("raw_clipped_fraction", 0.0) or 0.0),
            float(payload.get("trimmed_clipped_fraction", 0.0) or 0.0),
            int(payload.get("capture_vad_positive_blocks", 0) or 0),
            int(payload.get("vad_window_samples", 0) or 0),
        )

    def _start_audio_streams(self) -> None:
        self._input_stream = sd.InputStream(
            samplerate=self.realtime_profile.input_sample_rate,
            blocksize=self.realtime_profile.input_block_size,
            channels=AUDIO_CHANNELS,
            dtype=AUDIO_DTYPE,
            callback=self._capture_callback,
            device=self.realtime_profile.input_device,
        )
        self._output_stream = sd.OutputStream(
            samplerate=self.realtime_profile.output_sample_rate,
            blocksize=self.realtime_profile.input_block_size,
            channels=AUDIO_CHANNELS,
            dtype=AUDIO_DTYPE,
            callback=self._playback_callback,
            device=self.realtime_profile.output_device,
        )
        self._input_stream.start()
        self._output_stream.start()

    def _capture_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status:
            self.logger.warning("audio input status=%s", status)
        if self._stop_event.is_set():
            return
        if not self._session_ready.is_set():
            return

        raw_chunk = indata.copy().tobytes()
        try:
            resampled, self._resample_state = audioop.ratecv(
                raw_chunk,
                np.dtype(np.int16).itemsize,
                AUDIO_CHANNELS,
                self.realtime_profile.input_sample_rate,
                VAD_SAMPLE_RATE,
                self._resample_state,
            )
        except Exception:
            self.logger.exception("Failed to resample audio input")
            return
        audio_16k = np.frombuffer(resampled, dtype=np.int16)

        try:
            voice_detected, _ = self._vad(audio_16k, {})
        except Exception:
            voice_detected = False
        try:
            wake_detected, wake_output = self._wake_word(audio_16k, {})
        except Exception:
            wake_detected, wake_output = False, {}
        self._log_wakeword_debug(wake_detected=wake_detected, wake_output=wake_output)

        now = time.time()
        interaction = self.engagement.snapshot()
        interaction_state = interaction.state
        playback_guard_active = self._input_playback_guard_active(now_s=now)
        if (interaction_state == "speaking" or playback_guard_active) and wake_detected:
            self.engagement.on_face_or_wake()
            self.interrupt_current_response(reason="wake_word_interrupt")
            return
        if playback_guard_active:
            return

        with self._recording_lock:
            if not self._recording_active:
                is_attention_present = getattr(
                    self._face_gate,
                    "is_attention_present",
                    lambda: False,
                )
                allowed, admission_reason, wake_until = resolve_record_admission(
                    face_present=self._face_gate.is_face_present(),
                    attention_present=bool(is_attention_present()),
                    interaction_state=interaction_state,
                    now_s=now,
                    wake_window_until_s=self._wake_window_until,
                    wake_detected=bool(wake_detected),
                    wake_window_sec=self.realtime_profile.wake_window_sec,
                    block_during_speaking=self.realtime_profile.admission.block_during_speaking,
                    block_during_engaged=bool(
                        getattr(
                            self.realtime_profile.admission,
                            "block_during_engaged",
                            False,
                        )
                    ),
                    open_on_face_presence=self.realtime_profile.admission.open_on_face_presence,
                    open_on_attention_presence=bool(
                        getattr(
                            self.realtime_profile.admission,
                            "open_on_attention_presence",
                            False,
                        )
                    ),
                    open_on_interaction_states=self.realtime_profile.admission.open_on_interaction_states,
                    open_on_wake_window=self.realtime_profile.admission.open_on_wake_window,
                    nav_active=interaction.nav_active,
                    nav_interruptible=interaction.nav_interruptible,
                    nav_passive_listen_allowed=interaction.nav_passive_listen_allowed,
                )
                self._wake_window_until = wake_until
                if allowed:
                    display_mode = getattr(self, "_set_display_mode_async", None)
                    display_state_still_current = True
                    if interaction_state in {"alert", "cooldown"}:
                        current_state = str(
                            getattr(self.engagement, "state_name", interaction_state) or ""
                        ).strip()
                        display_state_still_current = current_state == interaction_state
                    if callable(display_mode) and display_state_still_current:
                        display_mode("alert")
                else:
                    clear_passive_alert = getattr(
                        self,
                        "_clear_passive_alert_display_if_needed",
                        None,
                    )
                    if callable(clear_passive_alert):
                        clear_passive_alert()
                if allowed and voice_detected:
                    self._candidate_voice_blocks = (
                        int(getattr(self, "_candidate_voice_blocks", 0) or 0) + 1
                    )
                    confirmation_blocks = (
                        1 if wake_detected else RECORDING_START_CONFIRMATION_BLOCKS
                    )
                    if self._candidate_voice_blocks >= confirmation_blocks:
                        confirmed_voice_blocks = int(self._candidate_voice_blocks)
                        pre_roll_chunks = self._take_preroll_chunks_locked(now_s=now)
                        self._start_recording_locked(
                            now_s=now,
                            admission_reason=admission_reason,
                            interaction_state=interaction_state,
                            wake_detected=bool(wake_detected),
                        )
                        for pre_raw_chunk, pre_audio_16k_pcm16 in pre_roll_chunks:
                            self._audio_send_queue.put(pre_raw_chunk)
                            self._current_turn_audio_chunks.append(pre_audio_16k_pcm16)
                        self._audio_send_queue.put(raw_chunk)
                        self._current_turn_audio_chunks.append(resampled)
                        self._current_turn_vad_positive_blocks = confirmed_voice_blocks
                        self._last_voice_at = now
                    else:
                        self._remember_preroll_chunk_locked(
                            now_s=now,
                            raw_chunk=raw_chunk,
                            audio_16k_pcm16=resampled,
                        )
                else:
                    self._candidate_voice_blocks = 0
                    self._remember_preroll_chunk_locked(
                        now_s=now,
                        raw_chunk=raw_chunk,
                        audio_16k_pcm16=resampled,
                    )
                return

            self._audio_send_queue.put(raw_chunk)
            self._current_turn_audio_chunks.append(resampled)
            if voice_detected:
                self._current_turn_vad_positive_blocks += 1
                self._last_voice_at = now
                return

            if (now - self._last_voice_at) >= self.realtime_profile.silence_grace_period:
                self._finalize_recording_locked(now_s=now)

    def _start_recording_locked(
        self,
        *,
        now_s: float,
        admission_reason: str = "",
        interaction_state: str = "",
        wake_detected: bool = False,
    ) -> None:
        self._recording_active = True
        self._recording_started_at = now_s
        self._last_voice_at = now_s
        self._cancel_preference_idle_flush()
        self._current_primary_face_person_id = self._get_current_primary_face_person_id()
        self._current_visible_face_person_ids = self._get_current_visible_face_person_ids()
        self._current_turn_audio_chunks = []
        self._current_turn_vad_positive_blocks = 0
        self._candidate_voice_blocks = 0
        self._set_recording_gesture_async(True)
        display_mode = getattr(self, "_set_display_mode_async", None)
        if callable(display_mode):
            display_mode("recording")
        try:
            self._send_event({"type": "input_audio_buffer.clear"})
        except Exception:
            self.logger.exception("Failed to clear input audio buffer")
        self.logger.info(
            "Audio recording started admission_reason=%s interaction_state=%s "
            "wake_detected=%s primary_face_person_id=%s visible_face_person_ids=%s",
            admission_reason or "unknown",
            interaction_state or "unknown",
            bool(wake_detected),
            self._current_primary_face_person_id,
            ",".join(self._current_visible_face_person_ids) or "<none>",
        )
        self._latency.emit(event="recording_started")

    def _finalize_recording_locked(self, *, now_s: float) -> None:
        self._recording_active = False
        self._recording_started_at = 0.0
        primary_face_person_id = self._current_primary_face_person_id
        visible_face_person_ids = self._current_visible_face_person_ids
        capture_vad_positive_blocks = int(self._current_turn_vad_positive_blocks)
        self._current_primary_face_person_id = None
        self._current_visible_face_person_ids = ()
        audio_pcm16 = b"".join(self._current_turn_audio_chunks)
        self._current_turn_audio_chunks = []
        self._current_turn_vad_positive_blocks = 0
        self._candidate_voice_blocks = 0
        self._set_recording_gesture_async(False)
        display_mode = getattr(self, "_set_display_mode_async", None)
        if callable(display_mode):
            display_mode("thinking")
        speech_end_perf_s = perf_now()
        speech_end_unix_s = now_s
        self._latency.emit(event="speech_end", speech_end_unix_s=speech_end_unix_s)
        threading.Thread(
            target=self._commit_audio_turn,
            args=(
                primary_face_person_id,
                visible_face_person_ids,
                audio_pcm16,
                capture_vad_positive_blocks,
                speech_end_perf_s,
                speech_end_unix_s,
            ),
            daemon=True,
        ).start()

    def _commit_audio_turn(
        self,
        primary_face_person_id: Optional[str],
        visible_face_person_ids: tuple[str, ...],
        audio_pcm16: bytes,
        capture_vad_positive_blocks: int,
        speech_end_perf_s: float,
        speech_end_unix_s: float,
    ) -> None:
        try:
            self._audio_send_queue.join()
            self._send_event({"type": "input_audio_buffer.commit"})
        except Exception:
            self.logger.exception("Failed to commit input audio buffer")
            return
        display_mode = getattr(self, "_set_display_mode_async", None)
        if callable(display_mode):
            display_mode("thinking")
        transcript_perf_s = perf_now()
        req_id = f"rt-{uuid4().hex[:12]}"
        self._latency.emit(event="audio_commit", req_id=req_id)
        if self.coalescer is not None:
            internal_text, merged_meta = self.coalescer.drain_internal_events_for_audio_turn(
                {
                    "req_id": req_id,
                    "capture_vad_positive_blocks": int(capture_vad_positive_blocks),
                    "speech_end_perf_s": speech_end_perf_s,
                    "speech_end_unix_s": speech_end_unix_s,
                    "transcript_perf_s": transcript_perf_s,
                }
            )
        else:
            internal_text, merged_meta = None, {
                "req_id": req_id,
                "capture_vad_positive_blocks": int(capture_vad_positive_blocks),
                "speech_end_perf_s": speech_end_perf_s,
                "speech_end_unix_s": speech_end_unix_s,
                "transcript_perf_s": transcript_perf_s,
            }
        # Speaker recognition now uses the raw 16 kHz turn audio. The eval repo
        # showed raw ECAPA embeddings with per-person centroids outperforming
        # the earlier duration/VAD-gated path on collected robot samples.
        trimmed_audio_pcm16 = audio_pcm16
        speaker_audio_debug = self._speaker_audio_debug_payload(
            raw_audio_pcm16=audio_pcm16,
            trimmed_audio_pcm16=trimmed_audio_pcm16,
            capture_vad_positive_blocks=capture_vad_positive_blocks,
        )
        merged_meta["speaker_audio_debug"] = speaker_audio_debug
        self._log_speaker_audio_preprocessing(
            req_id=req_id,
            payload=speaker_audio_debug,
        )

        resolution = self._face_owner_resolution(
            primary_face_person_id=primary_face_person_id,
            visible_face_person_ids=visible_face_person_ids,
        )
        if self.speaker_service is not None and trimmed_audio_pcm16 is not None and trimmed_audio_pcm16:
            try:
                resolution = self.speaker_service.resolve_turn_owner(
                    audio_pcm16=trimmed_audio_pcm16,
                    primary_face_person_id=primary_face_person_id,
                    visible_face_person_ids=visible_face_person_ids,
                )
            except Exception:
                self.logger.exception("Speaker resolution failed req_id=%s", req_id)
                resolution = self._face_owner_resolution(
                    primary_face_person_id=primary_face_person_id,
                    visible_face_person_ids=visible_face_person_ids,
                )

        turn = QueuedTurn(
            kind="audio",
            req_id=req_id,
            speech_end_perf_s=speech_end_perf_s,
            speech_end_unix_s=speech_end_unix_s,
            transcript_perf_s=transcript_perf_s,
            primary_face_person_id=primary_face_person_id,
            audio_speaker_id=resolution.audio_speaker_id,
            owner_id=resolution.owner_id,
            owner_source=resolution.owner_source,
            owner_confidence=resolution.owner_confidence,
            speaker_visible=resolution.speaker_visible,
            pending_internal_text=internal_text,
            metadata=merged_meta,
            context_snapshot=self._capture_turn_context(
                primary_face_person_id=primary_face_person_id,
                audio_speaker_id=resolution.audio_speaker_id,
                owner_id=resolution.owner_id,
                owner_source=resolution.owner_source,
                owner_confidence=resolution.owner_confidence,
                speaker_visible=resolution.speaker_visible,
            ),
            input_audio_pcm16=audio_pcm16,
            trimmed_input_audio_pcm16=trimmed_audio_pcm16,
        )
        self._log_speaker_resolution(
            req_id=req_id,
            phase="initial",
            primary_face_person_id=primary_face_person_id,
            result=resolution,
        )
        owner_turn_controller = getattr(self, "owner_turn_controller", None)
        if owner_turn_controller is not None:
            try:
                owner_turn_controller.request_turn(
                    person_id=resolution.owner_id,
                    req_id=req_id,
                    owner_source=resolution.owner_source,
                )
            except Exception:
                self.logger.exception("Failed to queue owner turn req_id=%s", req_id)
        with self._turn_lock:
            self._turns_by_req_id[turn.req_id] = turn
        self._supersede_unanswered_turn(turn)
        self._register_pending_audio_turn(turn)
        self.engagement.on_human_input(req_id)
        self._last_external_input_s = time.time()
        self._turn_queue.put(turn)

    def _playback_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del time_info
        if status:
            self.logger.warning("audio output status=%s", status)
        audio, actual_frames = self._playback_buffer.pop_frames(frames)
        outdata[:] = audio
        if actual_frames <= 0:
            return
        self._input_suppressed_until_s = max(
            float(getattr(self, "_input_suppressed_until_s", 0.0) or 0.0),
            time.time() + PLAYBACK_ECHO_SUPPRESSION_SEC,
        )
        with self._turn_lock:
            if self._playback_req_id:
                self._played_output_frames += actual_frames
                turn = self._turns_by_req_id.get(self._playback_req_id)
                if turn is not None:
                    turn.last_playback_progress_at = time.time()

    def _audio_sender_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_send_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._send_event(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            finally:
                self._audio_send_queue.task_done()

    def _input_playback_guard_active(self, *, now_s: float | None = None) -> bool:
        """Suppress mic admission while assistant audio may still be in the room."""
        now = time.time() if now_s is None else float(now_s)
        if now < float(getattr(self, "_input_suppressed_until_s", 0.0) or 0.0):
            return True
        try:
            if self._playback_buffer.buffered_frames() > 0:
                return True
        except Exception:
            return True
        with self._turn_lock:
            return bool(self._playback_req_id)
