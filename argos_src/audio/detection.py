"""Wake-word and VAD models used by the Argos realtime runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray


class BaseVoiceDetectionModel(ABC):
    """Small callable interface shared by Argos voice-detection models."""

    def __call__(
        self,
        audio_data: NDArray,
        input_parameters: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        return self.detect(audio_data, input_parameters)

    @abstractmethod
    def detect(
        self,
        audio_data: NDArray,
        input_parameters: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Return whether the model detected its event and updated metadata."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any model-local state."""


class SileroVAD(BaseVoiceDetectionModel):
    """Voice activity detector backed by the Silero VAD Torch Hub model."""

    def __init__(
        self,
        sampling_rate: Literal[8000, 16000] = 16000,
        threshold: float = 0.5,
    ) -> None:
        self.model_name = "silero_vad"
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model=self.model_name,
        )
        if sampling_rate == 16000:
            self.sampling_rate = 16000
            self.window_size = 512
        elif sampling_rate == 8000:
            self.sampling_rate = 8000
            self.window_size = 256
        else:
            raise ValueError("Only 8000 and 16000 sampling rates are supported")
        self.threshold = float(threshold)

    def _int2float(self, sound: NDArray[np.int16]) -> NDArray[np.float32]:
        converted = sound.astype("float32")
        converted *= 1 / 32768
        return converted.squeeze()

    def detect(
        self,
        audio_data: NDArray,
        input_parameters: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        window = audio_data[-self.window_size :]
        vad_confidence = self.model(
            torch.tensor(self._int2float(window)),
            self.sampling_rate,
        ).item()
        metadata = input_parameters.copy()
        metadata.update({self.model_name: {"vad_confidence": vad_confidence}})
        return vad_confidence > self.threshold, metadata

    def reset(self) -> None:
        self.model.reset()


class OpenWakeWord(BaseVoiceDetectionModel):
    """Wake-word detector backed by OpenWakeWord ONNX inference."""

    def __init__(self, wake_word_model_path: str, threshold: float = 0.1) -> None:
        from openwakeword.model import Model as OWWModel
        from openwakeword.utils import download_models

        self.model_name = "open_wake_word"
        download_models()
        self.model = OWWModel(
            wakeword_models=[wake_word_model_path],
            inference_framework="onnx",
        )
        self.threshold = float(threshold)

    def detect(
        self,
        audio_data: NDArray,
        input_parameters: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        predictions = self.model.predict(audio_data)
        metadata = input_parameters.copy()
        metadata.update({self.model_name: {"predictions": predictions}})
        if not isinstance(predictions, dict):
            raise TypeError(
                f"Unexpected OpenWakeWord prediction payload: {type(predictions)}"
            )
        for value in predictions.values():
            if value > self.threshold:
                self.model.reset()
                return True, metadata
        return False, metadata

    def reset(self) -> None:
        self.model.reset()
