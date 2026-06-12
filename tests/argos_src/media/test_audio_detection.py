from argos_src.media.audio_detection import BaseVoiceDetectionModel


class _FakeDetectionModel(BaseVoiceDetectionModel):
    def detect(self, audio_data, input_parameters):
        metadata = dict(input_parameters)
        metadata["called"] = True
        metadata["samples"] = len(audio_data)
        return True, metadata

    def reset(self) -> None:
        pass


def test_base_voice_detection_model_delegates_call_to_detect():
    detected, metadata = _FakeDetectionModel()([1, 2, 3], {"source": "test"})

    assert detected is True
    assert metadata == {"source": "test", "called": True, "samples": 3}
