from argos_src.observability.pricing import (
    estimate_realtime_response_cost,
    estimate_text_generation_cost,
    estimate_transcription_cost,
)


def test_estimate_realtime_response_cost_uses_modality_specific_rates():
    fields = estimate_realtime_response_cost(
        {
            "input_tokens": 200,
            "output_tokens": 40,
            "input_token_details": {
                "text_tokens": 190,
                "audio_tokens": 10,
                "cached_tokens": 120,
                "cached_tokens_details": {
                    "text_tokens": 110,
                    "audio_tokens": 10,
                },
            },
            "output_token_details": {
                "text_tokens": 8,
                "audio_tokens": 32,
            },
        },
        model_name="gpt-realtime-1.5",
    )

    assert fields["uncached_input_text_tokens"] == 80
    assert fields["uncached_input_audio_tokens"] == 0
    assert fields["estimated_cost_usd"] == 0.002544
    assert fields["estimated_cached_savings_usd"] == 0.000712


def test_estimate_transcription_cost_uses_audio_input_and_text_output():
    fields = estimate_transcription_cost(
        {
            "input_tokens": 120,
            "output_tokens": 20,
            "input_token_details": {"audio_tokens": 120},
            "output_token_details": {"text_tokens": 20},
        },
        model_name="gpt-4o-mini-transcribe",
    )

    assert fields["input_audio_tokens"] == 120
    assert fields["output_text_tokens"] == 20
    assert fields["estimated_cost_usd"] == 0.00025


def test_estimate_text_generation_cost_uses_cached_input_rate():
    fields = estimate_text_generation_cost(
        {
            "input_tokens": 1000,
            "output_tokens": 50,
            "total_tokens": 1050,
            "input_token_details": {"cache_read": 800},
        },
        model_name="gpt-4.1-mini",
    )

    assert fields["cached_tokens"] == 800
    assert fields["uncached_input_tokens"] == 200
    assert fields["estimated_cost_usd"] == 0.00024
    assert fields["estimated_cached_savings_usd"] == 0.00024
