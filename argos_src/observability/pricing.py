"""Lightweight cost estimation helpers for Argos observability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TokenRates:
    input_per_million: float = 0.0
    cached_input_per_million: float = 0.0
    output_per_million: float = 0.0


@dataclass(frozen=True)
class ModelRates:
    text: TokenRates = TokenRates()
    audio: TokenRates = TokenRates()
    image: TokenRates = TokenRates()


GPT_REALTIME_1_5_RATES = ModelRates(
    text=TokenRates(input_per_million=4.0, cached_input_per_million=0.4, output_per_million=16.0),
    audio=TokenRates(input_per_million=32.0, cached_input_per_million=0.4, output_per_million=64.0),
    image=TokenRates(input_per_million=5.0, cached_input_per_million=0.5, output_per_million=0.0),
)

GPT_REALTIME_2_1_RATES = ModelRates(
    text=TokenRates(input_per_million=4.0, cached_input_per_million=0.4, output_per_million=24.0),
    audio=TokenRates(input_per_million=32.0, cached_input_per_million=0.4, output_per_million=64.0),
    image=TokenRates(input_per_million=5.0, cached_input_per_million=0.5, output_per_million=0.0),
)

GPT_REALTIME_2_1_MINI_RATES = ModelRates(
    text=TokenRates(input_per_million=0.6, cached_input_per_million=0.06, output_per_million=2.4),
    audio=TokenRates(input_per_million=10.0, cached_input_per_million=0.3, output_per_million=20.0),
    image=TokenRates(input_per_million=0.8, cached_input_per_million=0.08, output_per_million=0.0),
)

GPT_4O_TRANSCRIBE_RATES = ModelRates(
    text=TokenRates(output_per_million=10.0),
    audio=TokenRates(input_per_million=2.5),
)

GPT_4O_MINI_TRANSCRIBE_RATES = ModelRates(
    text=TokenRates(output_per_million=5.0),
    audio=TokenRates(input_per_million=1.25),
)

GPT_4_1_MINI_RATES = ModelRates(
    text=TokenRates(input_per_million=0.4, cached_input_per_million=0.1, output_per_million=1.6),
)

MODEL_RATES_BY_PREFIX: tuple[tuple[str, ModelRates], ...] = tuple(
    sorted(
        {
            "gpt-realtime-2.1-mini": GPT_REALTIME_2_1_MINI_RATES,
            "gpt-realtime-mini": GPT_REALTIME_2_1_MINI_RATES,
            "gpt-realtime-2.1": GPT_REALTIME_2_1_RATES,
            "gpt-realtime-2": GPT_REALTIME_2_1_RATES,
            "gpt-realtime-1.5": GPT_REALTIME_1_5_RATES,
            "gpt-realtime": GPT_REALTIME_2_1_RATES,
            "gpt-4o-mini-transcribe": GPT_4O_MINI_TRANSCRIBE_RATES,
            "gpt-4o-transcribe": GPT_4O_TRANSCRIBE_RATES,
            "gpt-4.1-mini": GPT_4_1_MINI_RATES,
        }.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def _token_cost(token_count: int, rate_per_million: float) -> float:
    if token_count <= 0 or rate_per_million <= 0:
        return 0.0
    return float(token_count) * float(rate_per_million) / 1_000_000.0


def _round_cost(value: float) -> float:
    return round(float(value), 8)


def resolve_model_rates(model_name: Optional[str]) -> Optional[ModelRates]:
    model = str(model_name or "").strip().lower()
    if not model:
        return None
    for prefix, rates in MODEL_RATES_BY_PREFIX:
        if model == prefix or model.startswith(f"{prefix}-"):
            return rates
    return None


def estimate_realtime_response_cost(
    usage: dict[str, Any],
    *,
    model_name: Optional[str],
) -> dict[str, Any]:
    rates = resolve_model_rates(model_name)
    usage = _as_dict(usage)
    input_details = _as_dict(usage.get("input_token_details"))
    output_details = _as_dict(usage.get("output_token_details"))
    cached_details = _as_dict(input_details.get("cached_tokens_details"))

    input_text_tokens = _to_int(input_details.get("text_tokens"))
    input_audio_tokens = _to_int(input_details.get("audio_tokens"))
    input_image_tokens = _to_int(input_details.get("image_tokens"))
    output_text_tokens = _to_int(output_details.get("text_tokens"))
    output_audio_tokens = _to_int(output_details.get("audio_tokens"))

    input_tokens = _to_int(usage.get("input_tokens"))
    output_tokens = _to_int(usage.get("output_tokens"))
    cached_text_tokens = _to_int(cached_details.get("text_tokens"))
    cached_audio_tokens = _to_int(cached_details.get("audio_tokens"))
    cached_image_tokens = _to_int(cached_details.get("image_tokens"))
    cached_tokens = _to_int(input_details.get("cached_tokens"))

    if input_tokens > 0 and (input_text_tokens + input_audio_tokens + input_image_tokens) == 0:
        known_cached_total = cached_text_tokens + cached_audio_tokens + cached_image_tokens
        input_text_tokens = cached_text_tokens
        input_audio_tokens = cached_audio_tokens
        input_image_tokens = cached_image_tokens
        if known_cached_total > 0:
            input_text_tokens += max(0, input_tokens - known_cached_total)
        else:
            input_text_tokens = input_tokens
    if output_tokens > 0 and (output_text_tokens + output_audio_tokens) == 0:
        output_text_tokens = output_tokens

    uncached_input_text_tokens = max(0, input_text_tokens - cached_text_tokens)
    uncached_input_audio_tokens = max(0, input_audio_tokens - cached_audio_tokens)
    uncached_input_image_tokens = max(0, input_image_tokens - cached_image_tokens)

    estimated_cost_usd = None
    estimated_cached_savings_usd = None
    if rates is not None:
        estimated_cost_usd = _round_cost(
            _token_cost(uncached_input_text_tokens, rates.text.input_per_million)
            + _token_cost(cached_text_tokens, rates.text.cached_input_per_million)
            + _token_cost(output_text_tokens, rates.text.output_per_million)
            + _token_cost(uncached_input_audio_tokens, rates.audio.input_per_million)
            + _token_cost(cached_audio_tokens, rates.audio.cached_input_per_million)
            + _token_cost(output_audio_tokens, rates.audio.output_per_million)
            + _token_cost(uncached_input_image_tokens, rates.image.input_per_million)
            + _token_cost(cached_image_tokens, rates.image.cached_input_per_million)
        )
        estimated_cached_savings_usd = _round_cost(
            _token_cost(
                cached_text_tokens,
                max(0.0, rates.text.input_per_million - rates.text.cached_input_per_million),
            )
            + _token_cost(
                cached_audio_tokens,
                max(0.0, rates.audio.input_per_million - rates.audio.cached_input_per_million),
            )
            + _token_cost(
                cached_image_tokens,
                max(0.0, rates.image.input_per_million - rates.image.cached_input_per_million),
            )
        )

    return {
        "input_text_tokens": input_text_tokens,
        "input_audio_tokens": input_audio_tokens,
        "input_image_tokens": input_image_tokens,
        "output_text_tokens": output_text_tokens,
        "output_audio_tokens": output_audio_tokens,
        "cached_tokens": cached_tokens,
        "cached_text_tokens": cached_text_tokens,
        "cached_audio_tokens": cached_audio_tokens,
        "cached_image_tokens": cached_image_tokens,
        "uncached_input_text_tokens": uncached_input_text_tokens,
        "uncached_input_audio_tokens": uncached_input_audio_tokens,
        "uncached_input_image_tokens": uncached_input_image_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "estimated_cached_savings_usd": estimated_cached_savings_usd,
    }


def estimate_transcription_cost(
    usage: dict[str, Any],
    *,
    model_name: Optional[str],
) -> dict[str, Any]:
    rates = resolve_model_rates(model_name)
    usage = _as_dict(usage)
    input_details = _as_dict(usage.get("input_token_details"))
    output_details = _as_dict(usage.get("output_token_details"))

    input_audio_tokens = _to_int(input_details.get("audio_tokens"))
    output_text_tokens = _to_int(output_details.get("text_tokens"))
    input_tokens = _to_int(usage.get("input_tokens"))
    output_tokens = _to_int(usage.get("output_tokens"))

    if input_audio_tokens <= 0:
        input_audio_tokens = input_tokens
    if output_text_tokens <= 0:
        output_text_tokens = output_tokens

    estimated_cost_usd = None
    if rates is not None:
        estimated_cost_usd = _round_cost(
            _token_cost(input_audio_tokens, rates.audio.input_per_million)
            + _token_cost(output_text_tokens, rates.text.output_per_million)
        )

    return {
        "input_audio_tokens": input_audio_tokens,
        "output_text_tokens": output_text_tokens,
        "estimated_cost_usd": estimated_cost_usd,
    }


def estimate_text_generation_cost(
    usage: dict[str, Any],
    *,
    model_name: Optional[str],
) -> dict[str, Any]:
    rates = resolve_model_rates(model_name)
    usage = _as_dict(usage)
    input_details = _as_dict(usage.get("input_token_details"))

    input_tokens = _to_int(usage.get("input_tokens"))
    output_tokens = _to_int(usage.get("output_tokens"))
    total_tokens = _to_int(usage.get("total_tokens")) or (input_tokens + output_tokens)
    cached_tokens = _to_int(input_details.get("cache_read") or input_details.get("cached_tokens"))
    uncached_input_tokens = max(0, input_tokens - cached_tokens)

    estimated_cost_usd = None
    estimated_cached_savings_usd = None
    if rates is not None:
        estimated_cost_usd = _round_cost(
            _token_cost(uncached_input_tokens, rates.text.input_per_million)
            + _token_cost(cached_tokens, rates.text.cached_input_per_million)
            + _token_cost(output_tokens, rates.text.output_per_million)
        )
        estimated_cached_savings_usd = _round_cost(
            _token_cost(
                cached_tokens,
                max(0.0, rates.text.input_per_million - rates.text.cached_input_per_million),
            )
        )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "estimated_cached_savings_usd": estimated_cached_savings_usd,
    }
