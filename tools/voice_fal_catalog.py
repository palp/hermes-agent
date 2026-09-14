"""FAL.ai speech model catalog for ``tools.tts_tool_fal`` and ``tools.transcription_fal``.

Keys are fal endpoint ids (sibling of :mod:`tools.image_generation_catalog`). FAL's speech
schemas differ per endpoint — kokoro reads the text from ``prompt``, elevenlabs/tts/* from
``text``, minimax nests the voice under ``voice_setting.voice_id`` — so each entry maps Hermes's
unified inputs to that endpoint's fields as **dotted paths**, and :func:`build_payload` writes
the nesting. ``<tts|stt>.fal.models.<endpoint>`` in config.yaml overrides a curated entry or
declares a new one; an endpoint that is neither gets the common ``{text, voice, speed}`` shape.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_FAL_TTS_MODEL = "fal-ai/minimax/speech-02-hd"
DEFAULT_FAL_STT_MODEL = "fal-ai/wizper"

# Shape assumed for an endpoint that is neither curated nor declared in config.
_GENERIC_TTS: Dict[str, Any] = {
    "display": "", "text_field": "text", "voice_field": "voice", "speed_field": "speed",
    "prompt_field": None, "default_voice": None, "max_text_length": 5000, "output_ext": "mp3",
    "extra": {}, "stream": None,
}
_GENERIC_STT: Dict[str, Any] = {
    "display": "", "audio_field": "audio_url", "language_field": "language",
    "transcript_field": "text", "prompt_field": None, "extra": {},
}
# Keys a ``models.<endpoint>`` config block may set; anything else is ignored with a warning.
_TTS_ENTRY_KEYS = frozenset(_GENERIC_TTS)
_STT_ENTRY_KEYS = frozenset(_GENERIC_STT)


def _tts(
    display: str, *, text_field: str = "text", voice_field: Optional[str] = "voice",
    speed_field: Optional[str] = "speed", prompt_field: Optional[str] = None,
    default_voice: Optional[str] = None, max_text_length: int = 5000, output_ext: str = "mp3",
    extra: Optional[Dict[str, Any]] = None, stream: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One TTS entry. ``*_field`` are dotted payload paths; None means the endpoint has no such
    input and the value is dropped. ``stream`` is ``{"endpoint", "args", "sample_rate"}`` for
    endpoints with a chunked-PCM path."""
    return {
        "display": display, "text_field": text_field, "voice_field": voice_field,
        "speed_field": speed_field, "prompt_field": prompt_field, "default_voice": default_voice,
        "max_text_length": max_text_length, "output_ext": output_ext, "extra": dict(extra or {}),
        "stream": stream,
    }


def _stt(
    display: str, *, audio_field: str = "audio_url", language_field: Optional[str] = "language",
    transcript_field: str = "text", prompt_field: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One STT entry. ``language_field`` / ``prompt_field`` None means the endpoint has no such input."""
    return {
        "display": display, "audio_field": audio_field, "language_field": language_field,
        "transcript_field": transcript_field, "prompt_field": prompt_field,
        "extra": dict(extra or {}),
    }


# MiniMax documents a top-level ``output_format: hex`` default; in practice a bare request returns
# an ``audio`` URL like every other endpoint, so nothing pins it.
FAL_TTS_MODELS: Dict[str, Dict[str, Any]] = {
    # Default. The voice id matches Hermes's existing ``tts.minimax.voice_id`` default.
    "fal-ai/minimax/speech-02-hd": _tts(
        "MiniMax Speech-02 HD", voice_field="voice_setting.voice_id",
        speed_field="voice_setting.speed", default_voice="English_expressive_narrator"),
    "fal-ai/minimax/speech-02-turbo": _tts(
        "MiniMax Speech-02 Turbo", voice_field="voice_setting.voice_id",
        speed_field="voice_setting.speed", default_voice="English_expressive_narrator"),
    # Kokoro's text input is called ``prompt``.
    "fal-ai/kokoro": _tts("Kokoro", text_field="prompt", default_voice="af_heart", output_ext="wav"),
    "fal-ai/elevenlabs/tts/multilingual-v2": _tts(
        "ElevenLabs Multilingual v2", default_voice="Rachel", max_text_length=10000),
    "fal-ai/elevenlabs/tts/turbo-v2.5": _tts(
        "ElevenLabs Turbo v2.5", default_voice="Rachel", max_text_length=10000),
    "fal-ai/gemini-tts": _tts("Google Gemini TTS", default_voice="Kore", max_text_length=32000),
    "fal-ai/xai/tts/v1": _tts("xAI TTS v1", default_voice="eve", max_text_length=15000),
    # Steered by a prose ``prompt`` rather than a voice id; the only entry with a stream path
    # (raw chunked audio/pcm, int16 LE mono).
    "fal-ai/maya": _tts(
        "Maya", voice_field=None, speed_field=None, prompt_field="prompt",
        stream={"endpoint": "fal-ai/maya/stream",
                "args": {"output_format": "pcm", "sample_rate": "24 kHz"},
                "sample_rate": 24000}),
}

FAL_STT_MODELS: Dict[str, Dict[str, Any]] = {
    # Default: whisper-v3-large accuracy at ~2x the speed.
    "fal-ai/wizper": _stt("Wizper (Whisper v3 large, fal-optimized)"),
    "fal-ai/whisper": _stt("Whisper v3 large", prompt_field="prompt"),
    "fal-ai/elevenlabs/speech-to-text": _stt("ElevenLabs Scribe", language_field="language_code"),
    # Canary returns ``output`` rather than ``text``.
    "fal-ai/speech-to-text": _stt("NVIDIA Canary", language_field=None, transcript_field="output"),
}


def _declared_entries(fal_config: Dict[str, Any]) -> Dict[str, Any]:
    """``<tts|stt>.fal.models`` from config.yaml, or {} when absent/malformed."""
    declared = fal_config.get("models") if isinstance(fal_config, dict) else None
    return declared if isinstance(declared, dict) else {}


def _merge_entry(
    endpoint: str, base: Optional[Dict[str, Any]], override: Any, generic: Dict[str, Any],
    valid_keys: frozenset, kind: str,
) -> Dict[str, Any]:
    """Curated entry (or the generic shape) with a config-declared override merged over it."""
    entry = dict(base if base is not None else generic)
    if base is None and not isinstance(override, dict):
        logger.debug("FAL %s endpoint %r is not curated and declares no %s.fal.models.%s block; "
                     "assuming the common {text, voice, speed} shape.", kind.upper(), endpoint, kind, endpoint)
    if isinstance(override, dict):
        unknown = set(override) - valid_keys
        if unknown:
            logger.warning("Ignoring unknown key(s) %s in %s.fal.models.%s — valid keys are: %s",
                           ", ".join(sorted(unknown)), kind, endpoint, ", ".join(sorted(valid_keys)))
        entry.update({k: v for k, v in override.items() if k in valid_keys})
    if not entry.get("display"):
        entry["display"] = endpoint
    return entry


def resolve_tts_model(fal_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """``(endpoint, entry)`` for ``tts.fal.model`` (default :data:`DEFAULT_FAL_TTS_MODEL`)."""
    cfg = fal_config if isinstance(fal_config, dict) else {}
    endpoint = str(cfg.get("model") or DEFAULT_FAL_TTS_MODEL).strip() or DEFAULT_FAL_TTS_MODEL
    entry = _merge_entry(endpoint, FAL_TTS_MODELS.get(endpoint),
                         _declared_entries(cfg).get(endpoint), _GENERIC_TTS, _TTS_ENTRY_KEYS, "tts")
    return endpoint, entry


def resolve_stt_model(fal_config: Dict[str, Any], model: Optional[str] = None
                      ) -> Tuple[str, Dict[str, Any]]:
    """``(endpoint, entry)`` for STT; ``model`` (the dispatcher's resolved value) wins over config."""
    cfg = fal_config if isinstance(fal_config, dict) else {}
    endpoint = str(model or cfg.get("model") or DEFAULT_FAL_STT_MODEL).strip() or DEFAULT_FAL_STT_MODEL
    entry = _merge_entry(endpoint, FAL_STT_MODELS.get(endpoint),
                         _declared_entries(cfg).get(endpoint), _GENERIC_STT, _STT_ENTRY_KEYS, "stt")
    return endpoint, entry


def _assign(payload: Dict[str, Any], dotted: str, value: Any) -> None:
    """Write ``value`` at a dotted path, creating (or replacing non-dict) intermediate dicts."""
    keys = [k for k in dotted.split(".") if k]
    if not keys:
        return
    node = payload
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = node[key] = {}
        node = child
    node[keys[-1]] = value


def build_payload(entry: Dict[str, Any], values: Dict[str, Any],
                  extra_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """FAL ``arguments`` from an entry's field map. None values and unmapped fields are omitted so
    an endpoint never receives a parameter it would reject; precedence is entry ``extra`` <
    mapped values < ``extra_args``."""
    payload: Dict[str, Any] = {}
    for key, value in (entry.get("extra") or {}).items():
        _assign(payload, key, value)
    for logical, field_key in (
        ("text", "text_field"), ("voice", "voice_field"), ("speed", "speed_field"),
        ("prompt", "prompt_field"), ("audio", "audio_field"), ("language", "language_field"),
    ):
        field = entry.get(field_key)
        value = values.get(logical)
        if field and value is not None:
            _assign(payload, str(field), value)
    for key, value in (extra_args or {}).items():
        _assign(payload, str(key), value)
    return payload


def stream_config(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The entry's ``stream`` block when it names a chunked-PCM endpoint, else None."""
    stream = entry.get("stream")
    return stream if isinstance(stream, dict) and stream.get("endpoint") else None
