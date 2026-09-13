"""FAL.ai speech model catalog for ``tools.tts_tool_fal`` and ``tools.transcription_fal``.

Sibling of :mod:`tools.image_generation_catalog`: dict keys ARE the fal endpoint ids. Each entry
translates Hermes's unified inputs (text/voice/speed, or audio_url/language) into that endpoint's
native payload, because fal's speech schemas are genuinely inconsistent — ``fal-ai/kokoro`` reads
``prompt``, ``fal-ai/elevenlabs/tts/*`` reads ``text``, and ``fal-ai/minimax/*`` nests the voice
under ``voice_setting.voice_id``. Field names are therefore **dotted paths** written by
:func:`build_payload`, so a nested payload needs no per-model code.

The catalog is curated but not closed: ``tts.fal.models.<endpoint>`` / ``stt.fal.models.<endpoint>``
in config.yaml overrides a curated entry or declares a brand-new one, and an endpoint that is
neither curated nor declared falls back to the common ``{text, voice, speed}`` shape. So reaching a
fal endpoint that shipped after this file is a config edit, not a code change.

TTS output is uniform across fal (``{"audio": {"url", ...}}``); STT output is not, hence
``transcript_field`` (``fal-ai/speech-to-text`` returns ``output``, everything else ``text``).
Pricing and voice lists drift — nothing here is asserted against the live API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_FAL_TTS_MODEL = "fal-ai/minimax/speech-02-hd"
DEFAULT_FAL_STT_MODEL = "fal-ai/wizper"

# Applied to an endpoint that is neither curated nor declared in config. Matches the most common
# fal TTS shape; a model that differs is reachable by declaring the fields in config.yaml.
_GENERIC_TTS: Dict[str, Any] = {
    "display": "", "text_field": "text", "voice_field": "voice", "speed_field": "speed",
    "prompt_field": None, "default_voice": None, "max_text_length": 5000, "output_ext": "mp3",
    "extra": {}, "stream": None,
}
_GENERIC_STT: Dict[str, Any] = {
    "display": "", "audio_field": "audio_url", "language_field": "language",
    "transcript_field": "text", "prompt_field": None, "extra": {},
}

# Config keys a user may set inside ``models.<endpoint>``. Anything else is ignored with a warning,
# so a typo surfaces instead of silently doing nothing.
_TTS_ENTRY_KEYS = frozenset(_GENERIC_TTS)
_STT_ENTRY_KEYS = frozenset(_GENERIC_STT)


def _tts(
    display: str, *, text_field: str = "text", voice_field: Optional[str] = "voice",
    speed_field: Optional[str] = "speed", prompt_field: Optional[str] = None,
    default_voice: Optional[str] = None, max_text_length: int = 5000, output_ext: str = "mp3",
    extra: Optional[Dict[str, Any]] = None, stream: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One TTS catalog entry.

    ``*_field`` values are dotted payload paths (``voice_setting.voice_id``); None means the
    endpoint has no such input and the value is dropped rather than sent and rejected.
    ``prompt_field`` is for endpoints steered by a prose voice description (maya) instead of a
    voice id. ``extra`` is constant payload additions. ``stream`` is
    ``{"endpoint": ..., "args": {...}}`` for endpoints exposing a chunked-PCM stream; None means
    this model has no chunked API and the sync path handles it.
    """
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
    """One STT catalog entry.

    ``language_field`` None means the endpoint takes no language hint; ``prompt_field`` None means
    it accepts no vocabulary prompt, so ``stt.prompt`` / a ``pre_transcription`` hook is dropped
    (and logged) instead of being sent and rejected.
    """
    return {
        "display": display, "audio_field": audio_field, "language_field": language_field,
        "transcript_field": transcript_field, "prompt_field": prompt_field,
        "extra": dict(extra or {}),
    }


# ── TTS ───────────────────────────────────────────────────────────────────────
# Verified against the live API where noted. MiniMax's documented top-level
# ``output_format: url|hex`` (default hex) does NOT apply in practice — a bare {"text": ...} returns
# a normal ``audio`` File object — so no entry pins it.
FAL_TTS_MODELS: Dict[str, Dict[str, Any]] = {
    # Default. Probed: voice_setting.voice_id + voice_setting.speed are both honoured, output is
    # always audio/mpeg. Voice id matches Hermes's existing ``tts.minimax.voice_id`` default.
    "fal-ai/minimax/speech-02-hd": _tts(
        "MiniMax Speech-02 HD", voice_field="voice_setting.voice_id",
        speed_field="voice_setting.speed", default_voice="English_expressive_narrator",
        max_text_length=5000, output_ext="mp3"),
    "fal-ai/minimax/speech-02-turbo": _tts(
        "MiniMax Speech-02 Turbo", voice_field="voice_setting.voice_id",
        speed_field="voice_setting.speed", default_voice="English_expressive_narrator",
        max_text_length=5000, output_ext="mp3"),
    # Kokoro is the odd one: the text input is called ``prompt``, not ``text``.
    "fal-ai/kokoro": _tts(
        "Kokoro", text_field="prompt", default_voice="af_heart", max_text_length=5000,
        output_ext="wav"),
    "fal-ai/elevenlabs/tts/multilingual-v2": _tts(
        "ElevenLabs Multilingual v2", default_voice="Rachel", max_text_length=10000,
        output_ext="mp3"),
    "fal-ai/elevenlabs/tts/turbo-v2.5": _tts(
        "ElevenLabs Turbo v2.5", default_voice="Rachel", max_text_length=10000, output_ext="mp3"),
    "fal-ai/gemini-tts": _tts(
        "Google Gemini TTS", default_voice="Kore", max_text_length=32000, output_ext="mp3"),
    "fal-ai/xai/tts/v1": _tts(
        "xAI TTS v1", default_voice="eve", max_text_length=15000, output_ext="mp3"),
    # Maya is steered by a prose voice description and has no voice id. Its /stream path returns a
    # raw chunked audio/pcm body (probed: int16 LE mono, 24 kHz), which is what StreamingTTSProvider
    # wants — so this is the one entry carrying a ``stream`` block.
    "fal-ai/maya": _tts(
        "Maya", voice_field=None, speed_field=None,
        prompt_field="prompt", max_text_length=5000, output_ext="mp3",
        stream={"endpoint": "fal-ai/maya/stream",
                "args": {"output_format": "pcm", "sample_rate": "24 kHz"},
                "sample_rate": 24000}),
}

# ── STT ───────────────────────────────────────────────────────────────────────
FAL_STT_MODELS: Dict[str, Dict[str, Any]] = {
    # Default. Probed: transcript on ``text``; same WER as whisper v3 large at ~2x the speed.
    "fal-ai/wizper": _stt("Wizper (Whisper v3 large, fal-optimized)"),
    "fal-ai/whisper": _stt("Whisper v3 large", prompt_field="prompt"),
    "fal-ai/elevenlabs/speech-to-text": _stt(
        "ElevenLabs Scribe", language_field="language_code"),
    # Canary returns ``output``, not ``text`` — the reason transcript_field exists.
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
        logger.debug(
            "FAL %s endpoint %r is not in the built-in catalog and declares no "
            "%s.fal.models.%s block; assuming the common {text, voice, speed} shape.",
            kind.upper(), endpoint, kind, endpoint)
    if isinstance(override, dict):
        unknown = set(override) - valid_keys
        if unknown:
            logger.warning(
                "Ignoring unknown key(s) %s in %s.fal.models.%s — valid keys are: %s",
                ", ".join(sorted(unknown)), kind, endpoint, ", ".join(sorted(valid_keys)))
        entry.update({k: v for k, v in override.items() if k in valid_keys})
    if not entry.get("display"):
        entry["display"] = endpoint
    return entry


def resolve_tts_model(fal_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """``(endpoint, entry)`` for ``tts.fal.model``, defaulting to :data:`DEFAULT_FAL_TTS_MODEL`.

    A ``tts.fal.models.<endpoint>`` block overrides a curated entry field-by-field or declares an
    endpoint the catalog has never heard of; an endpoint that is neither gets the generic shape.
    """
    cfg = fal_config if isinstance(fal_config, dict) else {}
    endpoint = str(cfg.get("model") or DEFAULT_FAL_TTS_MODEL).strip() or DEFAULT_FAL_TTS_MODEL
    entry = _merge_entry(endpoint, FAL_TTS_MODELS.get(endpoint),
                         _declared_entries(cfg).get(endpoint), _GENERIC_TTS, _TTS_ENTRY_KEYS, "tts")
    return endpoint, entry


def resolve_stt_model(fal_config: Dict[str, Any], model: Optional[str] = None
                      ) -> Tuple[str, Dict[str, Any]]:
    """``(endpoint, entry)`` for STT. ``model`` (the dispatcher's resolved value) wins over config."""
    cfg = fal_config if isinstance(fal_config, dict) else {}
    endpoint = str(model or cfg.get("model") or DEFAULT_FAL_STT_MODEL).strip() or DEFAULT_FAL_STT_MODEL
    entry = _merge_entry(endpoint, FAL_STT_MODELS.get(endpoint),
                         _declared_entries(cfg).get(endpoint), _GENERIC_STT, _STT_ENTRY_KEYS, "stt")
    return endpoint, entry


def _assign(payload: Dict[str, Any], dotted: str, value: Any) -> None:
    """Write ``value`` at a dotted path, creating intermediate dicts (``voice_setting.voice_id``).

    A non-dict already sitting at an intermediate key is replaced: the catalog owns the structure,
    and silently merging into a user's scalar would send a payload neither side intended.
    """
    keys = [k for k in dotted.split(".") if k]
    if not keys:
        return
    node = payload
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def build_payload(entry: Dict[str, Any], values: Dict[str, Any],
                  extra_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a fal ``arguments`` dict from an entry's field map.

    ``values`` is keyed by logical name (``text``, ``voice``, ``speed``, ``prompt``, ``audio``,
    ``language``); a value of None, or a field the entry maps to None, is omitted so an endpoint
    never receives a parameter it would reject. Precedence is entry ``extra`` < mapped values <
    caller ``extra_args``, so a user's ``fal.extra_args`` can always override.
    """
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
    """The entry's ``stream`` block when it exposes a chunked-PCM endpoint, else None."""
    stream = entry.get("stream")
    return stream if isinstance(stream, dict) and stream.get("endpoint") else None
