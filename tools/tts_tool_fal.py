"""FAL.ai text-to-speech backend for ``tools.tts_tool`` (sibling of :mod:`tools.tts_tool_openai`).

Payload shapes come from :mod:`tools.voice_fal_catalog`; submit and download from
:mod:`tools.fal_voice`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from tools import fal_voice
from tools.tts_tool_delivery import _section
from tools.voice_fal_catalog import build_payload, resolve_tts_model

logger = logging.getLogger("tools.tts_tool")

# Containers FAL speech endpoints return, for the mismatch warning in _generate_fal_tts.
_CONTENT_TYPE_EXT = {"audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
                     "audio/ogg": "ogg", "audio/flac": "flac"}


def resolve_tts_inputs(
    fal_config: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Optional[str], Optional[str]]:
    """``(endpoint, entry, voice, prompt)`` for a ``tts.fal`` section; shared with the streamer.

    ``ValueError`` when the endpoint is steered by a prose voice description (maya) and
    ``tts.fal.prompt`` is unset — failing early beats a 422 from upstream.
    """
    endpoint, entry = resolve_tts_model(fal_config)
    voice = fal_config.get("voice") or entry.get("default_voice")
    prompt = fal_config.get("prompt")
    if entry.get("prompt_field") and not prompt:
        raise ValueError(
            f"FAL endpoint {endpoint!r} is steered by a prose voice description rather than a "
            'voice id. Set tts.fal.prompt, e.g. "a calm adult male voice, neutral accent".')
    return endpoint, entry, voice, prompt


def _resolve_speed(fal_config: Dict[str, Any], tts_config: Dict[str, Any]) -> Optional[float]:
    """``tts.fal.speed`` over ``tts.speed``; None when unset or 1.0 (a neutral speed is omitted
    because some endpoints range-check the field and others reject it outright)."""
    raw = fal_config.get("speed", tts_config.get("speed") if isinstance(tts_config, dict) else None)
    try:
        speed = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric TTS speed %r for FAL", raw)
        return None
    return None if speed == 1.0 else speed


def _warn_on_container_mismatch(result: Dict[str, Any], output_path: str, endpoint: str) -> None:
    """The bytes are written as-is (players and ffmpeg sniff content), but a WAV endpoint behind an
    ``.mp3`` path deserves a warning rather than a silent mislabel."""
    audio = result.get("audio")
    content_type = str((audio if isinstance(audio, dict) else {}).get("content_type") or "")
    actual = _CONTENT_TYPE_EXT.get(content_type.split(";")[0].strip().lower())
    expected = output_path.rsplit(".", 1)[-1].lower() if "." in output_path else ""
    if actual and expected and actual != expected:
        logger.warning("FAL endpoint %s returned %s audio but the pipeline expected .%s; writing it as-is.",
                       endpoint, actual, expected)


def _generate_fal_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Synthesize ``text`` into ``output_path``. ``ValueError`` = config/credential problem
    (reported without a traceback); ``RuntimeError`` = upstream failure, with FAL's ``detail``."""
    fal_voice.require_fal_key()
    fal_config = _section(tts_config, "fal")
    endpoint, entry, voice, prompt = resolve_tts_inputs(fal_config)
    extra_args = fal_config.get("extra_args")
    arguments = build_payload(
        entry,
        {"text": text, "voice": voice, "speed": _resolve_speed(fal_config, tts_config), "prompt": prompt},
        extra_args if isinstance(extra_args, dict) else None)
    try:
        result = fal_voice.submit_and_wait(endpoint, arguments)
        url = fal_voice.audio_url_from_result(result, endpoint)
    except (ImportError, ValueError, fal_voice.FalVoiceInterrupted):
        raise
    except Exception as exc:  # noqa: BLE001 — normalized into a provider-prefixed RuntimeError
        logger.error("FAL TTS failed on %s: %s", endpoint, exc, exc_info=True)
        raise RuntimeError(
            f"FAL TTS failed on {endpoint}: {fal_voice.upstream_detail(exc) or type(exc).__name__}") from exc
    _warn_on_container_mismatch(result, output_path, endpoint)
    fal_voice.download_audio_to(url, output_path)
    logger.info("Generated speech with FAL %s (%d chars -> %s)", endpoint, len(text), output_path)
    return output_path


def _fal_tts_available() -> bool:
    """A ``FAL_KEY`` resolves. Deliberately not gated on ``fal_client`` being importable: it is a
    lazy dep that installs on first use, so gating would hide ``text_to_speech`` on a ready install."""
    try:
        return bool(fal_voice.resolve_fal_key())
    except Exception:  # noqa: BLE001 — availability probes never raise
        return False
