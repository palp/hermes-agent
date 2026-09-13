"""FAL.ai text-to-speech backend for ``tools.tts_tool``.

Sibling of :mod:`tools.tts_tool_openai`: a per-vendor handler for a backend whose SDK and auth
model differ from the OpenAI-compatible ones. The unified inputs are translated into each
endpoint's native payload by :mod:`tools.voice_fal_catalog`, because FAL's speech schemas are
inconsistent; the queue submit / bounded download live in :mod:`tools.fal_voice`.

Seams (``_load_tts_config``, ``_resolve_provider_key``) are resolved through
:func:`tools.tts_tool_delivery._origin` at call time so test monkeypatches on ``tools.tts_tool``
apply.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from tools.tts_tool_delivery import _origin, _section
from tools.voice_fal_catalog import build_payload, resolve_tts_model

logger = logging.getLogger("tools.tts_tool")

# Extensions FAL speech endpoints actually return, for the container-mismatch warning below.
_CONTENT_TYPE_EXT = {
    "audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
    "audio/ogg": "ogg", "audio/opus": "opus", "audio/flac": "flac", "audio/aac": "aac",
}


def _resolve_speed(fal_config: Dict[str, Any], tts_config: Dict[str, Any]) -> Optional[float]:
    """``tts.fal.speed`` over the global ``tts.speed``; None when neither is set or it is 1.0.

    Omitting a neutral speed keeps the payload minimal, which matters because some endpoints
    validate the field's range and others don't accept it at all.
    """
    raw = fal_config.get("speed", tts_config.get("speed") if isinstance(tts_config, dict) else None)
    if raw is None:
        return None
    try:
        speed = float(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric TTS speed %r for FAL", raw)
        return None
    return None if speed == 1.0 else speed


def _warn_on_container_mismatch(result: Dict[str, Any], output_path: str, endpoint: str) -> None:
    """Log when the returned container differs from the caller's expected extension.

    The bytes are still written as-is (ffmpeg and every player sniff content), but a
    ``tts.fal.model`` that emits WAV while the pipeline asked for ``.mp3`` is worth surfacing
    rather than leaving as a silent mislabel.
    """
    audio = result.get("audio") if isinstance(result, dict) else None
    content_type = (audio or {}).get("content_type") if isinstance(audio, dict) else None
    actual = _CONTENT_TYPE_EXT.get(str(content_type or "").split(";")[0].strip().lower())
    expected = output_path.rsplit(".", 1)[-1].lower() if "." in output_path else ""
    if actual and expected and actual != expected:
        logger.warning(
            "FAL endpoint %s returned %s audio but the pipeline expected .%s; writing it as-is. "
            "Pick an endpoint whose container matches, or set tts.fal.extra_args if it can be "
            "configured.", endpoint, actual, expected)


def _generate_fal_tts(text: str, output_path: str, tts_config: Dict[str, Any]) -> str:
    """Synthesize ``text`` into ``output_path`` via a FAL speech endpoint; returns the path.

    ``ValueError`` for credential/config problems (reported to the model without a traceback);
    ``RuntimeError`` for upstream failures, carrying FAL's structured ``detail`` when it has one.
    """
    from tools import fal_voice

    fal_voice.require_fal_key()
    fal_config = _section(tts_config, "fal")
    endpoint, entry = resolve_tts_model(fal_config)

    voice = fal_config.get("voice") or entry.get("default_voice")
    prompt = fal_config.get("prompt") or fal_config.get("voice_description")
    if entry.get("prompt_field") and not prompt:
        raise ValueError(
            f"FAL endpoint {endpoint!r} is steered by a prose voice description rather than a "
            "voice id. Set tts.fal.prompt, e.g. \"a calm adult male voice, neutral accent\".")

    extra_args = fal_config.get("extra_args")
    arguments = build_payload(
        entry,
        {"text": text, "voice": voice, "speed": _resolve_speed(fal_config, tts_config),
         "prompt": prompt},
        extra_args if isinstance(extra_args, dict) else None)

    try:
        result = fal_voice.submit_and_wait(endpoint, arguments)
        url = fal_voice.audio_url_from_result(result, endpoint)
    except (ImportError, ValueError, fal_voice.FalVoiceInterrupted):
        raise
    except Exception as exc:  # noqa: BLE001 — normalized into a provider-prefixed RuntimeError
        detail = fal_voice.upstream_detail(exc)
        logger.error("FAL TTS failed on %s: %s", endpoint, exc, exc_info=True)
        raise RuntimeError(
            f"FAL TTS failed on {endpoint}: {detail or type(exc).__name__}") from exc

    _warn_on_container_mismatch(result, output_path, endpoint)
    fal_voice.download_audio_to(url, output_path)
    logger.info("Generated speech with FAL %s (%d chars -> %s)", endpoint, len(text), output_path)
    return output_path


def _fal_tts_available() -> bool:
    """Whether the FAL TTS route is usable — i.e. whether a ``FAL_KEY`` resolves.

    Deliberately does **not** require ``fal_client`` to be importable, unlike the ElevenLabs and
    Mistral checks. FAL's SDK is a lazy dep (``LAZY_DEPS["fal"]``) that installs on first use, and
    ``_LAZY_SDK_FEATURES`` pre-installs it when speech output is switched on — so gating on import
    would hide the whole ``text_to_speech`` tool on a fresh install that is otherwise ready.
    """
    origin = _origin()
    try:
        return bool(origin._resolve_provider_key("FAL_KEY", "fal"))
    except Exception:  # noqa: BLE001 — availability probes never raise
        return False
