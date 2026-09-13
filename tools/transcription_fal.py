"""FAL.ai speech-to-text backend for ``tools.transcription_tools``.

A dedicated sibling rather than another handler in :mod:`tools.transcription_cloud`: every handler
there is either OpenAI-SDK shaped or a Bearer multipart POST, and FAL is neither — it is a queue
API authenticated with ``Authorization: Key <FAL_KEY>`` that takes an ``audio_url`` rather than an
uploaded file part.

Per the module envelope contract this **must not raise**: every failure becomes
``_error_result(...)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from tools.transcription_common import _error_result, _log_prompt_unsupported, _ok_result
from tools.voice_fal_catalog import build_payload, resolve_stt_model

logger = logging.getLogger(__name__)


def _transcribe_fal(
    file_path: str, model_name: str, *, language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe ``file_path`` via a FAL speech-to-text endpoint.

    The local file is uploaded to FAL storage and passed by URL — see
    :func:`tools.fal_voice.upload_audio_file` for why a data URI is not used. The upload happens
    inline because the caller deletes the trimmed temp file as soon as this returns.
    """
    from tools import fal_voice
    from tools.transcription_tools import _load_stt_config, _resolve_stt_language

    try:
        fal_voice.require_fal_key()
    except ValueError as exc:
        return _error_result(str(exc))

    stt_config = _load_stt_config()
    fal_config = stt_config.get("fal") or {}
    endpoint, entry = resolve_stt_model(fal_config, model_name)

    # Hook override > stt.fal.language(_code) > stt.language.
    language_hint = language or _resolve_stt_language(
        "fal", stt_config, extra_keys=("language_code",)) or None
    # Only some FAL ASR endpoints take a vocabulary prompt (whisper does, wizper does not), so an
    # unsupported prompt is dropped and logged rather than sent and rejected.
    if prompt and not entry.get("prompt_field"):
        _log_prompt_unsupported(f"STT provider 'fal' endpoint {endpoint!r}")
        prompt = None

    extra_args = fal_config.get("extra_args")
    try:
        audio_url = fal_voice.upload_audio_file(file_path)
        arguments = build_payload(
            entry, {"audio": audio_url, "language": language_hint, "prompt": prompt},
            extra_args if isinstance(extra_args, dict) else None)
        result = fal_voice.submit_and_wait(endpoint, arguments)
    except PermissionError:
        return _error_result(f"Permission denied: {file_path}")
    except fal_voice.FalVoiceInterrupted as exc:
        return _error_result(str(exc))
    except Exception as exc:  # noqa: BLE001 — the envelope is the contract; never raise
        detail = fal_voice.upstream_detail(exc)
        logger.error("FAL STT failed on %s: %s", endpoint, exc, exc_info=True)
        return _error_result(f"FAL STT failed on {endpoint}: {detail or exc}")

    transcript = result.get(entry.get("transcript_field") or "text")
    if isinstance(transcript, dict):  # some endpoints nest {"text": ...}
        transcript = transcript.get("text")
    transcript = str(transcript or "").strip()
    if not transcript:
        return _error_result(f"FAL STT ({endpoint}) returned empty transcript", no_speech=True)

    logger.info("Transcribed %s via FAL (%s, lang=%s, %d chars)",
                Path(file_path).name, endpoint, language_hint or "auto", len(transcript))
    return _ok_result(transcript, "fal")
