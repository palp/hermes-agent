"""FAL.ai speech-to-text backend for ``tools.transcription_tools``.

Its own sibling rather than a handler in :mod:`tools.transcription_cloud`: those are OpenAI-SDK
or Bearer-multipart shaped, and FAL is a queue API that takes an ``audio_url``. Per the envelope
contract this never raises.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from tools import fal_voice
from tools.transcription_common import (
    _error_result, _get_stt_section, _log_prompt_unsupported, _ok_result)
from tools.voice_fal_catalog import build_payload, resolve_stt_model

logger = logging.getLogger(__name__)


def _transcribe_fal(
    file_path: str, model_name: str, *, language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload ``file_path`` to FAL storage and transcribe it by URL (see
    :func:`tools.fal_voice.upload_audio_file` for why not a data URI)."""
    from tools.transcription_tools import _load_stt_config, _resolve_stt_language  # circular at import

    try:
        fal_voice.require_fal_key()
    except ValueError as exc:
        return _error_result(str(exc))
    stt_config = _load_stt_config()
    fal_config = _get_stt_section(stt_config, "fal")
    endpoint, entry = resolve_stt_model(fal_config, model_name)
    # Hook override > stt.fal.language(_code) > stt.language.
    language_hint = language or _resolve_stt_language("fal", stt_config, extra_keys=("language_code",)) or None
    if prompt and not entry.get("prompt_field"):  # wizper has no prompt input; whisper does
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
    except Exception as exc:  # noqa: BLE001 — the envelope is the contract
        logger.error("FAL STT failed on %s: %s", endpoint, exc, exc_info=True)
        return _error_result(f"FAL STT failed on {endpoint}: {fal_voice.upstream_detail(exc) or exc}")
    transcript = str(result.get(entry.get("transcript_field") or "text") or "").strip()
    if not transcript:
        return _error_result(f"FAL STT ({endpoint}) returned empty transcript", no_speech=True)
    logger.info("Transcribed %s via FAL (%s, lang=%s, %d chars)",
                Path(file_path).name, endpoint, language_hint or "auto", len(transcript))
    return _ok_result(transcript, "fal")
