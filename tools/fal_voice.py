"""Shared FAL.ai plumbing for the speech providers: client, credential, submit/wait, transfer.

``fal_client`` is cached on this module's global rather than in ``fal_common`` so tests can
``monkeypatch.setattr(fal_voice, "fal_client", fake)``. Direct ``FAL_KEY`` only — the Nous
managed gateway is not wired for speech.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

fal_client: Any = None
_fal_client_lock = threading.Lock()

# Mirrors the sync TTS providers' bounded-upstream-body invariant.
MAX_AUDIO_DOWNLOAD_BYTES = 16 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 64 * 1024


class FalVoiceInterrupted(Exception):
    """The user interrupted while a FAL speech job was in flight."""


def load_fal_client() -> Any:
    """Lazy-load and cache ``fal_client`` via :func:`tools.fal_common.import_fal_client`."""
    global fal_client
    with _fal_client_lock:
        if fal_client is None:
            from tools.fal_common import import_fal_client
            fal_client = import_fal_client()
        return fal_client


def resolve_fal_key() -> str:
    """``FAL_KEY`` via the shared secret ladder, so ``hermes auth add fal`` pool entries resolve too."""
    from tools.tool_backend_helpers import resolve_provider_secret
    return resolve_provider_secret("FAL_KEY", "fal")


def require_fal_key() -> str:
    """:func:`resolve_fal_key`, raising ``ValueError`` with remediation when unset."""
    key = resolve_fal_key()
    if not key:
        raise ValueError(
            "FAL_KEY not set. Run `hermes tools` to configure FAL, or set the env var directly. "
            "Get a key at https://fal.ai/dashboard/keys")
    return key


def wait_for_result(handler, *, poll_seconds: float = 0.5) -> Any:
    """Interrupt-aware ``handler.get()``: the SDK blocks 30-60s hiding Ctrl-C, so the get runs on
    a daemon worker and the interrupt bit is polled between join slices (the remote job is left
    running). Mirrors ``image_generation_tool._wait_fal_result``."""
    from tools.interrupt import is_interrupted
    result_box: list = []
    error_box: list = []

    def _get() -> None:
        try:
            result_box.append(handler.get())
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller thread
            error_box.append(exc)

    worker = threading.Thread(target=_get, daemon=True, name="fal-voice-result-wait")
    worker.start()
    while worker.is_alive():
        if is_interrupted():
            raise FalVoiceInterrupted("Speech request interrupted by user — abandoned the in-flight FAL job.")
        worker.join(timeout=poll_seconds)
    if error_box:
        raise error_box[0]
    return result_box[0] if result_box else None


def submit_and_wait(endpoint: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Submit ``arguments`` to ``endpoint`` on the FAL queue and return the completed result."""
    handler = load_fal_client().submit(endpoint, arguments=arguments,
                                       headers={"x-idempotency-key": str(uuid.uuid4())})
    result = wait_for_result(handler)
    if not isinstance(result, dict):
        raise RuntimeError(f"FAL endpoint {endpoint!r} returned no result")
    return result


def upload_audio_file(path: str) -> str:
    """Upload ``path`` to FAL storage and return its URL.

    Not ``fal_client.encode_file``: FAL's data-URI handler rejects the MIME ``mimetypes`` assigns
    ``.m4a`` (``audio/mp4a-latm`` → HTTP 400), and ``.m4a`` is what the STT dispatcher hands us.
    Uploading sidesteps the allowlist; ``tests/tools/test_transcription_fal.py`` has the measurements.
    """
    url = load_fal_client().upload_file(path)
    if not url:
        raise RuntimeError(f"FAL upload of {path} returned no URL")
    return str(url)


def audio_url_from_result(result: Dict[str, Any], endpoint: str) -> str:
    """The audio URL from a TTS result (``{"audio": {"url": ...}}``, uniform across FAL)."""
    audio = result.get("audio")
    url = audio.get("url") if isinstance(audio, dict) else None
    if not url:
        raise RuntimeError(f"FAL endpoint {endpoint!r} returned no audio URL (keys: {sorted(result)})")
    return str(url)


def download_audio_to(url: str, output_path: str, *, timeout: float = 120.0) -> str:
    """Stream ``url`` into ``output_path``, capped at :data:`MAX_AUDIO_DOWNLOAD_BYTES`; a partial
    file is removed rather than left for a caller to mistake for good audio."""
    import requests

    total = 0
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(output_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    total += len(chunk)
                    if total > MAX_AUDIO_DOWNLOAD_BYTES:
                        raise RuntimeError(f"FAL audio exceeded {MAX_AUDIO_DOWNLOAD_BYTES} bytes; aborted")
                    handle.write(chunk)
        if not total:
            raise RuntimeError("FAL audio download was empty")
    except BaseException:
        try:
            os.unlink(output_path)
        except OSError:
            pass
        raise
    return output_path


def upstream_detail(exc: BaseException) -> Optional[str]:
    """A user-facing detail from a FAL HTTP error (its structured ``detail`` names the offending
    field, e.g. "Voice with id X does not exist"), or None."""
    response = getattr(exc, "response", None)
    try:
        payload = response.json() if response is not None else None
    except Exception:  # noqa: BLE001 — diagnostics must not mask the provider error
        return None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        messages = [str(i["msg"]) for i in detail if isinstance(i, dict) and i.get("msg")]
        return "; ".join(messages[:3]) or None
    return None
