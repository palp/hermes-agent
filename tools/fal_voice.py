"""Shared FAL.ai plumbing for the speech providers: client, credential, submit/wait, transfer.

Sits between :mod:`tools.fal_common` (stateless SDK helpers) and the two handlers
(:mod:`tools.tts_tool_fal`, :mod:`tools.transcription_fal`) so the queue-submit, interrupt-aware
wait and bounded-download logic exists once instead of twice.

The ``fal_client`` reference is cached on this module's global rather than in ``fal_common`` for the
reason that module's docstring gives: caches live with their caller so
``monkeypatch.setattr(fal_voice, "fal_client", fake)`` actually takes effect.

Direct ``FAL_KEY`` only — the Nous managed route is not wired for speech (``_FEATURES["tts"]`` and
``["stt"]`` in :mod:`hermes_cli.nous_subscription` both point at the ``openai-audio`` gateway, and
a managed FAL voice route would need a Portal-side coverage category).
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
    """Raised when the user interrupts while a FAL speech job is in flight."""


def load_fal_client() -> Any:
    """Lazy-load and cache ``fal_client`` via :func:`tools.fal_common.import_fal_client`."""
    global fal_client
    with _fal_client_lock:
        if fal_client is None:
            from tools.fal_common import import_fal_client
            fal_client = import_fal_client()
        return fal_client


def resolve_fal_key() -> str:
    """``FAL_KEY`` via the shared secret ladder (config > scope/env > .env > credential pool).

    Goes through ``resolve_provider_secret`` rather than reading the env directly so a key stored
    with ``hermes auth add fal`` resolves too (#68003).
    """
    try:
        from tools.tool_backend_helpers import resolve_provider_secret
    except ImportError:  # pragma: no cover — helper is in-repo
        return str(os.getenv("FAL_KEY") or "").strip()
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
    """Interrupt-aware ``handler.get()``.

    The SDK's blocking get hides user interrupts for 30-60s, so the get runs on a daemon worker and
    the interrupt bit is polled between join slices; on interrupt the worker is abandoned and the
    remote job is left to finish. Mirrors ``image_generation_tool._wait_fal_result``.
    """
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
            raise FalVoiceInterrupted(
                "Speech request interrupted by user — abandoned the in-flight FAL job.")
        worker.join(timeout=poll_seconds)
    if error_box:
        raise error_box[0]
    return result_box[0] if result_box else None


def submit_and_wait(endpoint: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Submit ``arguments`` to ``endpoint`` on the FAL queue and return the completed result."""
    client = load_fal_client()
    handler = client.submit(endpoint, arguments=arguments,
                            headers={"x-idempotency-key": str(uuid.uuid4())})
    result = wait_for_result(handler)
    if not isinstance(result, dict):
        raise RuntimeError(f"FAL endpoint {endpoint!r} returned no result")
    return result


def upload_audio_file(path: str) -> str:
    """Upload a local audio file to FAL storage and return its URL.

    **Deliberately not** ``fal_client.encode_file``: that builds a data URI whose MIME comes from
    ``mimetypes.guess_type``, and FAL's data-URI handler enforces a far narrower allowlist than its
    endpoints' documented formats. Measured against ``fal-ai/wizper``: ``audio/mpeg``,
    ``audio/x-wav`` and ``audio/aac`` are accepted, while ``audio/mp4a-latm`` (what
    ``mimetypes`` returns for ``.m4a`` — the exact format the STT dispatcher hands us, see
    ``transcription_audio._STT_M4A_ENCODE_ARGS``) is rejected with HTTP 400 "Unsupported data URL",
    along with ``audio/wav``, ``audio/mp4``, ``audio/ogg``, ``audio/webm`` and ``audio/flac``.
    Uploading sidesteps the allowlist entirely, and a plain URL for ``audio_url`` is verified good.
    """
    client = load_fal_client()
    url = client.upload_file(path)
    if not url:
        raise RuntimeError(f"FAL upload of {path} returned no URL")
    return str(url)


def audio_url_from_result(result: Dict[str, Any], endpoint: str) -> str:
    """Pull the audio URL out of a TTS result (``{"audio": {"url": ...}}``, uniform across FAL)."""
    audio = result.get("audio")
    url = audio.get("url") if isinstance(audio, dict) else None
    if not url:
        raise RuntimeError(
            f"FAL endpoint {endpoint!r} returned no audio URL (keys: {sorted(result)})")
    return str(url)


def download_audio_to(url: str, output_path: str, *, timeout: float = 120.0) -> str:
    """Stream ``url`` into ``output_path``, capped at :data:`MAX_AUDIO_DOWNLOAD_BYTES`.

    A partial file is removed rather than left behind for a caller to mistake for good audio.
    """
    import requests

    total = 0
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(output_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_AUDIO_DOWNLOAD_BYTES:
                        raise RuntimeError(
                            f"FAL audio exceeded {MAX_AUDIO_DOWNLOAD_BYTES} bytes; aborted")
                    handle.write(chunk)
        if not total:
            raise RuntimeError("FAL audio download was empty")
    except BaseException:
        _unlink_quietly(output_path)
        raise
    return output_path


def _unlink_quietly(path: str) -> None:
    """Best-effort removal of a partial artifact."""
    try:
        os.unlink(path)
    except OSError:
        pass


def upstream_detail(exc: BaseException) -> Optional[str]:
    """A short, user-facing detail from a FAL HTTP error, or None.

    FAL returns structured validation errors (a bad voice id yields
    ``{"detail": [{"loc": [...], "msg": "Voice with id X does not exist"}]}``), which are far more
    actionable than the bare exception type.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 — diagnostics must not mask the provider error
        return None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        messages = [
            str(item.get("msg")) for item in detail
            if isinstance(item, dict) and item.get("msg")
        ]
        if messages:
            return "; ".join(messages[:3])
    return None
