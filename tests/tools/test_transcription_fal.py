"""Tests for the FAL.ai STT provider.

The load-bearing test here is ``test_uploads_rather_than_inlining_a_data_uri``: FAL's data-URI
handler enforces a narrower MIME allowlist than its endpoints' documented formats, and the format
the dispatcher hands us (``.m4a``) is on the wrong side of it. See that test for the measurements.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "test-fal-key")
    from tools import fal_voice
    monkeypatch.setattr(fal_voice, "fal_client", None)
    yield


@pytest.fixture
def audio(tmp_path):
    """A stand-in for what the dispatcher actually passes: a trimmed temp ``.m4a``."""
    path = tmp_path / "trimmed.m4a"
    path.write_bytes(b"\x00" * 64)
    return str(path)


def _fake_client(captured: dict, result=None):
    handle = MagicMock()
    handle.get = MagicMock(return_value=result if result is not None else {"text": "hello world"})

    def _submit(endpoint, arguments=None, headers=None):
        captured["endpoint"] = endpoint
        captured["arguments"] = arguments
        return handle

    def _upload(path):
        captured["uploaded"] = path
        return "https://fal.media/uploaded.m4a"

    client = MagicMock()
    client.submit = _submit
    client.upload_file = MagicMock(side_effect=_upload)
    # A regression to encode_file must fail loudly, not silently 400 against the live API.
    client.encode_file = MagicMock(side_effect=AssertionError("encode_file must not be used"))
    return client


class TestRegistrationWiring:
    def test_fal_is_a_builtin_cloud_stt_provider(self):
        from agent import transcription_registry
        from tools import transcription_tools
        from tools.transcription_common import BUILTIN_STT_PROVIDERS, CLOUD_STT_PROVIDERS

        assert "fal" in BUILTIN_STT_PROVIDERS
        # Cloud membership is what enrols fal in the 25 MB cap, CAF->WAV and the silence trim.
        assert "fal" in CLOUD_STT_PROVIDERS
        assert "fal" in transcription_registry._BUILTIN_NAMES
        assert callable(getattr(transcription_tools, "_transcribe_fal", None))

    def test_model_keys_entry_exists(self):
        """A missing ``_BUILTIN_MODEL_KEYS`` row is a KeyError at dispatch time, not import time."""
        from tools.transcription_tools import _BUILTIN_MODEL_KEYS, _builtin_model_name

        assert "fal" in _BUILTIN_MODEL_KEYS
        assert _builtin_model_name("fal", {}, None) == "fal-ai/wizper"
        assert _builtin_model_name("fal", {"fal": {"model": "fal-ai/whisper"}}, None) == \
            "fal-ai/whisper"

    def test_fal_is_last_in_the_autodetect_ladder(self):
        """FAL_KEY is commonly already set for image/video generation. If fal auto-selected ahead of
        another provider, configuring image gen would silently move a user's STT backend."""
        from tools.transcription_tools import _CLOUD_PROVIDER_SPECS

        assert list(_CLOUD_PROVIDER_SPECS)[-1] == "fal"

    def test_a_fal_block_is_not_read_as_a_command_provider(self):
        from tools.transcription_command import _NON_COMMAND_STT_NAMES

        assert "fal" in _NON_COMMAND_STT_NAMES


class TestProviderGating:
    def test_get_provider_gating_keys_on_fal_key(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        from tools.transcription_tools import _get_provider

        assert _get_provider({"provider": "fal"}) == "none"
        monkeypatch.setenv("FAL_KEY", "test-key")
        assert _get_provider({"provider": "fal"}) == "fal"


class TestTranscription:
    def test_uploads_rather_than_inlining_a_data_uri(self, monkeypatch, audio):
        """FAL's data-URI MIME allowlist rejects the format we actually have.

        Measured against ``fal-ai/wizper``: ``audio/mpeg``, ``audio/x-wav`` and ``audio/aac`` are
        accepted, but ``audio/mp4a-latm`` — what ``mimetypes.guess_type`` returns for ``.m4a``, and
        therefore what ``fal_client.encode_file`` would label the trimmed temp file produced by
        ``transcription_audio._STT_M4A_ENCODE_ARGS`` — is rejected with HTTP 400 "Unsupported data
        URL", along with ``audio/wav``, ``audio/mp4``, ``audio/ogg``, ``audio/webm`` and
        ``audio/flac``. Uploading sidesteps the allowlist. Regressing to ``encode_file`` would break
        every default-configured fal STT call, so the fake client fails loudly if it is used.
        """
        from tools import fal_voice, transcription_tools

        captured: dict = {}
        client = _fake_client(captured)
        monkeypatch.setattr(fal_voice, "load_fal_client", lambda: client)
        monkeypatch.setattr(transcription_tools, "_load_stt_config", lambda: {})

        result = transcription_tools._transcribe_fal(audio, "fal-ai/wizper")

        assert result["success"] is True
        client.upload_file.assert_called_once_with(audio)
        client.encode_file.assert_not_called()
        assert captured["arguments"]["audio_url"] == "https://fal.media/uploaded.m4a"
        assert not str(captured["arguments"]["audio_url"]).startswith("data:")

    def test_happy_path_envelope(self, monkeypatch, audio):
        from tools import fal_voice, transcription_tools

        captured: dict = {}
        monkeypatch.setattr(fal_voice, "load_fal_client", lambda: _fake_client(captured))
        monkeypatch.setattr(transcription_tools, "_load_stt_config", lambda: {})

        result = transcription_tools._transcribe_fal(audio, "fal-ai/wizper", language="en")

        assert result == {"success": True, "transcript": "hello world", "provider": "fal"}
        assert captured["endpoint"] == "fal-ai/wizper"
        assert captured["arguments"]["language"] == "en"

    def test_alternate_transcript_field_is_read(self, monkeypatch, audio):
        """``fal-ai/speech-to-text`` returns ``output`` rather than ``text``."""
        from tools import fal_voice, transcription_tools

        captured: dict = {}
        monkeypatch.setattr(fal_voice, "load_fal_client",
                            lambda: _fake_client(captured, result={"output": "canary text"}))
        monkeypatch.setattr(transcription_tools, "_load_stt_config", lambda: {})

        result = transcription_tools._transcribe_fal(audio, "fal-ai/speech-to-text")
        assert result["transcript"] == "canary text"

    def test_unsupported_prompt_is_dropped_not_sent(self, monkeypatch, audio):
        """wizper has no prompt input; whisper does. Sending one anyway would 422."""
        from tools import fal_voice, transcription_tools

        captured: dict = {}
        monkeypatch.setattr(fal_voice, "load_fal_client", lambda: _fake_client(captured))
        monkeypatch.setattr(transcription_tools, "_load_stt_config", lambda: {})

        transcription_tools._transcribe_fal(audio, "fal-ai/wizper", prompt="Hermes, fal.ai")
        assert "prompt" not in captured["arguments"]

        transcription_tools._transcribe_fal(audio, "fal-ai/whisper", prompt="Hermes, fal.ai")
        assert captured["arguments"]["prompt"] == "Hermes, fal.ai"

    def test_missing_key_returns_an_envelope_and_does_not_raise(self, monkeypatch, audio):
        """The module contract: every failure is an envelope, never an exception."""
        from tools import fal_voice, transcription_tools

        monkeypatch.setattr(fal_voice, "resolve_fal_key", lambda: "")
        result = transcription_tools._transcribe_fal(audio, "fal-ai/wizper")
        assert result["success"] is False
        assert "FAL_KEY" in result["error"]

    def test_upstream_failure_becomes_an_envelope(self, monkeypatch, audio):
        from tools import fal_voice, transcription_tools

        client = MagicMock()
        client.upload_file = MagicMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(fal_voice, "load_fal_client", lambda: client)
        monkeypatch.setattr(transcription_tools, "_load_stt_config", lambda: {})

        result = transcription_tools._transcribe_fal(audio, "fal-ai/wizper")
        assert result["success"] is False and result["transcript"] == ""
        assert "boom" in result["error"]

    def test_empty_transcript_is_flagged_no_speech(self, monkeypatch, audio):
        """Silence is non-fatal — the gateway treats no_speech differently from an error."""
        from tools import fal_voice, transcription_tools

        captured: dict = {}
        monkeypatch.setattr(fal_voice, "load_fal_client",
                            lambda: _fake_client(captured, result={"text": "   "}))
        monkeypatch.setattr(transcription_tools, "_load_stt_config", lambda: {})

        result = transcription_tools._transcribe_fal(audio, "fal-ai/wizper")
        assert result["success"] is False
        assert result.get("no_speech") is True
