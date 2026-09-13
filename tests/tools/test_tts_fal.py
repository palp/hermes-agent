"""Tests for the FAL.ai TTS provider.

Dispatch goes through ``tools.tts_tool`` (the facade) rather than the bare handler, so the
registration wiring — ``_BUILTIN_DISPATCH``, ``_BUILTIN_REQUIREMENTS``, the ``globals()`` generator
lookup — is exercised for real. A name that is in ``BUILTIN_TTS_PROVIDERS`` but missing from
``_BUILTIN_DISPATCH`` silently falls through to Edge, so that wiring is the thing worth pinning.
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


def _fake_client(captured: dict, *, url="https://fal.media/out.mp3", content_type="audio/mpeg"):
    """A stand-in ``fal_client`` whose submit records the call and returns a normal FAL result."""
    handle = MagicMock()
    handle.get = MagicMock(return_value={
        "audio": {"url": url, "content_type": content_type}, "duration_ms": 1000})

    def _submit(endpoint, arguments=None, headers=None):
        captured["endpoint"] = endpoint
        captured["arguments"] = arguments
        captured["headers"] = headers
        return handle

    client = MagicMock()
    client.submit = _submit
    return client


class TestRegistrationWiring:
    def test_fal_is_a_builtin_on_every_surface(self):
        """All four tables must agree, or dispatch degrades silently."""
        from agent import tts_registry
        from tools import tts_tool
        from tools.tts_tool_delivery import PROVIDER_MAX_TEXT_LENGTH

        assert "fal" in tts_tool.BUILTIN_TTS_PROVIDERS
        assert "fal" in tts_registry._BUILTIN_NAMES
        assert "fal" in tts_tool._BUILTIN_DISPATCH
        assert "fal" in tts_tool._BUILTIN_REQUIREMENTS
        assert "fal" in PROVIDER_MAX_TEXT_LENGTH

    def test_dispatch_entry_names_a_resolvable_generator(self):
        """``_synthesize_builtin`` resolves the generator by name out of the facade's globals."""
        from tools import tts_tool

        generator_name = tts_tool._BUILTIN_DISPATCH["fal"][2]
        assert callable(getattr(tts_tool, generator_name, None))

    def test_needs_ffmpeg_for_voice_bubbles(self):
        """FAL returns mp3/wav, so voice-bubble delivery must route through the ffmpeg leg;
        landing in neither set is the bug deepinfra has (no voice bubbles, no error)."""
        from tools import tts_tool

        assert "fal" in tts_tool._FFMPEG_OPUS_PROVIDERS
        assert "fal" not in tts_tool._NATIVE_OPUS_PROVIDERS

    def test_a_fal_block_is_not_read_as_a_command_provider(self):
        """Built-ins win: ``tts.providers.fal`` must never shadow the native handler."""
        from tools.tts_tool import _resolve_command_provider_config

        config = {"provider": "fal", "providers": {"fal": {"command": "say hello"}}}
        assert _resolve_command_provider_config("fal", config) is None


class TestRequirements:
    def test_requirements_follow_explicit_fal_provider(self, monkeypatch):
        from tools import tts_tool

        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "fal", "fal": {}})
        assert tts_tool.check_tts_requirements() is True

    def test_missing_key_makes_fal_unavailable(self, monkeypatch):
        from tools import tts_tool

        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "fal", "fal": {}})
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *a, **kw: "")
        assert tts_tool.check_tts_requirements() is False

    def test_unselected_fal_credentials_do_not_expose_the_edge_tool(self, monkeypatch):
        """A FAL_KEY set for image generation must not make an unusable Edge default look ready."""
        from tools import tts_tool

        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
        monkeypatch.setattr(tts_tool, "_import_edge_tts", MagicMock(side_effect=ImportError))
        monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: False)
        assert tts_tool.check_tts_requirements() is False


class TestSynthesis:
    def test_happy_path_submits_catalog_payload_and_writes_the_download(self, monkeypatch, tmp_path):
        from tools import fal_voice, tts_tool

        captured: dict = {}
        monkeypatch.setattr(fal_voice, "load_fal_client", lambda: _fake_client(captured))
        monkeypatch.setattr(fal_voice, "download_audio_to",
                            lambda url, path, **kw: (captured.update(download=url), path)[1])

        out = str(tmp_path / "speech.mp3")
        result = tts_tool._generate_fal_tts("Hello there", out, {"provider": "fal", "fal": {}})

        assert result == out
        assert captured["endpoint"] == "fal-ai/minimax/speech-02-hd"
        # The default endpoint nests its voice; that shape is the catalog's job.
        assert captured["arguments"]["text"] == "Hello there"
        assert captured["arguments"]["voice_setting"]["voice_id"] == "English_expressive_narrator"
        assert captured["download"] == "https://fal.media/out.mp3"
        # Idempotency key guards against a duplicate charge on transport retry.
        assert "x-idempotency-key" in captured["headers"]

    def test_missing_key_raises_valueerror_not_runtimeerror(self, monkeypatch, tmp_path):
        """ValueError is the config-error channel: reported to the model without a traceback."""
        from tools import fal_voice, tts_tool

        monkeypatch.setattr(fal_voice, "resolve_fal_key", lambda: "")
        with pytest.raises(ValueError, match="FAL_KEY not set"):
            tts_tool._generate_fal_tts("hi", str(tmp_path / "o.mp3"), {"fal": {}})

    def test_prompt_steered_endpoint_without_a_prompt_is_a_config_error(self, monkeypatch, tmp_path):
        """maya has no voice id; failing early beats a 422 from upstream."""
        from tools import tts_tool

        with pytest.raises(ValueError, match="voice description"):
            tts_tool._generate_fal_tts(
                "hi", str(tmp_path / "o.mp3"), {"fal": {"model": "fal-ai/maya"}})

    def test_upstream_validation_detail_is_surfaced(self, monkeypatch, tmp_path):
        """FAL's structured 422 names the offending field — far more useful than the exc type."""
        from tools import fal_voice, tts_tool

        response = MagicMock()
        response.json = MagicMock(return_value={
            "detail": [{"loc": ["body", "voice_setting", "voice_id"],
                        "msg": "Voice with id Nope does not exist"}]})
        failure = RuntimeError("422")
        failure.response = response

        client = MagicMock()
        client.submit = MagicMock(side_effect=failure)
        monkeypatch.setattr(fal_voice, "load_fal_client", lambda: client)

        with pytest.raises(RuntimeError, match="Voice with id Nope does not exist"):
            tts_tool._generate_fal_tts("hi", str(tmp_path / "o.mp3"), {"fal": {}})

    def test_speed_override_reaches_the_nested_field(self, monkeypatch, tmp_path):
        from tools import fal_voice, tts_tool

        captured: dict = {}
        monkeypatch.setattr(fal_voice, "load_fal_client", lambda: _fake_client(captured))
        monkeypatch.setattr(fal_voice, "download_audio_to", lambda url, path, **kw: path)

        tts_tool._generate_fal_tts("hi", str(tmp_path / "o.mp3"),
                                   {"fal": {"speed": 1.5}, "provider": "fal"})
        assert captured["arguments"]["voice_setting"]["speed"] == 1.5

    def test_neutral_speed_is_omitted(self, monkeypatch, tmp_path):
        """Sending speed=1.0 buys nothing and some endpoints validate the field's range."""
        from tools import fal_voice, tts_tool

        captured: dict = {}
        monkeypatch.setattr(fal_voice, "load_fal_client", lambda: _fake_client(captured))
        monkeypatch.setattr(fal_voice, "download_audio_to", lambda url, path, **kw: path)

        tts_tool._generate_fal_tts("hi", str(tmp_path / "o.mp3"), {"fal": {"speed": 1.0}})
        assert "speed" not in captured["arguments"].get("voice_setting", {})


class TestTextLengthCap:
    def test_cap_comes_from_the_resolved_endpoint(self):
        """FAL fronts endpoints whose caps differ by an order of magnitude, so the cap must be
        model-aware rather than one number for the whole provider."""
        from tools.tts_tool_delivery import _resolve_max_text_length

        assert _resolve_max_text_length("fal", {"fal": {"model": "fal-ai/gemini-tts"}}) == 32000
        assert _resolve_max_text_length(
            "fal", {"fal": {"model": "fal-ai/minimax/speech-02-hd"}}) == 5000

    def test_explicit_config_override_still_wins(self):
        from tools.tts_tool_delivery import _resolve_max_text_length

        assert _resolve_max_text_length(
            "fal", {"fal": {"model": "fal-ai/gemini-tts", "max_text_length": 111}}) == 111

    def test_malformed_catalog_override_falls_back_to_the_provider_floor(self):
        """A broken override must not disable truncation — this is what the ``fal`` entry in
        PROVIDER_MAX_TEXT_LENGTH exists for."""
        from tools.tts_tool_delivery import PROVIDER_MAX_TEXT_LENGTH, _resolve_max_text_length

        bad = {"fal": {"model": "fal-ai/gemini-tts",
                       "models": {"fal-ai/gemini-tts": {"max_text_length": 0}}}}
        assert _resolve_max_text_length("fal", bad) == PROVIDER_MAX_TEXT_LENGTH["fal"]
