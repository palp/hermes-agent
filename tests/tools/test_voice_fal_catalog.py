"""Tests for the FAL speech model catalog.

Behaviour contracts, not snapshots: these pin *how* a catalog entry turns into a FAL payload and
how config extends the table, never the specific endpoints shipped (that list is expected to grow).
"""

from __future__ import annotations

import pytest

from tools.voice_fal_catalog import (
    DEFAULT_FAL_STT_MODEL, DEFAULT_FAL_TTS_MODEL, FAL_STT_MODELS, FAL_TTS_MODELS, build_payload,
    resolve_stt_model, resolve_tts_model, stream_config,
)


class TestPayloadBuilding:
    def test_dotted_voice_field_nests(self):
        """MiniMax needs voice/speed under ``voice_setting``; the dotted path is the mechanism."""
        _, entry = resolve_tts_model({"model": "fal-ai/minimax/speech-02-hd"})
        payload = build_payload(entry, {"text": "hi", "voice": "Wise_Woman", "speed": 1.5})
        assert payload == {"text": "hi", "voice_setting": {"voice_id": "Wise_Woman", "speed": 1.5}}

    def test_flat_fields_stay_flat(self):
        _, entry = resolve_tts_model({"model": "fal-ai/elevenlabs/tts/multilingual-v2"})
        payload = build_payload(entry, {"text": "hi", "voice": "Rachel", "speed": None})
        assert payload == {"text": "hi", "voice": "Rachel"}

    def test_kokoro_text_field_is_prompt(self):
        """Kokoro reads the text from ``prompt`` — the asymmetry the catalog exists for."""
        _, entry = resolve_tts_model({"model": "fal-ai/kokoro"})
        assert build_payload(entry, {"text": "hi"})["prompt"] == "hi"
        assert "text" not in build_payload(entry, {"text": "hi"})

    def test_none_values_and_unmapped_fields_are_omitted(self):
        """An endpoint must never receive a parameter it would reject."""
        _, entry = resolve_tts_model({"model": "fal-ai/maya"})
        payload = build_payload(entry, {"text": "hi", "voice": "ignored", "speed": 2.0,
                                        "prompt": "a calm voice"})
        assert payload == {"text": "hi", "prompt": "a calm voice"}  # maya has no voice/speed input

    def test_extra_args_win_over_mapped_values(self):
        _, entry = resolve_tts_model({"model": "fal-ai/kokoro"})
        payload = build_payload(entry, {"text": "hi", "voice": "af_heart"},
                                {"voice": "am_adam", "new_knob": 7})
        assert payload["voice"] == "am_adam"
        assert payload["new_knob"] == 7

    def test_extra_args_can_write_nested_paths(self):
        _, entry = resolve_tts_model({"model": "fal-ai/minimax/speech-02-hd"})
        payload = build_payload(entry, {"text": "hi"}, {"audio_setting.format": "pcm"})
        assert payload["audio_setting"] == {"format": "pcm"}

    def test_non_dict_at_intermediate_path_is_replaced(self):
        """A scalar sitting where a dict must go is overwritten, not merged into."""
        _, entry = resolve_tts_model({"model": "fal-ai/minimax/speech-02-hd"})
        payload = build_payload(entry, {"text": "hi", "voice": "v"},
                                {"voice_setting": "oops", "voice_setting.speed": 1.2})
        assert payload["voice_setting"] == {"speed": 1.2}


class TestResolution:
    def test_defaults_when_unconfigured(self):
        assert resolve_tts_model({})[0] == DEFAULT_FAL_TTS_MODEL
        assert resolve_stt_model({})[0] == DEFAULT_FAL_STT_MODEL

    def test_explicit_stt_model_argument_wins_over_config(self):
        """The dispatcher's resolved model beats ``stt.fal.model``."""
        endpoint, _ = resolve_stt_model({"model": "fal-ai/whisper"}, "fal-ai/wizper")
        assert endpoint == "fal-ai/wizper"

    def test_unknown_endpoint_gets_the_common_shape(self):
        """Reaching a brand-new FAL endpoint must not require a code change."""
        endpoint, entry = resolve_tts_model({"model": "fal-ai/invented-tomorrow"})
        assert endpoint == "fal-ai/invented-tomorrow"
        assert build_payload(entry, {"text": "hi", "voice": "x"}) == {"text": "hi", "voice": "x"}

    def test_config_declared_entry_reaches_an_unknown_endpoint(self):
        cfg = {"model": "fal-ai/new-tts",
               "models": {"fal-ai/new-tts": {"text_field": "input", "voice_field": "speaker.id"}}}
        _, entry = resolve_tts_model(cfg)
        assert build_payload(entry, {"text": "hi", "voice": "v7"}) == {
            "input": "hi", "speaker": {"id": "v7"}}

    def test_config_override_patches_a_curated_entry_field_by_field(self):
        cfg = {"model": "fal-ai/kokoro", "models": {"fal-ai/kokoro": {"default_voice": "am_adam"}}}
        _, entry = resolve_tts_model(cfg)
        assert entry["default_voice"] == "am_adam"
        assert entry["text_field"] == "prompt"  # untouched curated field survives

    def test_unknown_override_key_is_ignored_with_a_warning(self, caplog):
        cfg = {"model": "fal-ai/kokoro", "models": {"fal-ai/kokoro": {"txt_field": "typo"}}}
        with caplog.at_level("WARNING"):
            _, entry = resolve_tts_model(cfg)
        assert "txt_field" not in entry
        assert entry["text_field"] == "prompt"
        assert any("txt_field" in r.getMessage() for r in caplog.records)

    def test_malformed_config_does_not_raise(self):
        """A ``tts.fal: null`` / wrong-typed block falls back rather than exploding mid-reply."""
        assert resolve_tts_model(None)[0] == DEFAULT_FAL_TTS_MODEL
        assert resolve_tts_model({"models": "not-a-dict"})[0] == DEFAULT_FAL_TTS_MODEL
        assert resolve_tts_model({"model": "   "})[0] == DEFAULT_FAL_TTS_MODEL


class TestStreamConfig:
    def test_default_tts_model_is_not_streamable(self):
        """The invariant: the default endpoint has no chunked API, so streaming stays off and the
        sync path keeps the user's configured voice instead of swapping it."""
        assert stream_config(resolve_tts_model({})[1]) is None

    def test_a_stream_capable_entry_declares_endpoint_and_rate(self):
        stream = stream_config(resolve_tts_model({"model": "fal-ai/maya"})[1])
        assert stream and stream["endpoint"].endswith("/stream")
        # StreamingTTSProvider requires int16 mono PCM at this rate.
        assert stream["sample_rate"] == 24000
        assert stream["args"]["output_format"] == "pcm"


class TestCatalogShape:
    @pytest.mark.parametrize("endpoint", sorted(FAL_TTS_MODELS))
    def test_every_tts_entry_can_build_a_payload(self, endpoint):
        """No curated entry may be missing a field the builder needs."""
        _, entry = resolve_tts_model({"model": endpoint})
        values = {"text": "hello", "voice": entry.get("default_voice") or "v", "speed": 1.25,
                  "prompt": "a calm voice"}
        payload = build_payload(entry, values)
        assert payload, endpoint
        # The text always lands somewhere, whatever the endpoint calls it.
        assert "hello" in str(payload), endpoint

    @pytest.mark.parametrize("endpoint", sorted(FAL_STT_MODELS))
    def test_every_stt_entry_maps_audio_and_a_transcript_field(self, endpoint):
        _, entry = resolve_stt_model({}, endpoint)
        payload = build_payload(entry, {"audio": "https://x/a.mp3", "language": "en"})
        assert payload["audio_url"] == "https://x/a.mp3"
        assert entry["transcript_field"]

    def test_defaults_are_themselves_curated(self):
        assert DEFAULT_FAL_TTS_MODEL in FAL_TTS_MODELS
        assert DEFAULT_FAL_STT_MODEL in FAL_STT_MODELS
