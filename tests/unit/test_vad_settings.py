"""VAD settings (VAD_METHOD, VAD_ONSET, VAD_OFFSET).

Whisper invents text such as "Thank you." in silence and background noise
that the VAD passes on as speech (Speakr #418). The settings are handed to
whisperx.load_model; with none of them set, the model must be loaded with
exactly the arguments it was loaded with before.
"""

import numpy as np
import pytest

from test_model_selection import FakeModel, fresh  # noqa: F401  (fixture)

VAD_ENV = ("VAD_METHOD", "VAD_ONSET", "VAD_OFFSET")


@pytest.fixture
def load_calls(monkeypatch):
    """Replace whisperx.load_model; returns the keyword arguments of each call."""
    import whisperx
    calls = []

    def _load_model(name, **kwargs):
        calls.append(kwargs)
        return FakeModel(name)

    monkeypatch.setattr(whisperx, "load_model", _load_model)
    return calls


@pytest.fixture
def pipeline(fresh, monkeypatch):  # noqa: F811
    def _load(**env):
        for key in VAD_ENV:
            monkeypatch.delenv(key, raising=False)
        return fresh(**env)
    return _load


def _transcribe(p):
    p.transcribe(np.zeros(16000, dtype=np.float32), model_name="tiny")


@pytest.mark.parametrize("env", [{}, {k: "" for k in VAD_ENV}, {k: "  " for k in VAD_ENV}])
def test_unset_settings_load_the_model_as_before(pipeline, load_calls, env):
    p = pipeline(**env)
    _transcribe(p)
    assert load_calls == [{"device": p.DEVICE, "compute_type": p.COMPUTE_TYPE, "download_root": p.CACHE_DIR}]


def test_settings_are_passed_to_whisperx(pipeline, load_calls):
    p = pipeline(VAD_METHOD="Silero", VAD_ONSET="0.65", VAD_OFFSET="0.4")
    _transcribe(p)
    assert load_calls[0]["vad_method"] == "silero"
    assert load_calls[0]["vad_options"] == {"vad_onset": 0.65, "vad_offset": 0.4}


def test_one_setting_leaves_the_others_at_the_whisperx_defaults(pipeline, load_calls):
    p = pipeline(VAD_ONSET="0.7")
    _transcribe(p)
    assert "vad_method" not in load_calls[0]
    assert load_calls[0]["vad_options"] == {"vad_onset": 0.7}


@pytest.mark.parametrize("env", [
    {"VAD_METHOD": "webrtc"}, {"VAD_ONSET": "high"}, {"VAD_ONSET": "1.2"}, {"VAD_ONSET": "0"},
    {"VAD_OFFSET": "-0.1"}, {"VAD_OFFSET": "abc"},
])
def test_invalid_values_are_ignored_with_a_warning(pipeline, load_calls, caplog, env):
    p = pipeline(**env)
    _transcribe(p)
    assert "vad_method" not in load_calls[0] and "vad_options" not in load_calls[0]
    assert any(next(iter(env)) in r.getMessage() for r in caplog.records)



def test_the_chunk_length_is_left_to_whisperx(pipeline, monkeypatch, load_calls):
    """Shorter chunks produced far more invented text in testing, so the
    service never sets one, even with VAD settings in place."""
    seen = {}

    def _spy(self, audio, **kwargs):
        seen.update(kwargs)
        return {"language": "en", "segments": []}

    monkeypatch.setattr(FakeModel, "transcribe", _spy)
    p = pipeline(VAD_METHOD="silero", VAD_ONSET="0.8", VAD_CHUNK_SIZE="10")
    _transcribe(p)
    assert "chunk_size" not in seen and "chunk_size" not in load_calls[0]["vad_options"]
