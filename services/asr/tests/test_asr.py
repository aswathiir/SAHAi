"""Stub-backend contract tests. The stub must never pretend to transcribe or
synthesize — the whole point of stubbing is that a caller can't mistake stub
output for the real thing.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_both_backends():
    with client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["asr_backend"] == "stub"
    assert body["tts_backend"] == "stub"


def test_transcribe_passthrough_ignores_backend():
    with client:
        r = client.post("/transcribe", data={"transcript": "hello tutor"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hello tutor"
    assert body["backend"] == "passthrough"


def test_transcribe_audio_is_501_on_stub():
    with client:
        r = client.post(
            "/transcribe",
            files={"audio": ("clip.wav", b"not real audio bytes", "audio/wav")},
        )
    assert r.status_code == 501


def test_transcribe_without_audio_or_transcript_is_400():
    with client:
        r = client.post("/transcribe")
    assert r.status_code == 400


def test_synthesize_is_501_on_stub():
    with client:
        r = client.post("/synthesize", json={"text": "hello", "language": "en"})
    assert r.status_code == 501


def test_audio_too_large_is_413():
    with client:
        big = b"x" * (25 * 1024 * 1024 + 1)
        r = client.post(
            "/transcribe", files={"audio": ("clip.wav", big, "audio/wav")}
        )
    assert r.status_code == 413


def test_audio_decode_does_not_use_torchaudio_load():
    """torchaudio.load dispatches to TorchCodec in 2.11, which is not installed,
    so every real-audio request failed with "TorchCodec is required for
    load_with_torchcodec". The voice path only ever worked through the
    `transcript` text field, which is why nothing caught it.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "app" / "backends.py"
    text = src.read_text()
    # The call, not the mention — the replacement's own comment explains what
    # it replaced, and an earlier version of this test matched that comment.
    assert "torchaudio.load(" not in text
    assert "_decode_pcm16" in text


def test_decode_runs_everything_through_ffmpeg():
    """MediaRecorder emits WebM/Opus on Chrome and MP4/AAC on Safari, and
    libsndfile reads neither. One decode path beats a format guess."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "app" / "backends.py"
    text = src.read_text()
    assert '"ffmpeg"' in text
    assert "s16le" in text and "16000" in text

    dockerfile = (pathlib.Path(__file__).resolve().parent.parent / "Dockerfile").read_text()
    assert "ffmpeg" in dockerfile, "the decoder must actually be in the image"


def test_synthesis_is_bounded_server_side_not_only_by_the_caller():
    """A client timeout does not cancel server-side work.

    The gateway abandoned a synthesis at 270s and this process kept generating
    for another ten minutes at 400% CPU, starving the next /transcribe until
    it timed out too — one abandoned request wedged the whole speech service.
    Capping the caller alone cannot fix that; generation has to stop itself.
    """
    import pathlib

    text = (pathlib.Path(__file__).resolve().parent.parent / "app" / "backends.py").read_text()
    assert "TTS_MAX_SECONDS" in text
    assert "max_time=TTS_MAX_SECONDS" in text
