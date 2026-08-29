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
