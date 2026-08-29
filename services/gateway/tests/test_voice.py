"""Voice websocket bridge. No live ASR/session services in this test
environment, so downstream calls fail to connect — which is itself a real
test: the handler must turn that into a clean JSON error over the socket,
not crash the connection or hang.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_voice_socket_accepts_connection_and_reports_unreachable_backend():
    with client:  # runs lifespan, so app.state.http exists
        with client.websocket_connect("/v1/ws/voice/some-session?learner_id=me&lang=hi") as ws:
            ws.send_bytes(b"fake audio bytes")
            msg = ws.receive_json()
    assert msg["error"] == "upstream_unreachable"


def test_voice_socket_survives_client_disconnect():
    # Connecting and closing immediately must not raise inside the handler.
    with client:
        with client.websocket_connect("/v1/ws/voice/some-session") as ws:
            pass
