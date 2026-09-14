"""Voice websocket bridge. No live ASR/session services in this test
environment, so downstream calls fail to connect — which is itself a real
test: the handler must turn that into a clean JSON error over the socket,
not crash the connection or hang.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_voice_socket_accepts_connection_and_reports_unreachable_backend(auth_headers):
    token = auth_headers["Authorization"].split()[1]
    with client:  # runs lifespan, so app.state.http exists
        with client.websocket_connect("/v1/ws/voice/some-session?lang=hi") as ws:
            ws.send_json({"token": token})
            assert ws.receive_json()["stage"] == "authenticated"
            ws.send_bytes(b"fake audio bytes")
            msg = ws.receive_json()
    assert msg["error"] == "upstream_unreachable"


def test_voice_socket_survives_client_disconnect(auth_headers):
    # Connecting and closing immediately must not raise inside the handler.
    token = auth_headers["Authorization"].split()[1]
    with client:
        with client.websocket_connect("/v1/ws/voice/some-session") as ws:
            ws.send_json({"token": token})
            ws.receive_json()


def test_voice_socket_refuses_an_unauthenticated_speaker():
    """`learner_id` was a query parameter defaulting to "me" and the gateway
    trusted it, so `?learner_id=<someone>` was enough to speak into another
    learner's session and have the reply written to their transcript. The
    ownership check in the session service could not help — it compares
    against the id the gateway forwards."""
    from starlette.websockets import WebSocketDisconnect

    with client:
        for hello in ({"token": "not-a-real-token"}, {}):
            with pytest.raises(WebSocketDisconnect) as excinfo:
                with client.websocket_connect("/v1/ws/voice/some-session") as ws:
                    ws.send_json(hello)
                    ws.receive_json()
            assert excinfo.value.code == 1008


def test_voice_socket_will_not_take_audio_before_the_token():
    """A binary first frame is a failed handshake, not an anonymous turn."""
    from starlette.websockets import WebSocketDisconnect

    with client:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/v1/ws/voice/some-session") as ws:
                ws.send_bytes(b"audio before auth")
                ws.receive_json()
        assert excinfo.value.code == 1008


def test_voice_socket_takes_no_identity_from_the_query_string():
    """A token in a query string lands in access logs, proxy logs and browser
    history. The handler must not accept one there."""
    import inspect

    from app.main import voice_turn

    assert "learner_id" not in inspect.signature(voice_turn).parameters
    assert "token" not in inspect.signature(voice_turn).parameters


def test_voice_turn_is_personalised_like_a_typed_turn():
    """A voice turn must carry the same learner context and full problem
    statement a typed turn does. It used to post {"content": transcript} only,
    so speaking to the tutor silently opted out of personalisation and handed
    it the 80-char truncated title — a second, worse tutor wearing the same
    face."""
    import inspect

    from app.main import voice_turn

    src = inspect.getsource(voice_turn)
    # Both paths call the one helper now. They built the block separately
    # before, which is how they drifted apart in the first place.
    assert "_context_for(" in src
    assert "problem_statement" in src
    assert "PROBLEMS_BY_ID" in src, "skills must be resolved server-side, not trusted"


def test_both_paths_build_the_same_context():
    import inspect

    from app.main import add_turn, voice_turn

    for fn in (add_turn, voice_turn):
        src = inspect.getsource(fn)
        assert "_context_for(" in src, f"{fn.__name__} builds its own context"
        assert "_learner_context(" not in src, (
            f"{fn.__name__} bypasses the shared helper — that is how the two "
            "paths diverged before"
        )


def test_voice_reports_each_stage_before_the_pipeline_finishes():
    """ASR takes seconds but generation can take minutes on CPU. Without stage
    events the learner who just spoke has no idea they were heard."""
    import inspect

    from app.main import voice_turn

    src = inspect.getsource(voice_turn)
    for stage in ("transcribed", "thinking", "speaking"):
        assert f'"{stage}"' in src, f"missing stage event: {stage}"
    # The transcript must go out before the turn call, not with the final frame.
    # Anchored on the POST itself, not the string "/turns" — that also appears
    # in the docstring above, which made an earlier version of this test pass
    # for the wrong reason.
    post_to_session = src.index("SESSION_URL}/sessions/{session_id}/turns")
    assert src.index('"stage": "transcribed"') < post_to_session
