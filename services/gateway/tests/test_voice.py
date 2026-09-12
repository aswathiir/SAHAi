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


def test_voice_turn_is_personalised_like_a_typed_turn():
    """A voice turn must carry the same learner context and full problem
    statement a typed turn does. It used to post {"content": transcript} only,
    so speaking to the tutor silently opted out of personalisation and handed
    it the 80-char truncated title — a second, worse tutor wearing the same
    face."""
    import inspect

    from app.main import voice_turn

    src = inspect.getsource(voice_turn)
    assert "_learner_context(" in src
    assert "problem_statement" in src
    assert "PROBLEMS_BY_ID" in src, "skills must be resolved server-side, not trusted"


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
