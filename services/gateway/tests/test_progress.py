"""Progress view — streaks, tracks, and the placement gate.

The streak is the part with real edge cases: it is the number a learner sees
first and the one they will notice is wrong.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import COURSE_TRACKS, PROBLEMS, _streak, app

client = TestClient(app)


def _days_ago(*offsets: int) -> list[str]:
    today = datetime.now(timezone.utc).date()
    return sorted((today - timedelta(days=o)).isoformat() for o in offsets)


def test_streak_empty_history():
    assert _streak([]) == (0, 0)


def test_streak_counts_consecutive_days():
    assert _streak(_days_ago(0, 1, 2, 3)) == (4, 4)


def test_streak_survives_an_empty_today():
    """A learner mid-morning has not broken a streak they may still extend.

    Showing 0 until they practise reads as punishment for the time of day.
    """
    current, longest = _streak(_days_ago(1, 2, 3))
    assert current == 3
    assert longest == 3


def test_streak_breaks_after_a_missed_day():
    # Last activity two days ago -> the chain is genuinely broken.
    assert _streak(_days_ago(2, 3, 4))[0] == 0


def test_streak_longest_outlives_a_break():
    """A broken streak must not erase the record of the best one."""
    current, longest = _streak(_days_ago(0, 5, 6, 7, 8, 9))
    assert current == 1
    assert longest == 5


def test_streak_ignores_duplicate_days():
    """Several attempts in one day is one day, not several."""
    today = datetime.now(timezone.utc).date().isoformat()
    assert _streak([today, today, today]) == (1, 1)


def test_courses_cover_every_skill_in_the_bank():
    """A skill nobody can find a course for is a dead end in the catalogue."""
    in_tracks = {s for t in COURSE_TRACKS for s in t["skills"]}
    in_bank = {s for p in PROBLEMS for s in p["skills"]}
    assert in_bank - in_tracks == set()


def test_courses_endpoint_reports_real_problem_counts():
    r = client.get("/v1/courses")
    assert r.status_code == 200
    tracks = r.json()
    assert len(tracks) == len(COURSE_TRACKS)
    for t in tracks:
        expected = len([p for p in PROBLEMS if set(p["skills"]) & set(t["skills"])])
        assert t["total"] == expected
    # An empty track would render as a course with nothing to do in it.
    assert all(t["total"] > 0 for t in tracks)


def test_progress_requires_a_learner():
    assert client.get("/v1/me/progress").status_code == 401


def test_placement_requires_a_learner():
    r = client.post("/v1/me/placement", json={"responses": {"arrays": "none"}})
    assert r.status_code == 401
