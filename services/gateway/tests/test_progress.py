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


def _static(name):
    from pathlib import Path

    import app.main as m

    return (Path(m.__file__).parent / "static" / name).read_text()


def test_every_track_and_skill_links_somewhere():
    """The track cards had a hover lift that promised a click and delivered
    nothing, and the skill bars were inert. A mastery bar you cannot act on
    only tells the learner they are weak at something."""
    page = _static("progress.html")
    assert 'class="track" href=' in page or "a class=\"track\" href=" in page
    assert "a class=\"skill" in page
    assert "/?skill=" in page


def test_tutor_page_honours_the_skill_filter():
    """Those links are only useful if the tutor page reads the parameter."""
    page = _static("index.html")
    assert "activeFilter" in page
    assert "URLSearchParams" in page and "'skill'" in page
    assert "filter-bar" in page, "a filtered list must say so and offer a way out"


def test_mastery_rows_are_not_a_dead_end():
    page = _static("index.html")
    assert "a.mastery-row" in page, "rows render as links"
    assert "/?skill=${encodeURIComponent(skill)}" in page


def test_mastery_label_splits_name_from_value():
    """Both spans live inside .label, so the split has to be on .label. Styling
    .mastery-row as the two-column grid ran them together as "arrays65%"."""
    page = _static("index.html")
    block = page[page.index(".mastery-row .label {"):]
    block = block[: block.index("}")]
    assert "flex" in block and "space-between" in block
