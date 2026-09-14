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


def test_next_up_never_recommends_the_fallback_tag():
    """`general` is what _tag_skills returns when it recognises nothing, so it
    is absent from SKILL_KEYWORDS and placement never asks about it. With no
    posterior it scores 0.0 and would win the "weakest skill" comparison for
    every learner forever — recommending the one tag nobody can be weak at."""
    from app.main import UNTRACKED_SKILL, _next_up

    rich = {"arrays": 0.9, "heaps": 0.2}
    picked = _next_up(rich, set())
    assert picked is not None
    assert picked["skill"] != UNTRACKED_SKILL


def test_next_up_picks_the_weakest_practisable_skill():
    """Every bank skill needs a posterior for this to mean anything: an unseen
    skill is 0.0 by design, so a partial mapping makes the *untouched* skills
    win rather than the weak one. An earlier version of this test passed only
    three and was surprised by binary_search."""
    from app.main import PROBLEMS, UNTRACKED_SKILL, _next_up

    bank_skills = {
        s for p in PROBLEMS for s in p["skills"] if s != UNTRACKED_SKILL
    }
    mastery = dict.fromkeys(bank_skills, 0.8)
    mastery["heaps"] = 0.1

    picked = _next_up(mastery, set())
    assert picked["skill"] == "heaps"
    assert picked["unsolved"] > 0
    assert picked["href"] == "/?skill=heaps"


def test_untouched_skill_outranks_a_weak_one():
    """No posterior means never attempted, which is a better place to start
    than something already practised badly — and it is why the test above has
    to supply the whole map."""
    from app.main import PROBLEMS, UNTRACKED_SKILL, _next_up

    bank_skills = {
        s for p in PROBLEMS for s in p["skills"] if s != UNTRACKED_SKILL
    }
    mastery = dict.fromkeys(bank_skills, 0.8)
    mastery["heaps"] = 0.1
    del mastery["trees"]          # never attempted

    assert _next_up(mastery, set())["skill"] == "trees"


def test_next_up_skips_a_skill_with_nothing_left_to_do():
    """Naming a weak skill whose problems are all solved would repeat forever."""
    from app.main import PROBLEMS, _next_up

    heaps_ids = {p["id"] for p in PROBLEMS if "heaps" in p["skills"]}
    picked = _next_up({"arrays": 0.9, "heaps": 0.1}, heaps_ids)
    assert picked["skill"] != "heaps"


def test_next_up_is_none_when_everything_is_solved():
    from app.main import PROBLEMS, _next_up

    assert _next_up({"arrays": 0.5}, {p["id"] for p in PROBLEMS}) is None


def test_diagnostic_spans_different_skills(auth_headers):
    """Four problems on arrays measures arrays four times and leaves the other
    fourteen skills exactly as unknown as before."""
    r = client.get("/v1/diagnostic", headers=auth_headers)
    assert r.status_code == 200
    problems = r.json()["problems"]
    assert len(problems) > 1
    # The skill each item was *picked for*, not its first tag: a problem tagged
    # ["arrays", "sorting", "heaps"] chosen for heaps still lists arrays first,
    # which is what an earlier version of this test wrongly measured.
    targets = [p["targets"] for p in problems]
    assert len(set(targets)) == len(targets), f"repeated target: {targets}"


def test_diagnostic_is_not_all_hardest_problems(auth_headers):
    """An item only discriminates near the learner's ability. Four difficulty-5
    problems produce four failures, which says "not expert" and nothing else."""
    problems = client.get("/v1/diagnostic", headers=auth_headers).json()["problems"]
    assert not all(p["difficulty"] >= 5 for p in problems)
    assert len({p["difficulty"] for p in problems}) > 1, "no difficulty spread"


def test_diagnostic_never_ships_the_answer(auth_headers):
    """Same rule as the problem catalogue: this is served straight to a browser."""
    body = client.get("/v1/diagnostic", headers=auth_headers).text
    assert "solution" not in body
    for p in client.get("/v1/diagnostic", headers=auth_headers).json()["problems"]:
        assert "test_cases" not in p, "tests are the grading criteria, not a hint"


def test_diagnostic_requires_a_token():
    assert client.get("/v1/diagnostic").status_code == 401
    r = client.post("/v1/diagnostic/grade", json={"problem_id": "x", "code": "pass"})
    assert r.status_code == 401


def test_claiming_a_learner_id_is_no_longer_enough():
    """The hole this replaces: `x-learner-id` was whatever the client typed,
    so naming someone else's id read their record."""
    for headers in ({"x-learner-id": "someone-else"}, {"Authorization": "Bearer nope"}):
        assert client.get("/v1/me/progress", headers=headers).status_code == 401
        assert client.get("/v1/diagnostic", headers=headers).status_code == 401


def test_diagnostic_grade_rejects_an_unknown_problem(auth_headers):
    r = client.post(
        "/v1/diagnostic/grade",
        json={"problem_id": "not_a_real_id", "code": "pass"},
        headers=auth_headers,
    )
    assert r.status_code == 404


def test_diagnostic_grades_against_the_bank_not_the_request():
    """Test cases are the grading criteria. Taking them from the client would
    let a caller mark their own work."""
    import inspect

    from app.main import diagnostic_grade

    src = inspect.getsource(diagnostic_grade)
    assert "PROBLEMS_BY_ID" in src
    assert 'problem["test_cases"]' in src


def test_skills_covered_uses_the_same_threshold_as_everything_else():
    """It was a hardcoded 0.7 while MASTERY_SOLID is 0.65, and placement caps
    confidence at exactly 0.65 — so a learner the tutor already treats as solid
    on a skill still read as "0 skills covered" on the same response. Two
    answers to the same question."""
    import inspect

    from app.main import MASTERY_SOLID, progress

    src = inspect.getsource(progress)
    assert ">= MASTERY_SOLID" in src
    assert ">= 0.7" not in src
    # And the threshold has to stay reachable by a placement.
    assert MASTERY_SOLID <= 0.65


def test_track_marks_a_mastery_that_is_only_an_estimate():
    """"0/31 solved, 60% mastery" reads as broken. `solved` is work done in
    this bank; `mastery` can come entirely from a placement survey or an
    imported repo. The card has to be able to say which."""
    import inspect

    from app.main import progress

    assert "estimate_only" in inspect.getsource(progress)
    page = _static("progress.html")
    assert "solved here" in page, "the two numbers must be labelled distinctly"
    assert "est. " in page


def test_activity_grid_days_can_be_opened():
    """The grid was the last inert element on the page — every square hovered a
    tooltip and nothing more. A tooltip is invisible on touch and gone as soon
    as the pointer moves."""
    page = _static("progress.html")
    assert "data-date=" in page
    assert "day-detail" in page
    assert 'role="button"' in page, "an activated day needs to be reachable by keyboard"


def test_resumable_list_requires_a_learner():
    assert client.get("/v1/me/sessions").status_code == 401


def test_resumable_list_fails_open():
    """An unavailable session service should cost the learner this list, not
    the whole record page."""
    import inspect

    from app.main import my_sessions

    src = inspect.getsource(my_sessions)
    assert "except Exception" in src
    assert "return []" in src


def test_tutor_page_can_reattach_to_a_session():
    """Sessions always survived a restart — they are rows in Postgres — but
    nothing surfaced one, so leaving mid-dialogue lost it in practice while the
    data sat there untouched."""
    page = _static("index.html")
    assert "resumeSession" in page
    assert "'session'" in page, "the page must read ?session= from the URL"
    # Resuming must reattach, not start a second conversation about the problem.
    assert "state.sessionId = sessionId" in page


def test_resume_warns_when_the_last_turn_went_unanswered():
    """A dialogue whose last turn is the student's is one whose tutor call
    died; sending a new message returns 409 by design. Saying so on resume
    beats the learner discovering it by being rejected."""
    page = _static("index.html")
    assert "never got a reply" in page
    progress = _static("progress.html")
    assert "last_role" in progress


def test_no_page_can_name_its_own_learner():
    """The free-text learner box is gone from all three pages.

    It used to set `x-learner-id`, so typing another learner's id read their
    sessions, mastery and submissions. Switching identity is signing out now,
    which reloads — the old handler had to tear down the conversation, the
    mastery panel and any in-flight request by hand, and a reload cannot miss
    a piece.
    """
    for page_name in ("index.html", "progress.html", "diagnostic.html"):
        page = _static(page_name)
        assert "x-learner-id" not in page, f"{page_name} still claims an identity"
        assert 'id="learner-id"' not in page, f"{page_name} still has the learner box"
        assert "authHeaders()" in page or "authToken()" in page, page_name
        assert "requireAuth(" in page, f"{page_name} renders before identity is settled"


def test_the_voice_socket_does_not_take_identity_from_the_url():
    """`learner_id` was a query parameter defaulting to "me", so any value
    opened someone else's session; and a token in a query string lands in
    access logs and browser history. It goes in the first frame instead."""
    page = _static("index.html")
    socket = page[page.index("function openVoiceSocket"):]
    socket = socket[: socket.index("function closeVoiceSocket")]
    assert "learner_id=" not in socket, "identity is back in the URL"
    assert "token=" not in socket, "token is back in the URL"
    assert "ws.send(JSON.stringify({ token" in socket


def test_gateway_forwards_the_learner_to_session():
    """Session checks that the caller owns the session; without the header
    forwarded, that check has nothing to compare against and every call 401s."""
    import inspect

    import app.main as m

    for name in ("add_turn", "submit", "get_session", "voice_turn"):
        src = inspect.getsource(getattr(m, name))
        assert "_trace(learner" in src, f"{name} calls session without identity"
        # Checking the call site is not enough: get_session forwarded
        # `_trace(learner)` while the handler only did `await _learner(...)`
        # without binding the result, so every read raised NameError at
        # runtime while this assertion passed.
        bound = "learner = await _learner(" in src or "await _learner(f\"Bearer" in src
        assert bound, f"{name} references `learner` without binding it"
