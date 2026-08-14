"""Problem catalogue served to the student assessment interface.

The one property that actually matters: the tutor's entire design rests on
never handing the student the answer, so the catalogue must not leak
`solution` even though the exporter has it available upstream.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import PROBLEMS, app

client = TestClient(app)


def test_problems_loaded_at_import():
    assert len(PROBLEMS) > 0


def test_list_problems_omits_heavy_fields():
    r = client.get("/v1/problems")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == len(PROBLEMS)
    for p in body:
        assert set(p.keys()) == {"id", "title", "difficulty", "skills"}


def test_get_problem_has_no_solution():
    problem_id = PROBLEMS[0]["id"]
    r = client.get(f"/v1/problems/{problem_id}")
    assert r.status_code == 200
    body = r.json()
    assert "solution" not in body
    assert body["id"] == problem_id
    assert body["test_cases"]
    assert body["function_name"]


def test_get_unknown_problem_404s():
    r = client.get("/v1/problems/does-not-exist")
    assert r.status_code == 404


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "SAHAi" in r.text
