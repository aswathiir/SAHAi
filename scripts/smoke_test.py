"""End-to-end smoke test against a running stack.

    make up && make smoke

Drives the public gateway only — no internal ports, no direct service access —
so a pass means the deployed topology actually works, not just the code.
"""

from __future__ import annotations

import sys

import httpx

GATEWAY = "http://localhost:8080"
# Registered fresh on every run rather than reusing a fixed id. The gateway
# mints the learner id now — a client cannot name one — and a smoke run that
# invented its own identity would be testing the hole this replaced.
LEARNER = None
HEADERS: dict[str, str] = {}

SOLUTION = """def first_repeated_char(s):
    seen = set()
    for c in s:
        if c in seen:
            return c
        seen.add(c)
    return None"""

TEST_CASES = [
    {"input": {"s": "abcabc"}, "expected": "a"},
    {"input": {"s": "abcb"}, "expected": "b"},
]


def main() -> int:
    ok = True

    with httpx.Client(base_url=GATEWAY, timeout=60.0) as c:
        # 1. aggregate health
        r = c.get("/health")
        deps = r.json().get("dependencies", {})
        print(f"health: {r.json().get('status')}")
        for name, state in deps.items():
            mark = "ok " if state == "ok" else "FAIL"
            print(f"  [{mark}] {name}: {state}")
            ok &= state == "ok"
        if not ok:
            print("\nstack is not healthy; aborting")
            return 1

        # 2. auth is enforced — no credential, and a claimed learner id
        global LEARNER, HEADERS
        if c.post("/v1/sessions", json={"problem_id": "p"}).status_code != 401:
            print("  [FAIL] a request with no bearer token should be rejected")
            ok = False
        else:
            print("  [ok ] unauthenticated request rejected")

        # The exact request that used to work.
        claimed = c.post(
            "/v1/sessions",
            headers={"x-learner-id": "someone-else"},
            json={"problem_id": "p"},
        )
        if claimed.status_code != 401:
            print("  [FAIL] a claimed x-learner-id is still accepted")
            ok = False
        else:
            print("  [ok ] claimed learner id rejected")

        r = c.post("/v1/auth/register", json={"display_name": "smoke test"})
        if r.status_code != 201:
            print(f"  [FAIL] register -> {r.status_code} {r.text[:160]}")
            return 1
        body = r.json()
        LEARNER = body["learner_id"]
        HEADERS = {"Authorization": f"Bearer {body['token']}"}
        print(f"  [ok ] registered {LEARNER}")

        # 3. start a session
        r = c.post("/v1/sessions", headers=HEADERS, json={
            "problem_id": "mbpp-11",
            "problem_title": "Find the first repeated character in a string",
        })
        r.raise_for_status()
        sid = r.json()["session_id"]
        print(f"\nsession: {sid}")

        # 4. converse
        for msg in ["I don't know where to start.", "Should I track what I've seen?"]:
            r = c.post(f"/v1/sessions/{sid}/turns", headers=HEADERS,
                       json={"content": msg})
            r.raise_for_status()
            print(f"  student: {msg}")
            print(f"  tutor  : {r.json()['tutor_reply']}")

        # 5. submit, which runs code in the sandbox and updates mastery
        r = c.post(f"/v1/sessions/{sid}/submit", headers=HEADERS, json={
            "code": SOLUTION,
            "function_name": "first_repeated_char",
            "test_cases": TEST_CASES,
            "skills": ["strings", "hash_maps"],
        })
        r.raise_for_status()
        body = r.json()
        print(f"\nsubmit: solved={body['solved']} pass_rate={body['pass_rate']} "
              f"pedagogy={body['pedagogy_score']}")
        ok &= body["solved"] is True

        # 6. the loop closed — mastery moved off the prior
        r = c.get("/v1/me/mastery", headers=HEADERS)
        skills = r.json()["skills"]
        print(f"mastery: {skills}")
        ok &= all(v > 0.3 for v in skills.values())

    print("\n" + ("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except httpx.ConnectError:
        print(f"cannot reach {GATEWAY} — is the stack up? (make up)")
        sys.exit(2)
