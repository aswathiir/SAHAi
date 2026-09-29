"""Seed a learner's skill priors from a NeetCode/LeetCode solutions repo.

Placement asks a learner what they think they know. This reads what they have
actually done. A repo of solved problems is far stronger evidence than a
self-report, and unlike scraping a site it needs no credentials: public repo,
documented API, stable layout.

Three signals:

  paths  ->  which problems were solved, and therefore which skills
  dates  ->  how long ago, so old work counts for less
  source ->  which technique was actually used

The source used to be deliberately unread, on the grounds that a solution is
the answer to a problem and this project's whole reward function exists to stop
answers reaching the learner. That reasoning was right about the risk and wrong
about where it lives. The risk is not in *reading* code; it is in code reaching
a prompt. So:

    **The source is parsed here and discarded here.** What leaves this script
    is a set of skill tags and one short phrase per problem — "a hash map",
    "binary search". Never a line of code, never a fragment of one.

That buys both things the old note was weighing against each other. Mastery
gets a far better signal than a filename can give: "kth-largest-element-in-an-
array" is a heap problem or a sorting problem depending on what the learner
actually wrote, and only the source knows which. And the tutor gets something
to teach *with* — "you used a hash map in Two Sum; what is similar here?" —
which is analogy, not an answer.

Two rules the tutor side keeps (see `_prior_work_for` in the gateway and
`PriorWork` in the tracer):

* the problem currently being worked on is excluded — everything here is
  already solved, so naming its technique ends the lesson;
* the prompt tells the tutor to *ask* the learner to make the connection, not
  to make it for them.

Usage:
    python scripts/import_neetcode.py --repo owner/name --learner me --dry-run
    python scripts/import_neetcode.py --repo owner/name --token $TOKEN
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sahai.core.dataset import _tag_skills  # noqa: E402

# NeetCode names problems after the story, not the technique — "house-robber"
# says nothing about dynamic programming — so the generic keyword tagger falls
# back to "general" for 121 of this repo's 319 slugs. These patterns recover
# the well-known ones.
#
# Kept here rather than in `sahai.core.dataset.SKILL_KEYWORDS` on purpose:
# that table also tags the MBPP training bank, and re-tagging training data in
# the middle of a reward experiment would change what the ZPD sampler offers
# for reasons unrelated to the experiment.
NEETCODE_PATTERNS: dict[str, tuple[str, ...]] = {
    "dynamic_programming": (
        "coin-change", "house-robber", "climbing-stairs", "decode-ways",
        "jump-game", "burst-balloons", "buy-and-sell", "partition-equal",
        "target-sum", "word-break", "edit-distance", "distinct-subsequences",
        "interleaving", "regular-expression", "tribonacci", "min-cost",
        "maximum-product-subarray", "palindromic-substrings", "unique-paths",
    ),
    "graphs": (
        "course-schedule", "island", "clone-graph", "pacific", "surrounded",
        "walls-and-gates", "network-delay", "cheapest-flight", "alien-dictionary",
        "redundant-connection", "word-ladder", "rotting", "swim-in",
    ),
    "stacks": (
        "asteroid-collision", "daily-temperatures", "baseball-game", "car-fleet",
        "reverse-polish", "basic-calculator", "largest-rectangle", "crawler-log",
        "min-stack", "generate-parentheses",
    ),
    "heaps": (
        "k-closest", "last-stone-weight", "meeting-rooms-iii", "kth-largest",
        "task-scheduler", "median-of", "top-k",
    ),
    "binary_search": (
        "eating-bananas", "capacity-to-ship", "guess-number", "koko",
        "find-minimum-in-rotated", "search-in-rotated", "split-array",
    ),
    "linked_lists": (
        "add-two-numbers", "lru-cache", "reverse-linked", "merge-two-sorted",
        "reorder-list", "copy-linked-list", "remove-node",
    ),
    "two_pointers": (
        "4sum", "3sum", "max-water-container", "trapping-rain",
        "valid-palindrome", "two-integer-sum",
    ),
    "arrays": (
        "find-pivot-index", "first-missing-positive", "max-consecutive-ones",
        "product-of-array", "rotate-matrix", "spiral-matrix", "set-matrix",
        "insert-new-interval", "meeting-schedule", "non-overlapping",
        "assign-cookies", "can-place-flowers", "height-checker",
        "maximum-product-difference", "find-missing-and-repeated",
        "minimum-number-of-operations", "grumpy-bookstore",
    ),
    "hash_maps": (
        "maximum-number-of-balloons", "design-twitter", "analyze-user",
        "concatenated-words", "fruit-into-baskets", "ransom-note",
    ),
    "recursion": ("combination", "subsets", "permutation", "n-queens", "sudoku"),
    "strings": ("length-of-last-word", "excel-sheet", "zigzag", "count-and-say"),
    "math": ("arranging-coins", "gas-station", "happy-number", "plus-one", "pow-x"),
    "sorting": ("hand-of-straights", "height-checker", "sort-colors", "largest-number"),
}

# Weeks, as a half-life on the weight a solved problem contributes. Knowledge
# does decay, and BKT has no forgetting transition at all, so something solved
# once in April and never revisited should not seed the same prior as last
# week's work. 26 weeks is chosen so a problem from six months ago still counts
# for half — deliberately gentle, because forgetting a concept you once
# understood is slower than forgetting a fact, and over-decaying would hand a
# capable learner problems they find trivial.
HALF_LIFE_WEEKS = 26.0

# Weighted problem count -> the categorical level /placement accepts. Reusing
# that endpoint rather than adding an evidence-specific one keeps the rule that
# a seeded prior can never overwrite a demonstrated observation.
#
# Absolute, not relative to the learner's own maximum. A relative scale would
# label somebody's strongest skill "confident" even if they had solved four
# problems in total, which is how a placement ends up flattering its own input.
# The cut points were first set at 6/2.5/0.5 and put 11 of 15 skills on
# "confident" for a 320-problem repo — nearly uniform, and a placement that
# says the same thing about everything discriminates nothing.
LEVEL_THRESHOLDS = ((12.0, "confident"), (4.0, "practiced"), (0.75, "seen"))


def tag(slug: str, techniques: set[str] | None = None) -> list[str]:
    """Skills for a problem, from its name and — when available — its source.

    The name is a guess and the source is evidence, so both contribute but the
    source is what makes the difference on problems the pattern table has never
    heard of. 121 of this repo's 319 slugs matched nothing at all before.
    """
    skills = set()
    for skill, patterns in NEETCODE_PATTERNS.items():
        if any(p in slug for p in patterns):
            skills.add(skill)

    generic = _tag_skills(slug.replace("-", " "))
    # "general" is the tagger's "no idea" fallback, not a skill anyone has.
    skills.update(s for s in generic if s != "general")

    for technique in techniques or ():
        skill = TECHNIQUE_SKILLS.get(technique)
        if skill:
            skills.add(skill)
    return sorted(skills)


# --------------------------------------------------------------- step 2
# Reading the solution source.
#
# Slug matching is a lookup table: `tag()` recognises the well-known NeetCode
# names and returns nothing for the rest, and a name never says *how* the
# learner solved it. "kth-largest-element-in-an-array" is a heap problem or a
# sorting problem depending on what they actually wrote, and only the source
# knows which.
#
# The source is parsed here and **discarded here**. What leaves this function
# is a set of skill tags and one short human phrase — never code, never a
# fragment of one. That boundary is the whole reason reading the source is
# safe: see the module docstring.

# The detector and its technique->skill map live in sahai-core so the gateway
# can run the identical logic on a repo connected from the browser. Importing
# rather than duplicating keeps one definition of "what this solution uses".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "libs" / "sahai-core"))
from sahai_core.techniques import (  # noqa: E402
    TECHNIQUE_ORDER,
    TECHNIQUE_SKILLS,
    headline_technique,
    techniques_used,
)



def read_solutions(repo_dir: Path) -> dict[str, set[str]]:
    """slug -> techniques, parsed from the Python submissions on disk.

    Non-Python submissions are skipped rather than guessed at. A repo that is
    mostly C++ simply falls back to slug tagging, which is what happened
    before this existed.
    """
    by_slug: dict[str, set[str]] = collections.defaultdict(set)
    for path in repo_dir.rglob("submission-*.py"):
        try:
            source = path.read_text(errors="replace")
        except OSError:
            continue
        by_slug[path.parent.name] |= techniques_used(source)
    return dict(by_slug)


def clone(repo: str, dest: Path) -> Path:
    url = repo if repo.startswith("http") else f"https://github.com/{repo}.git"
    subprocess.run(["git", "clone", "-q", url, str(dest)], check=True)
    return dest


def first_commits(repo_dir: Path) -> dict[str, tuple[date, bool]]:
    """slug -> (first commit date, came from a bulk import).

    The bulk flag matters. This repo opens with "Bulk sync: 266 submissions",
    so for well over half the problems the commit date records when the repo
    was created, not when the problem was solved. Treating that date as a
    learning date would systematically *overestimate* how recent — and so
    how well-retained — that work is.
    """
    log = subprocess.run(
        ["git", "log", "--reverse", "--format=%H|%ad|%s", "--date=short", "--name-only"],
        cwd=repo_dir, capture_output=True, text=True, check=True,
    ).stdout

    seen: dict[str, tuple[date, bool]] = {}
    bulk_shas: set[str] = set()
    current_sha = current_date = None

    for line in log.splitlines():
        header = re.match(r"^([0-9a-f]{7,40})\|(\d{4}-\d{2}-\d{2})\|(.*)$", line)
        if header:
            current_sha, raw_date, subject = header.groups()
            current_date = date.fromisoformat(raw_date)
            if re.search(r"bulk|initial", subject, re.I):
                bulk_shas.add(current_sha)
            continue
        parts = line.strip().split("/")
        if len(parts) >= 2 and parts[-1]:
            # The slug is the directory holding the submission, whatever the
            # depth: `<slug>/submission-1.py` and `python/<slug>/submission-1.py`
            # must both key the same way `read_solutions` does
            # (`path.parent.name`). Indexing `parts[1]` assumed a fixed
            # two-level layout and, on a flat repo, returned the *filename* --
            # so every problem collapsed onto one bogus slug named
            # `submission-1.py`, techniques never joined to solved problems,
            # and the import silently produced no skills at all.
            slug = parts[-2]
            if slug not in seen and current_date:
                seen[slug] = (current_date, current_sha in bulk_shas)
    return seen


def weigh(
    solved: dict[str, tuple[date, bool]],
    today: date,
    techniques: dict[str, set[str]] | None = None,
) -> dict[str, float]:
    """Recency-weighted problem count per skill."""
    techniques = techniques or {}
    totals: dict[str, float] = collections.defaultdict(float)
    for slug, (when, bulk) in solved.items():
        weeks = max((today - when).days, 0) / 7.0
        weight = 0.5 ** (weeks / HALF_LIFE_WEEKS)
        if bulk:
            # The true date is at or *before* the bulk commit, so its real age
            # is at least this. Halving keeps the estimate on the cautious
            # side rather than crediting unknown-age work in full.
            weight *= 0.5
        for skill in tag(slug, techniques.get(slug)):
            totals[skill] += weight
    return dict(totals)


def to_levels(weights: dict[str, float], all_skills: list[str]) -> dict[str, str]:
    levels = {}
    for skill in all_skills:
        score = weights.get(skill, 0.0)
        levels[skill] = next(
            (name for cut, name in LEVEL_THRESHOLDS if score >= cut), "none"
        )
    return levels


def title_for(slug: str) -> str:
    """"two-sum" -> "Two Sum". The tutor says this out loud, so it has to read
    like a problem name and not a URL fragment."""
    small = {"a", "an", "the", "of", "in", "to", "and", "with", "from", "for"}
    words = [w for w in slug.split("-") if w and not w.isdigit()]
    return " ".join(
        w.upper() if w in {"ii", "iii", "iv", "bst", "lru", "lfu"}
        else w if i and w in small
        else w.capitalize()
        for i, w in enumerate(words)
    )


def prior_work_items(
    solved: dict[str, tuple[date, bool]], techniques: dict[str, set[str]]
) -> list[dict]:
    """One record per solved problem that has a technique worth naming.

    A problem whose source could not be parsed, or whose technique is not one
    of the recognised phrases, is skipped rather than described vaguely: "you
    solved this somehow" gives a tutor nothing to build an analogy from.

    One record per *skill* the technique implies, because the tutor looks these
    up by the current problem's skills.
    """
    items = []
    for slug, (when, bulk) in sorted(solved.items()):
        technique = headline_technique(techniques.get(slug, set()))
        if not technique:
            continue
        skill = TECHNIQUE_SKILLS.get(technique)
        if not skill:
            continue
        items.append({
            "slug": slug,
            "title": title_for(slug),
            "skill": skill,
            "technique": technique,
            "solved_on": when.isoformat(),
            # Bulk commits date the repo's creation, not the solve. The tutor
            # says "before" instead of inventing a week.
            "dated": not bulk,
        })
    return items


def post(url: str, payload: dict, token: str) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="owner/name or a clone URL")
    ap.add_argument("--path", help="use an existing local clone instead of cloning")
    ap.add_argument("--learner", help="only used in the report; the token decides "
                                      "which learner is written to")
    ap.add_argument("--token", help="bearer token from POST /v1/auth/register")
    ap.add_argument("--gateway", default="http://localhost:8080")
    ap.add_argument("--dry-run", action="store_true", help="report, do not POST")
    args = ap.parse_args()

    if not args.repo and not args.path:
        ap.error("one of --repo or --path is required")
    if not args.dry_run and not args.token:
        ap.error("--token is required to write (or use --dry-run). The gateway "
                 "verifies identity now; an x-learner-id header is not accepted.")

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(args.path) if args.path else clone(args.repo, Path(tmp) / "repo")
        solved = first_commits(repo_dir)
        submissions = collections.Counter(
            p.parent.name for p in repo_dir.rglob("submission-*.*")
        )
        # Parsed inside the clone's lifetime and never held: what survives this
        # block is technique names, not source.
        techniques = read_solutions(repo_dir)

    today = datetime.now(timezone.utc).date()
    weights = weigh(solved, today, techniques)

    from sahai.core.dataset import SKILL_KEYWORDS
    all_skills = sorted(SKILL_KEYWORDS)
    levels = to_levels(weights, all_skills)

    bulk = sum(1 for _, b in solved.values() if b)
    print(f"problems solved      : {len(solved)}")
    print(f"  individually dated : {len(solved) - bulk}")
    print(f"  from a bulk import : {bulk}  (age unknown, weighted at half)")
    print(f"submissions on disk  : {sum(submissions.values())}")
    print(f"half-life            : {HALF_LIFE_WEEKS:.0f} weeks\n")

    parsed = sum(1 for t in techniques.values() if t)
    print(f"sources parsed       : {parsed} of {len(solved)} (Python only)")
    named = prior_work_items(solved, techniques)
    print(f"techniques named     : {len(named)}\n")

    print(f"{'skill':22}{'weighted':>10}{'raw':>6}  level")
    raw = collections.Counter(
        s for slug in solved for s in tag(slug, techniques.get(slug))
    )
    for skill in sorted(all_skills, key=lambda s: -weights.get(s, 0.0)):
        print(
            f"{skill:22}{weights.get(skill, 0.0):>10.2f}{raw.get(skill, 0):>6}"
            f"  {levels[skill]}"
        )

    if named:
        print("\nsample of what the tutor may reference (never any code):")
        for item in named[:6]:
            when = item["solved_on"] if item["dated"] else "undated"
            print(f"  {item['title']:<38} {item['technique']:<18} {when}")

    untagged = [s for s in solved if not tag(s, techniques.get(s))]
    if untagged:
        print(f"\n{len(untagged)} slugs matched no skill and contribute nothing:")
        for slug in sorted(untagged)[:10]:
            print(f"  {slug}")

    if args.dry_run:
        print("\n--dry-run: nothing posted.")
        return

    try:
        skills = post(
            f"{args.gateway}/v1/me/placement", {"responses": levels}, args.token
        ).get("skills", {})
        print(f"\nseeded {len(skills)} skills")
        print("(skills with real observations were left untouched)")

        stored = post(
            f"{args.gateway}/v1/me/prior_work", {"items": named}, args.token
        ).get("stored", 0)
        print(f"stored {stored} prior-work records the tutor can reference")
        print("(problem titles and technique names only — no source code)")
    except urllib.error.HTTPError as exc:
        print(f"\nPOST failed: HTTP {exc.code} {exc.read()[:200].decode(errors='replace')}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
