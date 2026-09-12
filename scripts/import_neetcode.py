"""Seed a learner's skill priors from a NeetCode/LeetCode solutions repo.

Placement asks a learner what they think they know. This reads what they have
actually done. A repo of solved problems is far stronger evidence than a
self-report, and unlike scraping a site it needs no credentials: public repo,
documented API, stable layout.

Two signals, both taken from the file tree and git history only:

  paths  ->  which problems were solved, and therefore which skills
  dates  ->  how long ago, so old work counts for less

The solution *source* is deliberately not read. It is a much richer signal —
reaching for `heapq` evidences heaps in a way a filename cannot — but it is
also the answer to a problem, and this project's entire reward function exists
to stop answers reaching the learner. Reading code into a prompt is a decision
that deserves its own design pass, not a side effect of an importer.

Usage:
    python scripts/import_neetcode.py --repo owner/name --learner me --dry-run
    python scripts/import_neetcode.py --repo owner/name --learner me
"""

from __future__ import annotations

import argparse
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


def tag(slug: str) -> list[str]:
    """Skills for a problem slug, generic tagger plus NeetCode patterns."""
    skills = set()
    for skill, patterns in NEETCODE_PATTERNS.items():
        if any(p in slug for p in patterns):
            skills.add(skill)

    generic = _tag_skills(slug.replace("-", " "))
    # "general" is the tagger's "no idea" fallback, not a skill anyone has.
    skills.update(s for s in generic if s != "general")
    return sorted(skills)


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
            slug = parts[1]
            if slug not in seen and current_date:
                seen[slug] = (current_date, current_sha in bulk_shas)
    return seen


def weigh(solved: dict[str, tuple[date, bool]], today: date) -> dict[str, float]:
    """Recency-weighted problem count per skill."""
    totals: dict[str, float] = collections.defaultdict(float)
    for slug, (when, bulk) in solved.items():
        weeks = max((today - when).days, 0) / 7.0
        weight = 0.5 ** (weeks / HALF_LIFE_WEEKS)
        if bulk:
            # The true date is at or *before* the bulk commit, so its real age
            # is at least this. Halving keeps the estimate on the cautious
            # side rather than crediting unknown-age work in full.
            weight *= 0.5
        for skill in tag(slug):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="owner/name or a clone URL")
    ap.add_argument("--path", help="use an existing local clone instead of cloning")
    ap.add_argument("--learner", required=True)
    ap.add_argument("--gateway", default="http://localhost:8080")
    ap.add_argument("--dry-run", action="store_true", help="report, do not POST")
    args = ap.parse_args()

    if not args.repo and not args.path:
        ap.error("one of --repo or --path is required")

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(args.path) if args.path else clone(args.repo, Path(tmp) / "repo")
        solved = first_commits(repo_dir)
        submissions = collections.Counter(
            p.parent.name for p in repo_dir.rglob("submission-*.*")
        )

    today = datetime.now(timezone.utc).date()
    weights = weigh(solved, today)

    from sahai.core.dataset import SKILL_KEYWORDS
    all_skills = sorted(SKILL_KEYWORDS)
    levels = to_levels(weights, all_skills)

    bulk = sum(1 for _, b in solved.values() if b)
    print(f"problems solved      : {len(solved)}")
    print(f"  individually dated : {len(solved) - bulk}")
    print(f"  from a bulk import : {bulk}  (age unknown, weighted at half)")
    print(f"submissions on disk  : {sum(submissions.values())}")
    print(f"half-life            : {HALF_LIFE_WEEKS:.0f} weeks\n")

    print(f"{'skill':22}{'weighted':>10}{'raw':>6}  level")
    raw = collections.Counter(s for slug in solved for s in tag(slug))
    for skill in sorted(all_skills, key=lambda s: -weights.get(s, 0.0)):
        print(
            f"{skill:22}{weights.get(skill, 0.0):>10.2f}{raw.get(skill, 0):>6}"
            f"  {levels[skill]}"
        )

    untagged = [s for s in solved if not tag(s)]
    if untagged:
        print(f"\n{len(untagged)} slugs matched no skill and contribute nothing:")
        for slug in sorted(untagged)[:10]:
            print(f"  {slug}")

    if args.dry_run:
        print("\n--dry-run: nothing posted.")
        return

    body = json.dumps({"responses": levels}).encode()
    req = urllib.request.Request(
        f"{args.gateway}/v1/me/placement",
        data=body,
        headers={"content-type": "application/json", "x-learner-id": args.learner},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            skills = json.load(resp).get("skills", {})
        print(f"\nseeded {len(skills)} skills for {args.learner!r}")
        print("(skills with real observations were left untouched)")
    except urllib.error.HTTPError as exc:
        print(f"\nPOST failed: HTTP {exc.code} {exc.read()[:200].decode(errors='replace')}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
