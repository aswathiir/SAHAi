"""Run the four post-training checks over dumped rollouts.

Usage:
    python scripts/inspect_rollouts.py kaggle_results/output/rollouts

Reports, per epoch:
  1. reward spread          — zero spread means that epoch produced no gradient
  2. truncated turns        — turns cut mid-sentence, which the next speaker completes
  3. code in tutor turns    — leakage the pedagogy judge should be penalizing
  4. role sanity            — whether the student is answering and the tutor asking

Exits non-zero if any check fails, so it can gate a run.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from dataclasses import dataclass, field

CODE_MARKERS = re.compile(r"```|^\s*(def |class |import |return )", re.MULTILINE)
ASSIGNMENT = re.compile(r"^\s*[\w.\[\]]+\s*(=[^=]|\+=|-=|\.\w+\()", re.MULTILINE)
# A closed fence ends a turn cleanly; a trailing colon or an open LaTeX/bracket
# delimiter does not. Getting this wrong inflates the truncation count.
COMPLETE_END = re.compile(r"(```|[.!?\"'])\s*$")
DANGLING_END = re.compile(r"(:|\\\[|\\\(|\{|\(|\[)\s*$")


@dataclass
class EpochFindings:
    epoch: int
    n_rollouts: int
    reward_min: float
    reward_max: float
    truncated: list[str] = field(default_factory=list)
    dangling: list[str] = field(default_factory=list)
    tutor_code: list[str] = field(default_factory=list)
    student_code: list[str] = field(default_factory=list)
    unpenalized_code: list[str] = field(default_factory=list)

    @property
    def spread(self) -> float:
        return self.reward_max - self.reward_min

    @property
    def ok(self) -> bool:
        return (
            self.spread > 1e-8
            and not self.truncated
            and not self.unpenalized_code
        )


def looks_truncated(text: str) -> bool:
    """Cut off mid-thought — ends mid-word or on an unclosed delimiter."""
    text = text.rstrip()
    if not text or COMPLETE_END.search(text) or text.endswith(":"):
        return False  # a trailing colon is reported separately, not as truncation
    if DANGLING_END.search(text):
        return True
    # A short unpunctuated line is usually a complete utterance ("I think I can
    # solve it now"); a long one that stops on a word is usually a cut-off.
    # This is a heuristic — the definitive signal is whether generation hit EOS,
    # which the rollout dump does not yet record.
    if len(text.split()) <= 12:
        return False
    return bool(re.search(r"[A-Za-z0-9_]$", text))


def dangling_colon(text: str) -> bool:
    """Ends promising something that never arrives ("Here's the code:").

    The next speaker completes the promise instead of replying to it, which is
    the mechanism behind tutor/student role inversion.
    """
    return text.rstrip().endswith(":")


def has_code(text: str) -> bool:
    return bool(CODE_MARKERS.search(text) or ASSIGNMENT.search(text))


def analyze_epoch(records: list[dict]) -> EpochFindings:
    rewards = [r["reward"] for r in records]
    f = EpochFindings(
        epoch=records[0]["epoch"],
        n_rollouts=len(records),
        reward_min=min(rewards),
        reward_max=max(rewards),
    )

    for rec in records:
        tag = f"rollout {rec['rollout']}"
        for i, turn in enumerate(rec["turns"]):
            content = turn["content"]
            where = f"{tag} turn {i} [{turn['role']}]"

            # Runs from 2026-07-29 onward record whether generation stopped on
            # EOS. Trust that over the text heuristic when it is available.
            if "complete" in turn:
                truncated = not turn["complete"]
            else:
                truncated = looks_truncated(content)
            if truncated:
                f.truncated.append(f"{where}: ...{content.rstrip()[-60:]!r}")

            if dangling_colon(content):
                f.dangling.append(f"{where}: ...{content.rstrip()[-60:]!r}")

            if turn["role"] == "tutor" and has_code(content):
                f.tutor_code.append(where)
                # Code in a tutor turn must cost pedagogy score. If it does not,
                # the judge is not seeing what the policy actually produced.
                if rec["r_ped"] > 0.75:
                    f.unpenalized_code.append(f"{where}: r_ped={rec['r_ped']}")

            if turn["role"] == "student" and has_code(content):
                f.student_code.append(where)

    return f


def report(f: EpochFindings) -> None:
    print(f"\n{'=' * 72}")
    print(f"EPOCH {f.epoch} — {f.n_rollouts} rollouts")
    print(f"{'=' * 72}")

    status = "OK" if f.spread > 1e-8 else "FAIL"
    print(f"[{status:4}] reward spread {f.spread:.4f} "
          f"(min {f.reward_min:.3f}, max {f.reward_max:.3f})")
    if f.spread <= 1e-8:
        print("         all advantages are 0 — this epoch changed nothing")

    status = "OK" if not f.truncated else "FAIL"
    print(f"[{status:4}] truncated turns: {len(f.truncated)}")
    for line in f.truncated[:5]:
        print(f"         {line}")
    if len(f.truncated) > 5:
        print(f"         ... and {len(f.truncated) - 5} more")

    print(f"[{'OK  ' if not f.tutor_code else 'WARN'}] "
          f"tutor turns containing code: {len(f.tutor_code)}")

    print(f"[{'OK  ' if not f.dangling else 'WARN'}] "
          f"turns ending on a dangling colon: {len(f.dangling)}"
          f"  (invites the next speaker to complete instead of reply)")

    status = "OK" if not f.unpenalized_code else "FAIL"
    print(f"[{status:4}] tutor code that went unpenalized: {len(f.unpenalized_code)}")
    for line in f.unpenalized_code[:5]:
        print(f"         {line}")

    # Role inversion shows up as the student writing code while the tutor does.
    if f.student_code and f.tutor_code:
        print(f"[WARN] student wrote code in {len(f.student_code)} turns while the "
              f"tutor did in {len(f.tutor_code)} — check for role inversion")


def main(directory: str) -> int:
    paths = sorted(glob.glob(f"{directory}/epoch-*.json"))
    if not paths:
        print(f"No epoch-*.json under {directory}", file=sys.stderr)
        return 2

    findings = []
    for path in paths:
        with open(path) as fh:
            records = json.load(fh)
        if not records:
            continue
        f = analyze_epoch(records)
        findings.append(f)
        report(f)

    print(f"\n{'=' * 72}")
    failed = [f.epoch for f in findings if not f.ok]
    if failed:
        print(f"VERDICT: epochs {failed} still have blocking problems")
        return 1
    print("VERDICT: all epochs clean — spread present, no truncation, "
          "no unpenalized tutor code")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "output/rollouts"))
