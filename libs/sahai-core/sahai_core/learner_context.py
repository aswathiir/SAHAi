"""Turn BKT posteriors into an instruction the tutor can act on.

One definition, used by both halves. The gateway calls it to personalise a
served turn; the trainer calls it to build the same block into the rollouts
the policy is optimised on. They must agree: a policy trained on one wording
and served another is being asked at inference time to follow an instruction
it never saw during training, which is exactly the gap this module closes.

Deliberately *not* the raw numbers. "arrays: 0.42" gives a language model
nothing to do — it has no calibration for what 0.42 should change about a
hint. Naming the pedagogical move instead ("build the idea" vs "assume it")
is the part that can actually alter the next turn.
"""

from __future__ import annotations

# Below NEW: the learner has not demonstrated this, so the idea has to be
# built before it can be applied. At or above SOLID: re-explaining it wastes
# a turn they would rather spend on what is new. Everything in between is
# where a tutor should already be pitching, so saying so would spend tokens
# agreeing with itself — those skills are omitted on purpose.
MASTERY_NEW = 0.35
MASTERY_SOLID = 0.65

HEADING = "WHAT THIS LEARNER ALREADY KNOWS:"


def learner_context(skills: list[str], mastery: dict[str, float]) -> str:
    """The block for these skills, or "" when there is nothing worth saying.

    Only skills the current problem touches are mentioned. A learner's full
    profile is mostly irrelevant to any one problem and would crowd a system
    prompt whose own rule is 2-3 sentences per reply.
    """
    if not skills:
        return ""

    # The tutor says these to a learner, and `hash_maps` next to the prior-work
    # block's "a hash map" reads like two different things.
    def human(skill: str) -> str:
        return skill.replace("_", " ")

    new, solid = [], []
    for skill in sorted(set(skills)):
        # An unseen skill has no posterior yet; treat it as new rather than
        # inventing a middling one.
        value = mastery.get(skill, 0.0)
        if value < MASTERY_NEW:
            new.append(skill)
        elif value >= MASTERY_SOLID:
            solid.append(skill)

    lines = []
    if new:
        lines.append(
            f"- New to {', '.join(human(s) for s in new)}. "
            "Build the idea before asking them to apply it."
        )
    if solid:
        lines.append(
            f"- Solid on {', '.join(human(s) for s in solid)}. "
            "Do not re-explain it; push on what is new here."
        )
    if not lines:
        return ""
    return HEADING + "\n" + "\n".join(lines)
