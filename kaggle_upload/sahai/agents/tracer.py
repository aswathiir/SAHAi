from __future__ import annotations

from sahai.settings import TracerSettings


class BKTTracer:
    def __init__(self, settings: TracerSettings | None = None):
        self.settings = settings or TracerSettings()
        self.skills: dict[str, float] = {}

    def _ensure_skill(self, skill: str) -> None:
        if skill not in self.skills:
            self.skills[skill] = self.settings.p_init

    def update(self, skill: str, correct: bool) -> None:
        self._ensure_skill(skill)
        p = self.skills[skill]
        p_guess = self.settings.p_guess
        p_slip = self.settings.p_slip
        p_learn = self.settings.p_learn

        if correct:
            p_correct = p * (1 - p_slip) + (1 - p) * p_guess
            posterior = (p * (1 - p_slip)) / p_correct if p_correct > 0 else p
        else:
            p_incorrect = p * p_slip + (1 - p) * (1 - p_guess)
            posterior = (p * p_slip) / p_incorrect if p_incorrect > 0 else p

        self.skills[skill] = posterior + (1 - posterior) * p_learn

    def update_batch(self, skills: list[str], correct: bool) -> None:
        for skill in skills:
            self.update(skill, correct)

    def get_mastery(self, skill: str) -> float:
        self._ensure_skill(skill)
        return self.skills[skill]

    def get_ability(self) -> float:
        if not self.skills:
            return self.settings.p_init
        return sum(self.skills.values()) / len(self.skills)

    # The nominal top of the difficulty scale. `_estimate_difficulty` emits
    # 1..5, and callers that know their own bank pass its real maximum.
    DIFFICULTY_MAX = 5

    def target_difficulty(self, max_difficulty: int = DIFFICULTY_MAX) -> float:
        """The difficulty this learner should be working at, on the 1..max scale.

        Ability is a probability; difficulty is an ordinal. Mapping one onto
        the other is what makes "just beyond reach" expressible at all — and
        the previous code never did it, which is why `in_zpd` accepted a
        `difficulty` argument and threw it away.
        """
        return 1.0 + self.get_ability() * (max_difficulty - 1)

    def in_zpd(
        self, difficulty: int, skills: list[str], max_difficulty: int = DIFFICULTY_MAX
    ) -> bool:
        """Is this problem one step beyond what the learner can already do?

        Selection used to be `zpd_low <= mean(mastery) <= zpd_high`, which is
        not a difficulty band at all. Two consequences, both measured:

        * `p_init` is 0.3 and `zpd_low` is 0.3, so an *untouched* skill sat
          exactly on the boundary and passed. The band therefore meant
          "skills the learner has never attempted" — an exploration frontier
          that empties permanently once every skill has been seen.
        * BKT at this student's competence drives a failed skill to a fixed
          point of ~0.109 within three observations, far under `zpd_low`, and
          there is no forgetting transition to bring it back. So every skill
          that had been tried was *also* out of band.

        In the 10-epoch run of 2026-09-12 those two facts combined exactly as
        they must: the band held the whole bank at epoch 0 and was empty from
        epoch 4 onward, so seven of ten epochs trained on the top-up path
        rather than on a curriculum.

        Difficulty is now the primary axis, and it cannot empty: `target`
        stays inside [1, max] and problems exist at difficulty 1 in bulk.
        Mastery survives only as a ceiling — do not re-offer what the learner
        has demonstrably got. The old *floor* is gone deliberately: a skill at
        0.11 means the learner is failing it, which is precisely when the
        difficulty target should take over and hand them something easier,
        not when they should be excluded from practice altogether.
        """
        if skills:
            avg = sum(self.get_mastery(s) for s in skills) / len(skills)
            if avg > self.settings.zpd_high:
                return False
        return abs(difficulty - self.target_difficulty(max_difficulty)) <= 1.0

    def mastery_gap(self, skills: list[str]) -> float:
        """Distance from the middle of the mastery band — smaller is a better fit.

        Used only to order problems that are equally close on difficulty.
        Preferring the middle spreads practice across skills instead of
        grinding whichever one the learner has failed most, which is what
        ordering by raw mastery would do.
        """
        if not skills:
            return 0.0
        avg = sum(self.get_mastery(s) for s in skills) / len(skills)
        midpoint = (self.settings.zpd_low + self.settings.zpd_high) / 2
        return abs(avg - midpoint)

    def reset(self) -> None:
        self.skills.clear()
