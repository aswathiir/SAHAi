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

    def in_zpd(self, difficulty: int, skills: list[str]) -> bool:
        if not skills:
            return True
        avg = sum(self.get_mastery(s) for s in skills) / len(skills)
        return self.settings.zpd_low <= avg <= self.settings.zpd_high

    def reset(self) -> None:
        self.skills.clear()
