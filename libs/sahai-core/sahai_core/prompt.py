"""The tutor's serving system prompt — one definition, used by every caller.

There were two. `services/tutor` defined a detailed one and exposed it on
`/system_prompt`; `services/session` built a shorter one inline and that is the
one that actually reached the model. Nothing called the endpoint, so the
detailed version was dead code that read like documentation of behaviour the
system did not have — including the rule that matters most for a code-mixed
tutor, "match the student's language".

Kept in the shared library rather than in either service because both need it
and neither owns it. Training builds its own prompt in `sahai/`, deliberately:
that one is part of an experiment and changing it mid-run would confound the
results this serving prompt has no part in.
"""

from __future__ import annotations

RULES = (
    "You are a tutor. You help students think, NOT give answers.\n\n"
    "STRICT RULES:\n"
    "- NEVER write code. No code blocks. No function definitions. No pseudocode.\n"
    "- NEVER show the solution or any part of it.\n"
    "- Ask ONE question per turn to guide the student's thinking.\n"
    "- Keep responses to 2-3 sentences maximum.\n"
    "- Never end a turn with a colon promising something you do not then say.\n"
    "- Match the student's language. If they write Hinglish, reply in Hinglish.\n\n"
    "GOOD: 'What data structure lets you check if you have seen a character in O(1)?'\n"
    "BAD:  'Here is the solution: def func(): ...'"
)


def system_prompt(
    problem: str, learner_context: str = "", turn_guidance: str = ""
) -> str:
    """Rules, then the problem, then who is being taught, then what they said.

    `problem` must be the full statement. The session service used to pass
    `problem_title`, which the exporter truncates to 80 characters — so for 43
    of the bank's 89 problems the tutor was guiding on a sentence cut
    mid-word ("...the largest triangle that can be inscribed in th") and could
    not know the shape was a semicircle. A tutor that cannot see the question
    cannot ask a useful question about it.

    `learner_context` and `turn_guidance` are appended after the rules so they
    can never displace them; the never-give-the-answer rules are what the whole
    design rests on, and a context block ahead of them would push them away
    from the generation point.

    `turn_guidance` comes last, nearest the generation point, because it is the
    most perishable thing here: the rules hold for every turn and the profile
    holds for the session, but "they just pasted code" is true of exactly this
    reply and stops being true at the next one. It is also the narrower
    instruction, and a narrow instruction placed before a broad one tends to be
    read as an exception to it rather than as a refinement of it.
    """
    prompt = f"{RULES}\n\nProblem: {problem}"
    if learner_context:
        prompt += f"\n\n{learner_context}"
    if turn_guidance:
        prompt += f"\n\n{turn_guidance}"
    return prompt
