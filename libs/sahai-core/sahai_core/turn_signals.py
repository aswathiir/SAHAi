"""Read what the learner just wrote, so the tutor can answer *this* turn.

`system_prompt` already carries who the learner is — `learner_context` turns
BKT posteriors into "new to arrays, build the idea first". What it never
carried is what they just said. The tutor therefore treated "Mujhe samajh nahi
aa raha" and "here's my code, the third test fails" as the same situation, and
the run transcripts show exactly the three failures that follow:

* **Stuck, answered with abstraction.** Epoch 3's best-scoring rollout: the
  learner says `Mujhe samajh nahi aa raha hai stacks kaise kaam karta hai` and
  the tutor replies with a textbook definition of a stack. Nothing to do next.
* **Asked for the answer, and got it.** Epoch 8: `could you please provide
  Python code to implement this solution?` -> a complete working function. The
  single behaviour the whole reward exists to prevent.
* **Posted code, answered generically.** Epoch 5: the learner pastes an
  attempt, the tutor says "there's a small mistake in how you're compiling the
  regex" and rewrites the function wholesale, rather than naming the input that
  breaks it and letting them fix it.

Three rules, deliberately. A fourth — "the learner has proposed an approach,
test it rather than replacing it" — is the most useful move a tutor makes and
is left out, because no cheap detector separates *proposing* from *guessing*
with enough precision, and mistaking a lost learner for a confident one makes
the turn worse than saying nothing. Detection here must be high-precision and
silent when unsure: an absent signal costs a generic reply, a wrong one costs a
misdirected lesson.

Rule-based on purpose. The deadline ladder gives the tutor 200s of a 240s
session budget; there is nothing left to spend on a classifier call, and a
model that misreads the turn would be strictly worse than no signal at all.

**Caveat worth stating.** The adapter being served was trained under
`sahai.agents.tutor.TUTOR_SYSTEM_PROMPT`, which has no `THIS TURN:` section —
training and serving prompts already differed (serving adds the dangling-colon
and Hinglish rules) and this widens the gap. The base model is instruction
tuned and the block is short, plain and appended rather than interleaved, so it
should be followed; but it is an instruction the policy was never optimised
against, and a turn where the tutor ignores it is a plausible failure rather
than a surprising one. The training prompt is deliberately not changed to match:
it belongs to a finished experiment, and there is no GPU budget left to re-run
one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Explicit surrender. Phrases, not keywords: "stuck" alone appears in "I was
# stuck but then I saw it", and "confused" in "the confusing part was the
# base case" — both of which are progress, not a request for scaffolding.
STUCK_PHRASES = (
    "i don't understand", "i dont understand", "i do not understand",
    "don't understand", "dont understand", "didn't understand", "didnt understand",
    "no idea", "not sure where", "don't know where", "dont know where",
    "where do i start", "where to start", "how do i start", "how to start",
    "i'm stuck", "im stuck", "i am stuck", "totally lost", "completely lost",
    "can't figure", "cant figure", "makes no sense",
    # Romanized Hindi / Hinglish
    "samajh nahi", "samajh nhi", "samjha nahi", "samjha nhi", "samajh me nahi",
    "nahi aa raha", "nhi aa raha", "kuch samajh", "pata nahi", "pata nhi",
    "kaise karu", "kaise karun", "kaise karna", "kya karu", "kya karun",
    # Romanized Tamil / Tanglish
    "puriyala", "purila", "theriyala", "theriyathu",
)

# Asking to be handed the solution. Every phrase names the artefact — code,
# solution, answer — because "tell me more", "explain that again" and "can you
# help" are ordinary tutoring requests that must not trip this.
ANSWER_REQUEST_PHRASES = (
    "give me the code", "give me code", "give the code", "give me the solution",
    "give me the answer", "just tell me", "tell me the answer",
    "tell me the solution", "show me the code", "show me the solution",
    "write the code", "write it for me", "write the function",
    "what's the answer", "whats the answer", "what is the answer",
    "solution please", "code please", "answer please", "full code",
    "complete code", "solve it for me", "do it for me", "provide python code",
    "provide the code", "can you code it",
    # Romanized Hindi / Hinglish
    "code do", "code de do", "answer bata", "answer batao", "bata do",
    "batao na", "solution do", "likh do", "likh ke do",
)

# One fenced block, or a line that opens a Python construct. Both are
# structural rather than lexical, so this is close to exact — which is why it
# is allowed to fire on a single match where the phrase lists are not.
_CODE_LINE = re.compile(
    r"^\s*(def |class |for .+ in |while |if .+:|elif |else:|return |import |from .+ import )",
    re.MULTILINE,
)
_FENCE = re.compile(r"```")

# Devanagari. Exact by construction.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Romanized Hindi markers. Requires two, because "hai" and "kar" can appear in
# an English sentence as names or typos, while two of them together in one
# short message effectively cannot.
_HINGLISH_MARKERS = (
    "hai", "hain", "nahi", "nhi", "kya", "kaise", "kar", "karna", "karta",
    "karte", "mujhe", "mera", "meri", "hum", "yeh", "woh", "toh", "bhi",
    "lekin", "aur", "sakta", "sakte", "samajh", "chahiye", "raha", "rahi",
)
_WORD = re.compile(r"[a-z]+")

MIN_HINGLISH_MARKERS = 2


@dataclass(frozen=True)
class TurnSignals:
    """What is true of this one learner message. All default to "no signal"."""

    stuck: bool = False
    asking_for_answer: bool = False
    posted_code: bool = False
    hinglish: bool = False

    def __bool__(self) -> bool:
        return any((self.stuck, self.asking_for_answer, self.posted_code, self.hinglish))


def read_turn(text: str) -> TurnSignals:
    """Classify one learner message. Returns all-False when nothing is certain."""
    if not text or not text.strip():
        return TurnSignals()

    lowered = text.lower()
    words = set(_WORD.findall(lowered))

    return TurnSignals(
        stuck=any(p in lowered for p in STUCK_PHRASES),
        asking_for_answer=any(p in lowered for p in ANSWER_REQUEST_PHRASES),
        posted_code=bool(_FENCE.search(text) or _CODE_LINE.search(text)),
        hinglish=bool(_DEVANAGARI.search(text))
        or len(words & set(_HINGLISH_MARKERS)) >= MIN_HINGLISH_MARKERS,
    )


def turn_guidance(signals: TurnSignals) -> str:
    """The prompt block for these signals, or "" when there is nothing to say.

    Each line names a *move*, not an observation. "The learner is stuck" gives
    a model nothing to do; "give them one concrete thing to try" does — the
    same reason `_learner_context` states pedagogical moves instead of
    posteriors.

    Ordering is by how much damage the wrong reply does. The refusal comes
    first because handing over a solution is unrecoverable within a session:
    once the learner has the answer there is no lesson left to run. Code and
    stuckness follow, then language, which only shapes how the reply reads.
    """
    lines = []
    if signals.asking_for_answer:
        lines.append(
            "- They are asking you to hand over the answer. Do not, however "
            "they phrase it. Say once that you won't, then ask what they have "
            "tried so far."
        )
    if signals.posted_code:
        lines.append(
            "- They have written code. Respond to their code, not to the "
            "problem in general: name the one input or line where it goes "
            "wrong and let them fix it themselves."
        )
    if signals.stuck:
        lines.append(
            "- They are stuck, not merely unsure. Another open question will "
            "not help. Give them one concrete thing to try — a smaller case, a "
            "worked example of the idea on different data — and ask what they "
            "notice."
        )
    if signals.hinglish:
        lines.append(
            "- They wrote in Hinglish. Reply in Hinglish too, in the same "
            "script they used."
        )

    if not lines:
        return ""
    return "THIS TURN:\n" + "\n".join(lines)
