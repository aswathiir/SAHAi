from __future__ import annotations

import re

# A digit before the period excludes numbered-list markers ("3. Count the ...")
# and step headings, which are not sentence boundaries.
SENTENCE_END = re.compile(r"(?<![0-9])[.!?](?=\s|$)|\n")


def terminator_ids(tokenizer, model=None) -> set[int]:
    """Every token id that legitimately ends a turn.

    Qwen chat models stop on <|im_end|>, which may live in eos_token_id (scalar
    or list) on either the tokenizer or the model's generation_config.
    """
    ids: set[int] = set()

    for source in (tokenizer, getattr(model, "generation_config", None)):
        eos = getattr(source, "eos_token_id", None)
        if eos is None:
            continue
        if isinstance(eos, int):
            ids.add(eos)
        else:
            ids.update(int(e) for e in eos)

    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end >= 0:
        ids.add(im_end)

    return ids


def hit_terminator(generated_ids, terminators: set[int]) -> bool:
    """True if generation stopped on its own rather than at max_new_tokens."""
    if not terminators:
        return True
    return any(int(t) in terminators for t in generated_ids)


def trim_incomplete(text: str) -> str:
    """Cut a length-truncated generation back to its last complete sentence.

    A turn that ends mid-sentence gets appended to the dialogue as a chat
    message, and the next model completes the dangling thought instead of
    replying to it. Trimming to a sentence boundary prevents that.
    """
    text = _drop_unclosed_fence(text)

    matches = list(SENTENCE_END.finditer(text))
    if not matches:
        # Nothing usable. Callers fall back to the raw generation rather than
        # emit an empty turn: substituting hand-written filler would put tokens
        # the policy never sampled into the trajectory.
        return ""

    return text[: matches[-1].end()].strip()


def drop_dangling_promise(text: str, max_passes: int = 10) -> str:
    """Remove a trailing clause that promises content which never follows.

    A turn ending in ':' ("Here is the implementation:") reads as an unfinished
    hand-off, so the next speaker completes the promise instead of replying to
    it. Measured at 19% of turns in the 2026-07-28 run, and it is the mechanism
    behind tutor/student role inversion.

    Applied to the student simulator only. The tutor is the policy under
    training, so editing its text would compute log-probs over tokens the model
    never sampled; the tutor is steered by the pedagogy reward instead.
    """
    s = text.rstrip()
    for _ in range(max_passes):
        if not s.endswith(":"):
            return s
        matches = list(SENTENCE_END.finditer(s))
        if not matches:
            return ""
        s = s[: matches[-1].end()].rstrip()
    return s


def _drop_unclosed_fence(text: str) -> str:
    """Remove a trailing ``` block that was cut off before its closing fence."""
    if text.count("```") % 2 == 1:
        text = text[: text.rfind("```")]
    return text


CODE_LINE = re.compile(
    r"^\s*(def |class |return |import |from |for |while |if |elif |else\b)"
)
ASSIGNMENT_LINE = re.compile(r"^\s*[\w.\[\]]+\s*(=[^=]|\+=|-=|\.\w+\()")


def strip_code(text: str) -> str:
    """Remove code from tutor text.

    Display and export only — never applied to a trajectory. GRPO must score
    the tokens the policy actually sampled, and the pedagogy judge must see the
    code in order to penalize it.
    """
    text = re.sub(r"```[\s\S]*?```", "", text)
    # A generation cut off mid-block leaves an unclosed fence that the
    # paired-fence pattern above cannot match.
    text = _drop_unclosed_fence(text)
    text = re.sub(r"`[^`]+`", "", text)
    lines = [
        l
        for l in text.splitlines()
        if not CODE_LINE.match(l) and not ASSIGNMENT_LINE.match(l)
    ]
    return "\n".join(lines).strip()
