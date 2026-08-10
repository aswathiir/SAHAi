from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sahai.settings import ASRSettings

PROGRAMMING_KEYWORDS = {
    "array", "list", "dict", "dictionary", "hash", "map", "set", "tuple",
    "stack", "queue", "tree", "graph", "node", "linked", "pointer",
    "sort", "search", "binary", "linear", "recursion", "recursive",
    "loop", "iterate", "index", "append", "pop", "push", "insert",
    "function", "return", "variable", "parameter", "argument",
    "string", "integer", "float", "boolean", "null", "none",
    "if", "else", "for", "while", "break", "continue",
    "class", "object", "method", "import", "print", "input",
    "complexity", "time", "space", "big", "notation",
    "dynamic", "programming", "greedy", "backtracking",
    "dfs", "bfs", "traversal", "preorder", "inorder", "postorder",
}

HINGLISH_NOISE = {
    "kaise": ["kese", "kaise", "kase"],
    "samajh": ["samjh", "samaj", "smjh"],
    "nahi": ["nhi", "nahin", "nai"],
    "karna": ["krna", "karna", "krna"],
    "hoga": ["hoga", "hga", "hoega"],
    "kya": ["kya", "kia", "kyaa"],
}

TANGLISH_NOISE = {
    "puriyala": ["purila", "puriyale", "puriyla"],
    "epdhi": ["epdi", "eppadi", "epdi"],
    "pannanum": ["pannum", "panum", "pannanum"],
    "sollu": ["sollu", "solu", "chollu"],
}


class ASREngine:
    def __init__(self, settings: ASRSettings | None = None):
        from sahai.settings import ASRSettings

        self.settings = settings or ASRSettings()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        import whisper

        model_name = self.settings.model.replace("openai/whisper-", "")
        self._model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str) -> str:
        self._load_model()
        result = self._model.transcribe(
            audio_path,
            language=self.settings.language,
            beam_size=self.settings.beam_size,
        )
        return result["text"].strip()

    def inject_asr_noise(self, text: str, error_rate: float | None = None) -> str:
        rate = error_rate if error_rate is not None else self.settings.noise_rate
        words = text.split()
        noisy = []
        for word in words:
            if word.lower() in PROGRAMMING_KEYWORDS and self.settings.preserve_keywords:
                noisy.append(word)
                continue
            if random.random() < rate:
                noisy.append(self._corrupt_word(word))
            else:
                noisy.append(word)
        return " ".join(noisy)

    def _corrupt_word(self, word: str) -> str:
        lower = word.lower()
        if lower in HINGLISH_NOISE:
            return random.choice(HINGLISH_NOISE[lower])
        if lower in TANGLISH_NOISE:
            return random.choice(TANGLISH_NOISE[lower])
        corruptions = [
            self._swap_chars,
            self._drop_char,
            self._duplicate_char,
        ]
        return random.choice(corruptions)(word)

    def _swap_chars(self, word: str) -> str:
        if len(word) < 3:
            return word
        i = random.randint(1, len(word) - 2)
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    def _drop_char(self, word: str) -> str:
        if len(word) < 3:
            return word
        i = random.randint(1, len(word) - 1)
        return word[:i] + word[i + 1 :]

    def _duplicate_char(self, word: str) -> str:
        if len(word) < 2:
            return word
        i = random.randint(0, len(word) - 1)
        return word[:i] + word[i] + word[i:]
