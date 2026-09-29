"""Which Indic script is this?

The session used to tell the translator every turn was Hindi. IndicTrans2
does not reject a wrong source tag, it transliterates: Tamil tagged as Hindi
came back as "Enakku puriyavillai" rather than "I don't understand". That
reads like a real answer, so the failure was invisible from the outside.

These pin the detection and, more importantly, the two ways it could still do
harm: calling Latin text Indic (which would round-trip Hinglish through a
translator and destroy the code-mixing), and picking the wrong script when a
sentence contains more than one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backends import detect_indic_lang, has_indic_script  # noqa: E402


def test_tamil_is_tamil_not_hindi():
    """The bug this function exists for."""
    assert detect_indic_lang("எனக்கு புரியவில்லை") == "ta"


def test_hindi_is_still_hindi():
    assert detect_indic_lang("मुझे समझ नहीं आया") == "hi"


def test_each_supported_script_resolves_to_its_own_language():
    cases = {
        "hi": "नमस्ते",
        "bn": "নমস্কার",
        "pa": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ",
        "gu": "નમસ્તે",
        "ta": "வணக்கம்",
        "te": "నమస్కారం",
        "kn": "ನಮಸ್ಕಾರ",
        "ml": "നമസ്കാരം",
    }
    for expected, text in cases.items():
        assert detect_indic_lang(text) == expected, f"{text} should be {expected}"


def test_romanised_hinglish_is_not_translated():
    """Already Latin. The tutor reads it better than Devanagari, and
    round-tripping it would lose the code-mixing the project is about."""
    text = "kya hum arrays use kar sakte hain yahan pe?"
    assert detect_indic_lang(text) is None
    assert has_indic_script(text) is False


def test_plain_english_is_not_indic():
    assert detect_indic_lang("I don't understand this problem") is None


def test_empty_and_none_safe():
    assert detect_indic_lang("") is None
    assert detect_indic_lang(None) is None  # type: ignore[arg-type]


def test_dominant_script_wins_in_a_mixed_sentence():
    """A Tamil sentence quoting one Devanagari word is still Tamil."""
    assert detect_indic_lang("எனக்கு नहीं புரியவில்லை") == "ta"


def test_indic_mixed_with_latin_still_detected():
    """Code-switching mid-sentence is the normal case here, not an edge one."""
    assert detect_indic_lang("मुझे hash map समझ नहीं आया") == "hi"


def test_every_detected_language_has_a_flores_tag():
    """Detection that names a language the translator cannot tag would trade
    one silent failure for another."""
    from app.backends import _FLORES, _SCRIPT_BLOCKS

    for lang, _lo, _hi in _SCRIPT_BLOCKS:
        assert lang in _FLORES, f"{lang} is detectable but has no FLORES tag"
