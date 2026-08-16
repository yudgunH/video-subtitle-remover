"""Conservative helpers for recognizing Chinese text in OCR output.

The feature originally accepted any CJK code point, including a single Kana
or Hangul character at low confidence. That is unsafe for full-frame video:
small components and decorative details can be misread as one character and
then passed to the inpainting model. These helpers deliberately prefer false
negatives over deleting real image content.
"""

from __future__ import annotations


_HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
    (0x30000, 0x323AF),
)

_KANA_RANGES = (
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0x31F0, 0x31FF),
    (0xFF66, 0xFF9D),
)

_HANGUL_RANGES = (
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7AF),
    (0xD7B0, 0xD7FF),
)


def _in_ranges(character: str, ranges) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in ranges)


def contains_han(text: str) -> bool:
    return bool(text) and any(_in_ranges(character, _HAN_RANGES) for character in text)


def contains_kana(text: str) -> bool:
    return bool(text) and any(_in_ranges(character, _KANA_RANGES) for character in text)


def contains_hangul(text: str) -> bool:
    return bool(text) and any(_in_ranges(character, _HANGUL_RANGES) for character in text)


def han_character_count(text: str) -> int:
    return sum(1 for character in str(text) if _in_ranges(character, _HAN_RANGES))


def is_chinese_recognition(
    text: str,
    score: float,
    minimum_score: float = 0.85,
    minimum_han_characters: int = 2,
) -> bool:
    """Return whether OCR output is safe enough for Chinese-text removal.

    Han characters are shared by Chinese and Japanese, so isolated Kanji can
    never be classified perfectly from Unicode alone. Rejecting any result
    containing Kana/Hangul and requiring multiple high-confidence Han
    characters makes full-frame removal intentionally conservative.
    """
    normalized = str(text or "").strip()
    try:
        confidence = float(score)
    except (TypeError, ValueError):
        return False
    if confidence < minimum_score:
        return False
    if contains_kana(normalized) or contains_hangul(normalized):
        return False
    return han_character_count(normalized) >= minimum_han_characters


# Compatibility names retained for external callers. Their behavior now
# follows the Chinese-only mode selected for this application.
contains_cjk = contains_han
is_cjk_recognition = is_chinese_recognition
