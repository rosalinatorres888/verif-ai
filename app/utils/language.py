"""Language detection helpers."""
from langdetect import detect, LangDetectException

SUPPORTED = {"en", "es"}


def detect_language(text: str, default: str = "en") -> str:
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED else default
    except LangDetectException:
        return default


def language_display(code: str) -> str:
    return {"en": "English", "es": "Spanish"}.get(code, "English")
