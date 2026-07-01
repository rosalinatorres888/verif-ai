"""
Layer 1 — Intake & Preprocessing
Detects language and extracts the core falsifiable assertion from a raw claim.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

# Force-load .env from project root, overriding any shell environment variables
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

CLAIM_EXTRACTION_PROMPT = """You are a fact-checking assistant. Extract only the core falsifiable factual claim \
from the following text. Remove all opinion, emotion, hedging, and context. \
Return only the bare claim as a single sentence. Do not add explanation.

Input: {claim}"""


def detect_language(text: str) -> str:
    """Return 'en' or 'es'. Defaults to 'en' if detection fails or language unsupported."""
    try:
        lang = detect(text)
        return lang if lang in ("en", "es") else "en"
    except LangDetectException:
        return "en"


def extract_claim(raw_claim: str) -> dict:
    """
    Args:
        raw_claim: raw user input string
    Returns:
        {"language": "en"|"es", "extracted_assertion": str, "original_claim": str}
    """
    import anthropic

    language = detect_language(raw_claim)

    # Re-read key at call time so .env override is always respected
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[intake] ANTHROPIC_API_KEY not set — using raw claim as fallback")
        return {"language": language, "extracted_assertion": raw_claim.strip(), "original_claim": raw_claim}

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": CLAIM_EXTRACTION_PROMPT.format(claim=raw_claim)
            }]
        )
        extracted = response.content[0].text.strip()
    except Exception as e:
        print(f"[intake] Claude API error — using raw claim as fallback: {e}")
        extracted = raw_claim.strip()

    return {
        "language": language,
        "extracted_assertion": extracted,
        "original_claim": raw_claim
    }
