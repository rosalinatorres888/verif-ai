"""Credibility score lookup by source name."""

CREDIBILITY_MAP = {
    "Reuters Fact Check": 0.95,
    "AP Fact Check": 0.95,
    "WHO": 0.97,
    "CDC": 0.97,
    "PolitiFact": 0.90,
    "Snopes": 0.88,
    "AFP Factuel": 0.92,
    "La Vanguardia Verificat": 0.88,
}


def get_credibility(source_name: str, default: float = 0.70) -> float:
    return CREDIBILITY_MAP.get(source_name, default)
