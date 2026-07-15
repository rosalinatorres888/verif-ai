"""
Loads the curated demo claim set used by src/model_runner.py (Milestone 4).
"""
import json


def load_claims(path: str = "data/sample_claims.json") -> list[dict]:
    """Return the list of demo claims (each with claim_id, claim, language,
    ground_truth, and domain) used for the model pipeline demonstration."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
