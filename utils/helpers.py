"""
Shared helpers for the src/ model pipeline (Milestone 4).
"""
from pathlib import Path

import yaml


def load_config(path: str = "configs/model_config.yaml") -> dict:
    """Load the model pipeline YAML config."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_sample(index: int, claim: dict, verdict: dict) -> str:
    """Render one claim + verdict as a readable block for outputs/samples.txt."""
    lang = "English" if claim.get("language") == "en" else "Spanish"
    lines = [
        f"[{index}] {claim.get('claim_id', 'n/a')} ({lang}, domain={claim.get('domain', 'n/a')})",
        f"Claim:        {claim['claim']}",
        f"Ground truth: {claim.get('ground_truth', 'n/a')}",
        f"Verdict:      {verdict.get('label', 'error')} "
        f"({verdict.get('confidence', 0):.0%} confidence)",
        f"Classifier:   {verdict.get('classifier_label', 'n/a')} "
        f"({verdict.get('classifier_confidence', 0):.0%})",
        f"Generation:   {verdict.get('generation_mode', 'n/a')}",
        f"Evidence:     {len(verdict.get('evidence', []))} source(s)",
        "Explanation:",
        f"  {verdict.get('explanation', '(none)')}",
    ]
    return "\n".join(lines)


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of `path` if it doesn't already exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
