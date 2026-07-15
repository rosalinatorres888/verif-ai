"""
src/model_runner.py — Milestone 4 model pipeline entry point.

Run with:
    python src/model_runner.py

Loads the demo claim set, runs each claim through the real VerifAI pipeline
(intake -> retrieval -> classifier + Claude verdict — the same code path the
FastAPI backend uses), and saves human-readable results to outputs/samples.txt.

This calls live services (ChromaDB + Tavily retrieval, Anthropic for verdict
synthesis when available) — same cost/latency profile as using the app
itself. A single claim failing does not stop the run; the error is recorded
in place of that sample's verdict.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_claims
from utils.helpers import load_config, format_sample, ensure_parent_dir

from app.pipeline.intake import extract_claim
from app.pipeline.retrieval import retrieve_evidence
from app.pipeline.verdict import generate_verdict


def run_pipeline(config: dict) -> list[dict]:
    claims = load_claims(config["paths"]["demo_claims"])
    print(f"[model_runner] Loaded {len(claims)} demo claims from "
          f"{config['paths']['demo_claims']}")

    results = []
    for i, claim in enumerate(claims, start=1):
        print(f"[model_runner] ({i}/{len(claims)}) {claim['claim_id']}: "
              f"{claim['claim'][:60]}...")
        try:
            intake_result = extract_claim(claim["claim"])
            extracted_assertion = intake_result["extracted_assertion"]
            language = intake_result["language"]

            evidence = (
                retrieve_evidence(extracted_assertion, language)
                if config["pipeline"]["use_retrieval"]
                else []
            )

            verdict = generate_verdict(
                extracted_assertion=extracted_assertion,
                evidence=evidence,
                language=language,
            )
        except Exception as error:
            verdict = {
                "label": "error",
                "confidence": 0.0,
                "explanation": f"{type(error).__name__}: {error}",
                "evidence": [],
            }
            print(f"[model_runner]   -> failed: {verdict['explanation']}")

        results.append({"claim": claim, "verdict": verdict})

    return results


def save_samples(results: list[dict], output_path: str) -> None:
    ensure_parent_dir(output_path)
    blocks = [
        format_sample(i, r["claim"], r["verdict"])
        for i, r in enumerate(results, start=1)
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"VerifAI — Model Pipeline Sample Outputs\n"
                 f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                 f"{'=' * 60}\n\n")
        f.write(f"\n\n{'-' * 60}\n\n".join(blocks))
        f.write("\n")
    print(f"[model_runner] Saved {len(results)} samples to {output_path}")


def main():
    config = load_config()
    results = run_pipeline(config)
    save_samples(results, config["paths"]["samples_output"])

    n_ok = sum(1 for r in results if r["verdict"]["label"] != "error")
    print(f"[model_runner] Done: {n_ok}/{len(results)} claims verified successfully.")


if __name__ == "__main__":
    main()
