"""
VerifAI — FastAPI backend entry point
Exposes POST /verify endpoint consumed by the Streamlit frontend.
"""

import traceback
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.pipeline.intake import extract_claim
from app.pipeline.retrieval import retrieve_evidence
from app.pipeline.verdict import generate_verdict


app = FastAPI(title="VerifAI", version="1.0.0")


class ClaimRequest(BaseModel):
    claim: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/verify")
def verify(request: ClaimRequest, no_rag: bool = False):
    """
    Main verification endpoint.

    - no_rag=True: skip retrieval
    - used by the Streamlit ablation toggle
    """

    if not request.claim.strip():
        raise HTTPException(
            status_code=400,
            detail="claim cannot be empty",
        )

    try:
        intake_result = extract_claim(request.claim)

        extracted_assertion = intake_result["extracted_assertion"]
        language = intake_result["language"]

        evidence = (
            []
            if no_rag
            else retrieve_evidence(
                extracted_assertion,
                language,
            )
        )

        verdict = generate_verdict(
            extracted_assertion=extracted_assertion,
            evidence=evidence,
            language=language,
        )

        verdict["claim_id"] = str(uuid.uuid4())
        verdict["original_claim"] = request.claim
        verdict["language"] = language
        verdict["retrieval_method"] = (
            "none"
            if no_rag
            else verdict.get("retrieval_method", "corpus")
        )
        verdict["timestamp"] = datetime.now(
            timezone.utc
        ).isoformat()

        return verdict

    except HTTPException:
        raise

    except Exception as error:
        print("\n[verify] Unhandled exception:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        ) from error