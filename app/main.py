"""
VerifAI — FastAPI backend entry point
Exposes POST /verify endpoint consumed by the Streamlit frontend.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime, timezone

from app.pipeline.intake import extract_claim
from app.pipeline.retrieval import retrieve_evidence
from app.pipeline.verdict import generate_verdict

app = FastAPI(title="VerifAI", version="1.0.0")


class ClaimRequest(BaseModel):
    claim: str


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/verify")
def verify(request: ClaimRequest, no_rag: bool = False):
    """
    Main verification endpoint.
    - no_rag=True: skip retrieval (used for ablation toggle in UI)
    """
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="claim cannot be empty")

    try:
        intake_result = extract_claim(request.claim)
        evidence = [] if no_rag else retrieve_evidence(
            intake_result["extracted_assertion"],
            intake_result["language"]
        )
        verdict = generate_verdict(
            extracted_assertion=intake_result["extracted_assertion"],
            evidence=evidence,
            language=intake_result["language"]
        )
        verdict["claim_id"] = str(uuid.uuid4())
        verdict["original_claim"] = request.claim
        verdict["language"] = intake_result["language"]
        verdict["retrieval_method"] = "none" if no_rag else verdict.get("retrieval_method", "corpus")
        verdict["timestamp"] = datetime.now(timezone.utc).isoformat()
        return verdict

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
