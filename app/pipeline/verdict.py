"""
Layer 3 — Verdict Generation

Block G:
Replaced the earlier XLM-RoBERTa sentiment proxy with the trained
VerifAIClassifier.

Claude is optional. If the Anthropic API is unavailable, VerifAI returns
a classifier-and-evidence fallback verdict instead of failing.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import anthropic
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env",
    override=True,
)

sys.path.insert(0, str(PROJECT_ROOT))

from model.architecture import VerifAIClassifier
from model.tokenizer import BPETokenizer


VOCAB_PATH = PROJECT_ROOT / "models" / "verifai-classifier" / "vocab.json"
CKPT_PATH = PROJECT_ROOT / "models" / "verifai-classifier" / "best_model.pt"

MAX_LENGTH = 256

LABEL_NAMES = [
    "true",
    "false",
    "misleading",
    "unverifiable",
]

_tokenizer = None
_classifier = None
_device = None


VERDICT_PROMPT = """
You are a fact-checking assistant.

Given the following claim and evidence, produce a structured verdict.
Respond in {language}.

Claim:
{extracted_assertion}

Evidence:
{evidence_passages}

VerifAI classifier signal:
{classifier_label} (confidence: {classifier_confidence:.2f})

This signal comes from a custom bilingual transformer trained from
scratch on LIAR, MultiFC, FakeDeS, and reviewed synthetic Spanish data.

Return ONLY valid JSON with these fields:

- label: one of "true", "false", "misleading", or "unverifiable"
- confidence: float from 0.0 to 1.0
- explanation: 3–5 plain-language sentences in {language}
- key_evidence: list containing 1–3 source names used

Rules:

- Do not fabricate sources.
- Use only the evidence provided above.
- Use "unverifiable" when the evidence is insufficient.
- Match the input language exactly.
""".strip()


def _load_classifier():
    """Load and cache the trained tokenizer and classifier."""

    global _tokenizer, _classifier, _device

    if _classifier is not None:
        return _tokenizer, _classifier, _device

    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    if not VOCAB_PATH.exists() or not CKPT_PATH.exists():
        print("[verdict] Classifier files not found — classifier disabled")
        return None, None, None

    try:
        _tokenizer = BPETokenizer.load(str(VOCAB_PATH))

        checkpoint = torch.load(
            CKPT_PATH,
            map_location=_device,
            weights_only=False,
        )

        config = checkpoint.get("config", {})

        _classifier = VerifAIClassifier(
            vocab_size=config.get("vocab_size", 16000),
            embed_dim=config.get("embed_dim", 256),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 4),
            hidden_dim=config.get("hidden_dim", 512),
            max_length=MAX_LENGTH,
            num_classes=config.get("num_labels", 4),
            dropout=0.0,
        ).to(_device)

        _classifier.load_state_dict(checkpoint["model_state"])
        _classifier.eval()

        print(
            "[verdict] VerifAIClassifier loaded "
            f"(epoch {checkpoint.get('epoch')}, "
            f"val_f1={checkpoint.get('val_f1', 0):.4f}, "
            f"device={_device})"
        )

    except Exception as error:
        print(f"[verdict] Classifier load failed: {error}")
        _tokenizer = None
        _classifier = None
        _device = None
        return None, None, None

    return _tokenizer, _classifier, _device


@torch.no_grad()
def run_classifier(
    text: str,
    language: str = "en",
) -> tuple[str, float]:
    """Run the custom four-class VerifAI classifier."""

    tokenizer, classifier, device = _load_classifier()

    if classifier is None or tokenizer is None:
        return "unverifiable", 0.0

    try:
        input_ids = tokenizer.encode(
            text,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )

        padded_ids, attention_mask = tokenizer.pad(
            input_ids,
            MAX_LENGTH,
        )

        ids_tensor = torch.tensor(
            [padded_ids],
            dtype=torch.long,
            device=device,
        )

        mask_tensor = torch.tensor(
            [attention_mask],
            dtype=torch.long,
            device=device,
        )

        language_id = 1 if language == "es" else 0

        language_tensor = torch.tensor(
            [language_id],
            dtype=torch.long,
            device=device,
        )

        logits = classifier(
            ids_tensor,
            mask_tensor,
            language_tensor,
        )

        probabilities = F.softmax(logits, dim=-1)[0]

        prediction_index = int(
            torch.argmax(probabilities).item()
        )

        confidence = float(
            probabilities[prediction_index].detach().cpu().item()
        )

        return (
            LABEL_NAMES[prediction_index],
            round(confidence, 4),
        )

    except Exception as error:
        print(f"[verdict] Classifier inference error: {error}")
        return "unverifiable", 0.0


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert tensor, NumPy, string, or numeric values safely."""

    try:
        if hasattr(value, "item"):
            value = value.item()

        return float(value)

    except (TypeError, ValueError):
        return default


def _format_evidence(evidence: list[dict]) -> str:
    """Format retrieved evidence for the Claude prompt."""

    if not evidence:
        return "No evidence retrieved."

    lines = []

    for index, item in enumerate(evidence[:5], start=1):
        source_name = item.get(
            "source_name",
            item.get("source", "Unknown source"),
        )

        credibility = _safe_float(
            item.get("credibility_score"),
            0.0,
        )

        reranker_score = item.get("reranker_score")

        reranker_text = ""

        if reranker_score is not None:
            reranker_text = (
                f" · reranker: "
                f"{_safe_float(reranker_score):.2f}"
            )

        passage = str(
            item.get(
                "passage",
                item.get("text", ""),
            )
        )

        lines.append(
            f"{index}. [{source_name}] "
            f"(credibility: {credibility:.2f}{reranker_text})\n"
            f"   {passage[:400]}"
        )

    return "\n\n".join(lines)


def _extract_source_names(
    evidence: list[dict],
    limit: int = 3,
) -> list[str]:
    """Return a deduplicated list of evidence source names."""

    sources = []

    for item in evidence:
        source_name = item.get(
            "source_name",
            item.get("source"),
        )

        if source_name and source_name not in sources:
            sources.append(str(source_name))

        if len(sources) >= limit:
            break

    return sources


def _build_fallback_explanation(
    classifier_label: str,
    classifier_confidence: float,
    evidence: list[dict],
    language: str,
) -> str:
    """Build an honest non-generative explanation."""

    evidence_count = len(evidence)

    if language == "es":
        if evidence_count:
            return (
                f"El clasificador de VerifAI asignó la etiqueta "
                f"“{classifier_label}” con una confianza de "
                f"{classifier_confidence:.0%}. "
                f"El sistema recuperó {evidence_count} fragmentos de evidencia "
                "para que el resultado pueda revisarse junto con las fuentes. "
                "La explicación generativa no está disponible porque el servicio "
                "externo de Anthropic no pudo utilizarse. "
                "Este resultado debe interpretarse como una señal preliminar del "
                "clasificador, no como una verificación definitiva."
            )

        return (
            f"El clasificador de VerifAI asignó la etiqueta "
            f"“{classifier_label}” con una confianza de "
            f"{classifier_confidence:.0%}. "
            "No se recuperó evidencia suficiente para respaldar una conclusión. "
            "La explicación generativa no está disponible porque el servicio "
            "externo de Anthropic no pudo utilizarse. "
            "Por lo tanto, el resultado debe considerarse preliminar."
        )

    if evidence_count:
        return (
            f"The VerifAI classifier assigned the label "
            f"“{classifier_label}” with "
            f"{classifier_confidence:.0%} confidence. "
            f"The system retrieved {evidence_count} evidence passages so the "
            "prediction can be reviewed alongside its sources. "
            "Generative explanation is unavailable because the external "
            "Anthropic service could not be used. "
            "This result should be treated as a preliminary classifier signal, "
            "not as a definitive fact-check."
        )

    return (
        f"The VerifAI classifier assigned the label "
        f"“{classifier_label}” with "
        f"{classifier_confidence:.0%} confidence. "
        "The system did not retrieve enough evidence to support a firm "
        "conclusion. "
        "Generative explanation is unavailable because the external Anthropic "
        "service could not be used. "
        "The result should therefore be treated as preliminary."
    )


def _fallback_verdict(
    classifier_label: str,
    classifier_confidence: float,
    evidence: list[dict],
    language: str,
    reason: str,
) -> dict:
    """Return a working verdict without Claude."""

    print(
        "[verdict] Using classifier-and-evidence fallback: "
        f"{reason}"
    )

    return {
        "label": classifier_label,
        "confidence": float(classifier_confidence),
        "classifier_label": classifier_label,
        "classifier_confidence": float(classifier_confidence),
        "evidence": evidence,
        "explanation": _build_fallback_explanation(
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
            evidence=evidence,
            language=language,
        ),
        "key_evidence": _extract_source_names(evidence),
        "retrieval_method": "corpus" if evidence else "none",
        "generation_mode": "classifier_fallback",
        "generation_warning": reason,
    }


def generate_verdict(
    extracted_assertion: str,
    evidence: list,
    language: str,
) -> dict:
    """
    Generate the final verdict.

    Claude is used when an API key and sufficient credits are available.
    Otherwise, return a classifier-and-evidence fallback response.
    """

    classifier_label, classifier_confidence = run_classifier(
        extracted_assertion,
        language,
    )

    api_key = os.environ.get(
        "ANTHROPIC_API_KEY",
        "",
    ).strip()

    if not api_key:
        return _fallback_verdict(
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
            evidence=evidence,
            language=language,
            reason="ANTHROPIC_API_KEY is not configured.",
        )

    evidence_text = _format_evidence(evidence)

    language_name = (
        "Spanish"
        if language == "es"
        else "English"
    )

    prompt = VERDICT_PROMPT.format(
        language=language_name,
        extracted_assertion=extracted_assertion,
        evidence_passages=evidence_text,
        classifier_label=classifier_label,
        classifier_confidence=classifier_confidence,
    )

    try:
        client = anthropic.Anthropic(
            api_key=api_key,
        )

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        raw = response.content[0].text.strip()

        raw = re.sub(
            r"^```json\s*|^```\s*|\s*```$",
            "",
            raw,
            flags=re.MULTILINE,
        ).strip()

        parsed = json.loads(raw)

        return {
            "label": parsed.get(
                "label",
                classifier_label,
            ),
            "confidence": _safe_float(
                parsed.get(
                    "confidence",
                    classifier_confidence,
                ),
                classifier_confidence,
            ),
            "classifier_label": classifier_label,
            "classifier_confidence": float(
                classifier_confidence
            ),
            "evidence": evidence,
            "explanation": parsed.get(
                "explanation",
                "",
            ),
            "key_evidence": parsed.get(
                "key_evidence",
                _extract_source_names(evidence),
            ),
            "retrieval_method": (
                "corpus"
                if evidence
                else "none"
            ),
            "generation_mode": "claude",
        }

    except json.JSONDecodeError as error:
        return _fallback_verdict(
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
            evidence=evidence,
            language=language,
            reason=(
                "Claude returned invalid JSON: "
                f"{error}"
            ),
        )

    except anthropic.APIError as error:
        return _fallback_verdict(
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
            evidence=evidence,
            language=language,
            reason=(
                "Anthropic API unavailable: "
                f"{type(error).__name__}"
            ),
        )

    except Exception as error:
        return _fallback_verdict(
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
            evidence=evidence,
            language=language,
            reason=(
                "Verdict generation failed: "
                f"{type(error).__name__}: {error}"
            ),
        )