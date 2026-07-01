"""
Layer 3 — Verdict Generation
Block G: replaced XLM-RoBERTa proxy with trained VerifAIClassifier.
"""
import os, json, re, sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import torch
import torch.nn.functional as F

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model.tokenizer import BPETokenizer
from model.architecture import VerifAIClassifier

VOCAB_PATH  = Path(__file__).parent.parent.parent / "models/verifai-classifier/vocab.json"
CKPT_PATH   = Path(__file__).parent.parent.parent / "models/verifai-classifier/best_model.pt"
MAX_LENGTH  = 256
LABEL_NAMES = ["true", "false", "misleading", "unverifiable"]

_tokenizer  = None
_classifier = None
_device     = None

VERDICT_PROMPT = """You are a fact-checking assistant. Given the following claim and evidence, produce a structured verdict. Respond in {language}.

Claim: {extracted_assertion}

Evidence:
{evidence_passages}

VerifAI classifier signal: {classifier_label} (confidence: {classifier_confidence:.2f})
This signal is from a custom bilingual transformer trained from scratch on LIAR + MultiFC + FakeDeS.

Return ONLY valid JSON with these fields:
- label: one of "true" | "false" | "misleading" | "unverifiable"
- confidence: float 0.0-1.0
- explanation: 3-5 sentences in {language}, plain language, no jargon
- key_evidence: list of 1-3 source names used

Rules:
- Do not fabricate sources. Use only sources provided in Evidence above.
- Use "unverifiable" if evidence is insufficient to reach a verdict.
- explanation must be in {language} — match the input language exactly."""


def _load_classifier():
    global _tokenizer, _classifier, _device
    if _classifier is not None:
        return _tokenizer, _classifier, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not VOCAB_PATH.exists() or not CKPT_PATH.exists():
        print("[verdict] Classifier files not found — disabled")
        return None, None, None
    try:
        _tokenizer = BPETokenizer.load(str(VOCAB_PATH))
        ckpt   = torch.load(CKPT_PATH, map_location=_device, weights_only=False)
        config = ckpt.get("config", {})
        _classifier = VerifAIClassifier(
            vocab_size  = config.get("vocab_size", 16000),
            embed_dim   = config.get("embed_dim", 256),
            num_heads   = config.get("num_heads", 8),
            num_layers  = config.get("num_layers", 4),
            hidden_dim  = config.get("hidden_dim", 512),
            max_length  = MAX_LENGTH,
            num_classes = config.get("num_labels", 4),
            dropout     = 0.0,
        ).to(_device)
        _classifier.load_state_dict(ckpt["model_state"])
        _classifier.eval()
        print(f"[verdict] VerifAIClassifier loaded (epoch {ckpt.get('epoch')}, val_f1={ckpt.get('val_f1', 0):.4f})")
    except Exception as e:
        print(f"[verdict] Classifier load failed: {e}")
        return None, None, None
    return _tokenizer, _classifier, _device


@torch.no_grad()
def run_classifier(text: str, language: str = "en") -> tuple:
    tokenizer, classifier, device = _load_classifier()
    if classifier is None:
        return "unknown", 0.0
    try:
        input_ids = tokenizer.encode(text, max_length=MAX_LENGTH, add_special_tokens=True)
        input_ids_padded, attention_mask = tokenizer.pad(input_ids, MAX_LENGTH)
        ids_tensor  = torch.tensor([input_ids_padded], dtype=torch.long).to(device)
        mask_tensor = torch.tensor([attention_mask],   dtype=torch.long).to(device)
        lang_tensor = torch.tensor([0 if language == "en" else 1], dtype=torch.long).to(device)
        logits = classifier(ids_tensor, mask_tensor, lang_tensor)
        probs  = F.softmax(logits, dim=-1)[0]
        pred_idx   = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].detach())
        return LABEL_NAMES[pred_idx], round(confidence, 4)
    except Exception as e:
        print(f"[verdict] Classifier inference error: {e}")
        return "unknown", 0.0


def _format_evidence(evidence: list) -> str:
    if not evidence:
        return "No evidence retrieved."
    lines = []
    for i, e in enumerate(evidence[:5], 1):
        rs = e.get("reranker_score", "")
        rs_str = f" · reranker: {rs:.2f}" if rs else ""
        lines.append(
            f"{i}. [{e['source_name']}] (credibility: {e['credibility_score']:.2f}{rs_str})\n"
            f"   {e['passage'][:400]}"
        )
    return "\n\n".join(lines)


def generate_verdict(extracted_assertion: str, evidence: list, language: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client  = anthropic.Anthropic(api_key=api_key)
    classifier_label, classifier_confidence = run_classifier(extracted_assertion, language)
    evidence_text = _format_evidence(evidence)
    lang_name     = "Spanish" if language == "es" else "English"
    prompt = VERDICT_PROMPT.format(
        language=lang_name,
        extracted_assertion=extracted_assertion,
        evidence_passages=evidence_text,
        classifier_label=classifier_label,
        classifier_confidence=classifier_confidence
    )
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude returned invalid JSON: {e}\nRaw: {raw}")
    return {
        "label":                 parsed.get("label", "unverifiable"),
        "confidence":            float(parsed.get("confidence", 0.5)),
        "classifier_label":      classifier_label,
        "classifier_confidence": classifier_confidence,
        "evidence":              evidence,
        "explanation":           parsed.get("explanation", ""),
        "key_evidence":          parsed.get("key_evidence", []),
        "retrieval_method":      "corpus" if evidence else "none",
    }
