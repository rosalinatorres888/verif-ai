"""
app/pipeline/reranker.py
Block F — Evidence reranker using trained VerifAIClassifier.

Scores each (claim, passage) pair for relevance using the trained model.
Returns P(true | claim + passage) as a proxy for claim-evidence relevance.

This is the novel dual-use of the from-scratch classifier:
  1. Verdict classifier (verdict.py)
  2. Evidence reranker (this file)

No pretrained weights. Loads ~/verif-ai/models/verifai-classifier/best_model.pt

Usage:
    from app.pipeline.reranker import score_relevance
    score = score_relevance("Vaccines cause autism.", "WHO: vaccines do not cause autism...")
    # returns float 0.0–1.0
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model.tokenizer import BPETokenizer
from model.architecture import VerifAIClassifier

VOCAB_PATH = Path(__file__).parent.parent.parent / "models/verifai-classifier/vocab.json"
CKPT_PATH  = Path(__file__).parent.parent.parent / "models/verifai-classifier/best_model.pt"
MAX_LENGTH = 256

_tokenizer = None
_model     = None
_device    = None


def _load():
    """Lazy-load tokenizer and model. Cached after first call."""
    global _tokenizer, _model, _device

    if _model is not None:
        return _tokenizer, _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    if not VOCAB_PATH.exists():
        print(f"[reranker] vocab.json not found at {VOCAB_PATH} — reranker disabled")
        return None, None, None

    _tokenizer = BPETokenizer.load(str(VOCAB_PATH))

    # Load checkpoint
    if not CKPT_PATH.exists():
        print(f"[reranker] best_model.pt not found at {CKPT_PATH} — reranker disabled")
        return None, None, None

    ckpt   = torch.load(CKPT_PATH, map_location=_device, weights_only=False)
    config = ckpt.get("config", {})

    _model = VerifAIClassifier(
        vocab_size  = config.get("vocab_size", 16000),
        embed_dim   = config.get("embed_dim", 256),
        num_heads   = config.get("num_heads", 8),
        num_layers  = config.get("num_layers", 4),
        hidden_dim  = config.get("hidden_dim", 512),
        max_length  = MAX_LENGTH,
        num_classes = config.get("num_labels", 4),
        dropout     = 0.0,   # eval mode
    ).to(_device)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()

    print(f"[reranker] Loaded checkpoint (epoch {ckpt.get('epoch')}, "
          f"val_f1={ckpt.get('val_f1', 0):.4f})")
    return _tokenizer, _model, _device


@torch.no_grad()
def score_relevance(claim: str, passage: str, language: str = "en") -> float:
    """
    Score how relevant a passage is to a claim.

    Uses encode_pair([CLS] claim [SEP] passage) → classifier → P(true).
    P(true) is used as a relevance proxy: if the model thinks this passage
    supports or refutes the claim clearly, it's more relevant than noise.

    Args:
        claim:    extracted assertion string
        passage:  evidence passage string
        language: "en" or "es"

    Returns:
        float 0.0–1.0 (higher = more relevant)
    """
    tokenizer, model, device = _load()

    if model is None:
        return 0.5   # neutral fallback if model not loaded

    try:
        # Encode claim + passage as a pair
        input_ids, token_type_ids = tokenizer.encode_pair(
            claim, passage[:400], max_length=MAX_LENGTH
        )
        input_ids_padded, attention_mask = tokenizer.pad(input_ids, MAX_LENGTH)

        # Build tensors
        ids_tensor  = torch.tensor([input_ids_padded], dtype=torch.long).to(device)
        mask_tensor = torch.tensor([attention_mask],   dtype=torch.long).to(device)
        lang_id     = torch.tensor([0 if language == "en" else 1],
                                   dtype=torch.long).to(device)

        # Forward pass
        logits = model(ids_tensor, mask_tensor, lang_id)
        probs  = F.softmax(logits, dim=-1)[0]

        # P(true) + P(false) as combined relevance signal
        # Both indicate the passage is directly related to the claim
        relevance = float((probs[0] + probs[1]).detach())   # true=0, false=1
        return round(min(max(relevance, 0.0), 1.0), 4)

    except Exception as e:
        print(f"[reranker] Scoring error: {e}")
        return 0.5


def score_batch(claim: str, passages: list, language: str = "en") -> list:
    """
    Score multiple passages for a single claim.
    Returns list of floats in same order as input passages.
    More efficient than calling score_relevance in a loop.
    """
    tokenizer, model, device = _load()

    if model is None:
        return [0.5] * len(passages)

    scores = []
    try:
        batch_ids, batch_masks, batch_langs = [], [], []
        lang_id = 0 if language == "en" else 1

        for passage in passages:
            input_ids, _ = tokenizer.encode_pair(
                claim, passage[:400], max_length=MAX_LENGTH
            )
            input_ids_padded, attention_mask = tokenizer.pad(input_ids, MAX_LENGTH)
            batch_ids.append(input_ids_padded)
            batch_masks.append(attention_mask)
            batch_langs.append(lang_id)

        ids_tensor  = torch.tensor(batch_ids,   dtype=torch.long).to(device)
        mask_tensor = torch.tensor(batch_masks, dtype=torch.long).to(device)
        lang_tensor = torch.tensor(batch_langs, dtype=torch.long).to(device)

        logits = model(ids_tensor, mask_tensor, lang_tensor)
        probs  = F.softmax(logits, dim=-1)

        for p in probs:
            relevance = float((p[0] + p[1]).detach())
            scores.append(round(min(max(relevance, 0.0), 1.0), 4))

    except Exception as e:
        print(f"[reranker] Batch scoring error: {e}")
        scores = [0.5] * len(passages)

    return scores
