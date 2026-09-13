"""
tests/test_segment_crossencoder.py
Sanity tests for lever 1: the segment (token-type) embedding that turns the
claim-only classifier into a claim+evidence cross-encoder.

Run where torch is installed (OOD / Colab):
    pytest tests/test_segment_crossencoder.py -q
"""
import torch
import pytest

from model.architecture import VerifAIClassifier

VOCAB, B, T, NUM_CLASSES = 200, 4, 16, 4


def _model():
    return VerifAIClassifier(
        vocab_size=VOCAB, embed_dim=32, num_heads=4,
        num_layers=2, hidden_dim=64, max_length=T, num_classes=NUM_CLASSES,
    )


def _batch():
    input_ids      = torch.randint(1, VOCAB, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    language_ids   = torch.randint(0, 2, (B,))
    token_type_ids = torch.randint(0, 2, (B, T))
    return input_ids, attention_mask, language_ids, token_type_ids


def test_claim_only_shape_unchanged():
    """Claim-only path (token_type_ids omitted) still returns [B, num_classes]."""
    model = _model()
    ids, mask, lang, _ = _batch()
    logits = model(ids, mask, lang)
    assert logits.shape == (B, NUM_CLASSES)


def test_cross_encoder_shape():
    """Passing token_type_ids returns the same logits shape."""
    model = _model()
    ids, mask, lang, tt = _batch()
    logits = model(ids, mask, lang, tt)
    assert logits.shape == (B, NUM_CLASSES)


def test_backward_compatible_when_none():
    """token_type_ids=None must be identical to not passing it at all."""
    model = _model().eval()
    ids, mask, lang, _ = _batch()
    with torch.no_grad():
        a = model(ids, mask, lang)
        b = model(ids, mask, lang, token_type_ids=None)
    assert torch.allclose(a, b)


def test_segment_signal_changes_output():
    """The segment embedding must actually influence the output when supplied."""
    model = _model().eval()
    ids, mask, lang, _ = _batch()
    seg_a = torch.zeros(B, T, dtype=torch.long)   # all claim
    seg_b = torch.ones(B, T, dtype=torch.long)    # all evidence
    with torch.no_grad():
        out_a = model(ids, mask, lang, seg_a)
        out_b = model(ids, mask, lang, seg_b)
    assert not torch.allclose(out_a, out_b), "segment embedding had no effect"


def test_gradients_flow_to_segment_embedding():
    """Backprop reaches segment_emb when token_type_ids is used."""
    model = _model()
    ids, mask, lang, tt = _batch()
    logits = model(ids, mask, lang, tt)
    loss = logits.sum()
    loss.backward()
    assert model.segment_emb.weight.grad is not None
    assert model.segment_emb.weight.grad.abs().sum() > 0
