# ADR-001: XLM-RoBERTa Sentiment Model as Fake/Real Proxy Signal

Status: **Superseded by [ADR-003](adr-003-verifai-classifier-replaces-xlm-roberta.md)** | Date: 2026-05-30

> **Note (Block G):** This decision was replaced once the from-scratch VerifAIClassifier
> reached a usable checkpoint. The XLM-RoBERTa sentiment proxy described below is no
> longer used in `app/pipeline/verdict.py`. This ADR is kept for historical context —
> the alternatives-considered reasoning below still explains why a proxy signal was
> needed in the first place.

## Context

The verdict layer needs a lightweight, bilingual first-pass classification signal before calling Claude. Options: train a dedicated NLI/fake-news classifier on labeled data, use a pre-trained multilingual NLI model, or repurpose a multilingual sentiment model as a proxy.

## Decision

Use `cardiffnlp/twitter-xlm-roberta-base-sentiment` — a multilingual Twitter sentiment classifier — as a fake/real proxy. Mapping: negative sentiment index → "fake", positive sentiment index → "real". This signal is passed to Claude as context, not used as the final verdict.

```python
# Sentiment proxy mapping (verdict.py)
if probs[0] > probs[2]:      # negative > positive
    return "fake", round(probs[0], 4)
else:
    return "real", round(probs[2], 4)
```

## Alternatives Considered

**Train a dedicated fake-news classifier:** Requires labeled bilingual training data that doesn't exist for this use case. Scope exceeds a single course project. Rejected.

**Use a pre-trained multilingual NLI model (e.g., `facebook/bart-large-mnli`):** Better alignment to the task but English-dominant. Weaker Spanish performance. Rejected.

**Skip classifier entirely:** Loses the lightweight signal that helps Claude calibrate confidence. Rejected — ablation study will measure the delta.

**Use a dedicated Spanish fact-check model:** Narrows language scope; Spanish-only models don't generalize to English. Rejected.

## Consequences

**Positive:**
- Fully bilingual (trained on multilingual Twitter data)
- Lightweight — runs on CPU, ~200ms inference
- Provides a calibration signal for Claude — acknowledged in prompt as "XLM-RoBERTa signal"
- Graceful fallback: if the model fails to load, verdict proceeds with `("unknown", 0.0)`

**Negative:**
- Sentiment ≠ veracity — this is a proxy, not a fact-checking classifier. A sensationalist true headline scores "fake"; a calm false claim scores "real". This limitation is documented in the evaluation and acknowledged in the course submission.
- The model was trained on tweets, not fact-checking corpora — domain mismatch
- Ablation study will likely show limited contribution to final verdict quality

**Academic note:** This decision is intentionally documented as a methodological limitation for the IE7374 course submission. The ablation condition `no_classifier` exists precisely to quantify this.
