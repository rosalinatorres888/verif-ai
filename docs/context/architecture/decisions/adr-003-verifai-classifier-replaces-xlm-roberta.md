# ADR-003: From-Scratch VerifAIClassifier Replaces XLM-RoBERTa Proxy

Status: Accepted | Date: 2026-06-16 (Block G) | Supersedes: [ADR-001](adr-001-xlm-roberta-as-proxy.md)

## Context

ADR-001 accepted `cardiffnlp/twitter-xlm-roberta-base-sentiment` as a stopgap fake/real
signal because no trained fact-checking classifier existed yet. That was always framed
as a proxy, not a destination — ADR-001 explicitly names "train a dedicated fake-news
classifier" as an alternative, rejected only because "labeled bilingual training data
that doesn't exist for this use case" at the time.

By Block D, that data existed: LIAR + MultiFC + FakeDeS + a synthetic ES corpus, combined
into a 37,608-row balanced training set (`training/prepare_data.py`). Blocks B–E trained
and evaluated a purpose-built classifier (`model/architecture.py`) on exactly this data,
producing a working checkpoint (`models/verifai-classifier/best_model.pt`, val F1=0.4049,
test F1=0.3647). The rejected alternative from ADR-001 became the accepted decision here.

## Decision

Replace the XLM-RoBERTa sentiment proxy with **VerifAIClassifier**, a 6.3M-parameter
transformer trained from scratch (BPE tokenizer, language embeddings for EN/ES, 4
transformer encoder blocks, 4-class head), in two roles:

1. **Verdict signal** (`app/pipeline/verdict.py`, `run_classifier()`) — predicts one of
   true / false / misleading / unverifiable directly, passed to Claude as
   `VerifAI classifier signal: {label} (confidence: {confidence})` instead of the old
   sentiment-derived fake/real proxy.
2. **Evidence reranker** (`app/pipeline/reranker.py`, Block F) — scores each
   (claim, passage) pair by encoding them as a pair (`[CLS] claim [SEP] passage`) and
   using `P(true) + P(false)` from the same checkpoint as a relevance signal: high
   combined probability means the model sees the passage as clearly supporting or
   refuting the claim, i.e. relevant, versus noise the model can't classify either way.
   This score feeds `combined_score = cosine_similarity × credibility_score × reranker_score`
   in the retrieval layer.

Both roles load the same `best_model.pt` checkpoint — one trained artifact, two uses.
This dual-use is the project's specific architectural contribution called out in
Section 1 of the M2 proposal and `docs/literature-review.md`.

## Alternatives Considered

**Keep XLM-RoBERTa as a permanent signal:** Rejected — ADR-001 always scoped it as
temporary pending labeled data; sentiment is not veracity (a calm false claim and a
sensationalist true one are conflated), and this was documented there as an acknowledged
limitation, not a design goal.

**Fine-tune a pretrained multilingual model (e.g., XLM-R, mBERT) on the combined corpus
instead of training from scratch:** Would likely reach a higher F1 faster. Rejected for
this project specifically because full engineering ownership of the architecture and
training loop is a stated learning objective (see M2 proposal, Section 1) — the honest
from-scratch baseline is the point, not just the classifier output.

**Use two separate models for verdict and reranking:** Simpler to reason about
individually, but doubles inference cost and checkpoint size for a course-project-scale
deployment, and the dual-use design is itself the thing being evaluated (ablation
condition `no_classifier` in `evaluation/ablation.py` measures its combined contribution,
not each role separately).

## Consequences

**Positive:**
- Verdict signal is now a real veracity prediction, not a sentiment proxy — closes the
  main limitation ADR-001 flagged.
- One checkpoint serves two pipeline stages, keeping the system lightweight.
- Directly measurable: Block E's test F1=0.3647 replaces ADR-001's unmeasured
  "acknowledged limitation."

**Negative:**
- Test F1=0.3647 is well above the 0.25 random baseline for a balanced 4-class task but
  still modest in absolute terms — the train/val F1 gap (0.5728 train acc by epoch 12 vs.
  val F1 peaking at 0.4049, epoch 15) documents data scarcity as the primary constraint,
  not an architecture flaw. This is reported as an honest finding in the M2 proposal
  rather than hidden.
- Per-class F1 is uneven (true=0.55, false=0.38, misleading=0.28, unverifiable=0.24,
  per `outputs/classifier_results.json`) — the reranker's `P(true)+P(false)` signal
  inherits this weakness on the "misleading" and "unverifiable" classes specifically.
- ADR-001's XLM-RoBERTa code path is fully removed from `verdict.py`, not kept as a
  fallback — if `best_model.pt` fails to load, the system now degrades to
  `("unknown", 0.0)` rather than falling back to the sentiment proxy.
