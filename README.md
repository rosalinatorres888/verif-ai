# VerifAI — Bilingual Fact Checker

A bilingual (English/Spanish) retrieval-augmented misinformation detection system.

**Course:** IE7374 — Generative AI · Northeastern University  
**Author:** Rosalina Torres · Solo Project · Summer 2026

---

## Quick Start

```bash
# 1. Create venv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Copy .env.template → .env and fill in your API keys
cp .env.template .env

# 3. Build corpus (run on OOD for large indexing)
python corpus/build_corpus.py

# 4. Start FastAPI backend (port 8000)
uvicorn app.main:app --reload --port 8000

# 5. Start Streamlit UI (separate terminal)
streamlit run frontend/streamlit_app.py --server.port 8502
```

## Architecture

Three-layer pipeline:
1. **Intake** (`app/pipeline/intake.py`) — language detection + claim extraction
2. **Retrieval** (`app/pipeline/retrieval.py`) — ChromaDB + Tavily + credibility rerank,
   using `app/pipeline/reranker.py` to score evidence relevance
3. **Verdict** (`app/pipeline/verdict.py`) — VerifAIClassifier + Claude API verdict generation

Frontend: Streamlit UI at localhost:8502
Backend: FastAPI at localhost:8000

## Model

**VerifAIClassifier** (`model/architecture.py`) is a 6.3M-parameter transformer trained
from scratch in PyTorch — no pretrained weights. Custom BPE tokenizer (16,000 tokens,
shared EN/ES vocabulary), language embeddings (EN=0, ES=1), 4 transformer encoder blocks,
4-class head (true / false / misleading / unverifiable).

It plays a dual role: a first-pass verdict signal in `verdict.py`, and a RAG evidence
reranker in `reranker.py` (`combined_score = cosine_similarity × credibility_score ×
reranker_score`). Both load the same checkpoint. See
[ADR-003](docs/context/architecture/decisions/adr-003-verifai-classifier-replaces-xlm-roberta.md)
for why this replaced an earlier XLM-RoBERTa sentiment proxy.

Trained on 37,608 examples (LIAR + MultiFC + FakeDeS + a synthetic ES corpus). Current
result: val F1=0.4049, test F1=0.3647 on the held-out LIAR test split (1,283 claims,
`outputs/classifier_results.json`, per-class breakdown and confusion matrix in `outputs/`).

```bash
# Rebuild the tokenizer after changing training data
python training/build_tokenizer.py

# Train (writes models/verifai-classifier/best_model.pt)
python training/train_classifier.py

# Evaluate the checkpoint against data/test.csv
python training/evaluate_classifier.py
```

## Evaluation

```bash
python evaluation/benchmark_liar.py   # LIAR benchmark (run on OOD)
python evaluation/ragas_eval.py       # RAGAS retrieval quality
python evaluation/ablation.py         # 4-condition ablation study
```

## Documentation

- `docs/literature-review.md` — literature review grounding the project's design choices
- `docs/context/architecture/system-design.md` — full pipeline architecture
- `docs/context/architecture/decisions/` — ADRs (design decisions and why)
- `docs/context/project-overview.md` — quick orientation for contributors

## PRD

Full build spec (personal planning doc, not required to reproduce this repo):
`~/Documents/mission-control/PRD-VerifAI.md`
