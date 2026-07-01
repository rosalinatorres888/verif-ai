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
2. **Retrieval** (`app/pipeline/retrieval.py`) — ChromaDB + Tavily + credibility rerank
3. **Verdict** (`app/pipeline/verdict.py`) — XLM-RoBERTa + Claude API verdict generation

Frontend: Streamlit UI at localhost:8502  
Backend: FastAPI at localhost:8000

## Evaluation

```bash
python evaluation/benchmark_liar.py   # LIAR benchmark (run on OOD)
python evaluation/ragas_eval.py       # RAGAS retrieval quality
python evaluation/ablation.py         # 4-condition ablation study
```

## PRD

Full build spec: `~/Documents/mission-control/PRD-VerifAI.md`
