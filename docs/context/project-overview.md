# VerifAI — Context Overview
Owner: Rosalina Torres | Last updated: 2026-05-30
Course: IE7374 — Generative AI · Northeastern University · Solo project · Summer 2026
Location: `~/verif-ai/`

---

## Quick Navigation for AI

| Task | Go to |
|---|---|
| Full system architecture and pipeline | `docs/context/architecture/system-design.md` |
| Why specific technical decisions were made | `docs/context/architecture/decisions/` |
| Working with the 3-layer pipeline | `docs/context/components/pipeline.md` |

---

## What VerifAI Is

A bilingual (English/Spanish) retrieval-augmented misinformation detection system. A user submits a claim in either language → a 3-layer pipeline detects the language, retrieves evidence from a curated corpus + live web, runs a multilingual NLI classifier as a first-pass signal, then calls Claude Sonnet to synthesize a structured verdict with explanation in the input language.

**Status:** Course project — active development (Summer 2026 semester). Not a production system. Evaluated against the LIAR benchmark and via RAGAS retrieval quality metrics.

---

## Services

| Service | Port | Entry Point | Purpose |
|---|---|---|---|
| FastAPI backend | :8000 | `app/main.py` | Single endpoint: `POST /verify` |
| Streamlit frontend | :8502 | `frontend/streamlit_app.py` | User-facing claim submission UI |

**Start:**
```bash
source venv/bin/activate
uvicorn app.main:app --reload --port 8000   # backend
streamlit run frontend/streamlit_app.py --server.port 8502  # frontend
```

---

## The 3-Layer Pipeline

```
User claim (English or Spanish)
        ↓
[Layer 1 — Intake]       app/pipeline/intake.py
  langdetect → language detection ("en" | "es", defaults to "en")
  Claude Sonnet → extract core falsifiable assertion (strips opinion/hedging)
        ↓
[Layer 2 — Retrieval]    app/pipeline/retrieval.py
  paraphrase-multilingual-MiniLM-L12-v2 → embed assertion
  ChromaDB "verif-ai-corpus" → query top-5 by similarity (threshold 0.65)
  Tavily API → supplement if corpus returns < 3 results
  Re-rank: similarity_score × credibility_score (descending)
        ↓
[Layer 3 — Verdict]      app/pipeline/verdict.py
  XLM-RoBERTa (cardiffnlp/twitter-xlm-roberta-base-sentiment) → first-pass signal
  Claude Sonnet → structured JSON verdict in input language
        ↓
Response: {label, confidence, explanation, key_evidence, classifier_label, evidence[]}
```

**Verdict labels:** `"true"` | `"false"` | `"misleading"` | `"unverifiable"`

---

## Key Files

```
~/verif-ai/
├── app/
│   ├── main.py                  ← FastAPI app, POST /verify endpoint
│   ├── pipeline/
│   │   ├── intake.py            ← Layer 1: language detect + claim extraction
│   │   ├── retrieval.py         ← Layer 2: ChromaDB + Tavily + credibility rerank
│   │   └── verdict.py           ← Layer 3: XLM-RoBERTa + Claude verdict
│   └── utils/
│       ├── credibility.py       ← Static credibility score map by source name
│       └── language.py          ← Language utilities
├── corpus/
│   ├── build_corpus.py          ← Build ChromaDB from sources (run on OOD/GPU cluster)
│   └── sources.json             ← Curated fact-check source list with credibility scores
├── data/
│   └── sample_claims.json       ← Test claims (English + Spanish)
├── evaluation/
│   ├── benchmark_liar.py        ← LIAR benchmark evaluation (run on OOD)
│   ├── ragas_eval.py            ← RAGAS retrieval quality evaluation
│   ├── ablation.py              ← 4-condition ablation study
│   └── human_eval_template.csv  ← Human evaluation template
├── frontend/
│   └── streamlit_app.py         ← Streamlit UI
├── requirements.txt
├── .env                         ← API keys (never commit)
└── .env.template                ← Template — copy to .env and fill
```

---

## Required API Keys (`.env`)

```
ANTHROPIC_API_KEY=...   # Claude Sonnet — intake + verdict
TAVILY_API_KEY=...      # Web search supplementation in retrieval
```

---

## Hard Rules for AI Working Here

1. **`corpus/sources.json` is read-only at runtime.** The retrieval layer explicitly never writes to it. Only `build_corpus.py` reads it to build the ChromaDB index.
2. **Corpus build runs on OOD (Northeastern's GPU cluster)**, not locally. Do not try to run `corpus/build_corpus.py` on a laptop — it's compute-heavy.
3. **Evaluation scripts run on OOD** as well (`benchmark_liar.py`, `ragas_eval.py`, `ablation.py`). Local runs are for development only.
4. **Verdict must be in the input language.** The Claude prompt enforces this — `explanation` and language must match. Do not break this constraint.
5. **Claude must not fabricate sources.** The verdict prompt hard-rules: "Use only sources provided in Evidence above." Do not weaken this instruction.
6. **No auto-commit of `.env`.** API keys live in `.env`, which is gitignored. `.env.template` is the committed reference.
7. **PRD lives in mission-control**, not here: `~/Documents/mission-control/PRD-VerifAI.md`.

---

## AI Collaboration Notes

**Multilingual embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` — chosen specifically for English/Spanish bilingual support. Do not swap for `all-MiniLM-L6-v2` (English-only).

**ChromaDB distance conversion:** ChromaDB returns L2 distance. Retrieval layer converts to approximate cosine similarity: `similarity = max(0.0, 1.0 - dist / 2.0)`. Threshold: ≥ 0.65.

**XLM-RoBERTa as proxy classifier:** Model is a sentiment classifier repurposed as a fake/real signal (negative sentiment → "fake" proxy, positive → "real" proxy). This is intentional — a limitation acknowledged in the evaluation. It provides a lightweight first-pass signal, not a final verdict.

**Ablation study conditions:** 4-condition — full pipeline (corpus + Tavily + classifier + Claude), no RAG, no Tavily, no classifier. Ablation toggle exposed in the Streamlit UI via `no_rag` query param on the `/verify` endpoint.

**Lazy loading:** All heavy models (XLM-RoBERTa, SentenceTransformer, ChromaDB client, Tavily client) are loaded on first call and cached in module-level globals. FastAPI startup is fast; first request takes longer.
