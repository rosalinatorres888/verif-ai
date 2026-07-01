# VerifAI — System Architecture
Last updated: 2026-05-30

---

## Architecture Pattern

Two-service system (FastAPI backend + Streamlit frontend). Single endpoint (`POST /verify`). Stateless — no DB writes at request time, no session storage. All state lives in ChromaDB (pre-built corpus, read-only at runtime).

---

## Request Lifecycle

```
User submits claim via Streamlit UI
        ↓
POST /verify {"claim": "..."} [optional: ?no_rag=true for ablation]
        ↓
[Layer 1 — intake.py]
  1a. langdetect.detect(claim) → "en" | "es" (default "en" on failure)
  1b. Claude Sonnet → CLAIM_EXTRACTION_PROMPT → bare falsifiable assertion
      Fallback: use raw claim if Claude API fails
        ↓
[Layer 2 — retrieval.py]
  2a. paraphrase-multilingual-MiniLM-L12-v2.encode(assertion) → embedding
  2b. ChromaDB "verif-ai-corpus".query(embedding, n_results=5)
      Filter: similarity ≥ 0.65 (converted from L2 distance: 1 - dist/2)
  2c. If corpus results < 3 → Tavily.search(assertion, max_results=5)
      Spanish claims: query prefixed with "fact check ... verificación"
  2d. Re-rank all evidence by (similarity_score × credibility_score) descending
  2e. Return top 5
        ↓
[Layer 3 — verdict.py]
  3a. XLM-RoBERTa (cardiffnlp/twitter-xlm-roberta-base-sentiment)
      → softmax logits → "fake" (negative) | "real" (positive) + confidence
      Fallback: ("unknown", 0.0) if model unavailable
  3b. Claude Sonnet → VERDICT_PROMPT with assertion + evidence + classifier signal
      → JSON: {label, confidence, explanation, key_evidence}
      explanation MUST be in input language
      label: "true" | "false" | "misleading" | "unverifiable"
        ↓
Response JSON:
{
  "claim_id": uuid,
  "original_claim": str,
  "language": "en"|"es",
  "label": str,
  "confidence": float,
  "classifier_label": str,
  "classifier_confidence": float,
  "explanation": str,         ← in input language
  "key_evidence": [str],      ← source names only, no fabrication
  "evidence": [...],          ← full evidence dicts
  "retrieval_method": str,
  "timestamp": ISO8601
}
```

---

## Data Architecture

### ChromaDB Corpus

- **Path:** `corpus/chroma_db/` (relative to repo root)
- **Collection:** `"verif-ai-corpus"`
- **Built by:** `corpus/build_corpus.py` (run on OOD once, then persisted)
- **Sources:** curated fact-check outlets defined in `corpus/sources.json`
- **Runtime access:** read-only — retrieval layer never writes

### Credibility Scoring

Two-tier system:
1. `app/utils/credibility.py` — static map, hardcoded scores per outlet
2. `corpus/sources.json` — source list used at corpus build time + credibility fallback in retrieval

Known sources and scores:
```
Reuters Fact Check: 0.95    AP Fact Check:          0.95
WHO:                0.97    CDC:                    0.97
PolitiFact:         0.90    Snopes:                 0.88
AFP Factuel:        0.92    La Vanguardia Verificat: 0.88
Default (unknown):  0.70    Tavily web results:     0.70
```

### No Persistence at Runtime

No DB writes during a verify request. No session state. No application DB. ChromaDB is pre-built and read-only during serving.

---

## Model Inventory

| Model | Purpose | Loaded by |
|---|---|---|
| `claude-sonnet-4-20250514` | Claim extraction (Layer 1) + Verdict (Layer 3) | Anthropic SDK, requires `ANTHROPIC_API_KEY` |
| `paraphrase-multilingual-MiniLM-L12-v2` | Bilingual embedding for retrieval (Layer 2) | sentence-transformers, local |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | First-pass fake/real signal (Layer 3) | transformers, downloaded on first run |

All local models are lazy-loaded and cached in module globals — loaded on first request.

---

## Evaluation Framework

All evaluation runs on **OOD (Northeastern GPU cluster)**, not locally.

| Script | What it measures |
|---|---|
| `evaluation/benchmark_liar.py` | Accuracy on LIAR benchmark dataset |
| `evaluation/ragas_eval.py` | Retrieval quality (RAGAS metrics: faithfulness, answer relevancy, context precision) |
| `evaluation/ablation.py` | 4-condition ablation: full / no-RAG / no-Tavily / no-classifier |
| `evaluation/human_eval_template.csv` | Template for manual human evaluation of verdict quality |

---

## Failure Modes and Fallbacks

| Failure | Fallback |
|---|---|
| Claude API error in Layer 1 | Use raw claim as extracted assertion |
| ChromaDB returns 0 results | Tavily only |
| Tavily error | Return whatever corpus found (may be empty) |
| XLM-RoBERTa load fails | `("unknown", 0.0)` — verdict proceeds without classifier signal |
| Claude returns invalid JSON in Layer 3 | Raises `ValueError` → 500 response |

The only hard failure (no fallback) is Claude returning malformed JSON in the verdict step.
