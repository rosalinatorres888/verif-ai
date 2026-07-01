# 3-Layer Pipeline — Context

Last updated: 2026-05-30

---

## Layer 1 — Intake (`app/pipeline/intake.py`)

**Purpose:** Language detection + claim normalization.

### Language Detection

```python
from langdetect import detect
lang = detect(text)
return lang if lang in ("en", "es") else "en"   # default "en" on failure
```

Supports only English and Spanish. Any other language detected defaults to English.

### Claim Extraction

Claude Sonnet call with `CLAIM_EXTRACTION_PROMPT` — strips opinion, emotion, hedging, context. Returns the bare falsifiable assertion as a single sentence.

**Fallback:** If the Claude API call fails for any reason, the raw claim is used as-is. The system never blocks on Layer 1 failure.

**Output:** `{"language": "en"|"es", "extracted_assertion": str, "original_claim": str}`

---

## Layer 2 — Retrieval (`app/pipeline/retrieval.py`)

**Purpose:** Find relevant evidence from corpus + live web, re-ranked by quality.

### Constants (change requires testing)

```python
SIMILARITY_THRESHOLD = 0.65   # minimum similarity to include corpus result
TOP_K = 5                     # max evidence items returned
MIN_CORPUS_RESULTS = 3        # trigger Tavily if corpus returns fewer than this
```

### Embedding Model

`paraphrase-multilingual-MiniLM-L12-v2` — bilingual English/Spanish. Loaded once and cached. **Do not swap for `all-MiniLM-L6-v2`** — that model is English-only.

### ChromaDB Query

```python
collection.query(query_embeddings=[embedding], n_results=TOP_K,
                 include=["documents", "metadatas", "distances"])
# ChromaDB returns L2 distance → convert: similarity = max(0, 1 - dist/2)
```

Collection name: `"verif-ai-corpus"`. Path: `corpus/chroma_db/` (relative to repo root).

**Runtime constraint:** read-only. Never writes to ChromaDB or `corpus/sources.json` at runtime.

### Credibility Re-ranking

```python
evidence.sort(key=lambda x: x["similarity_score"] * x["credibility_score"], reverse=True)
```

This is the core quality signal — a highly similar but low-credibility source ranks below a moderately similar high-credibility source.

### Tavily Supplement

Triggered when `len(corpus_evidence) < MIN_CORPUS_RESULTS`. Spanish claims get bilingual query. All Tavily results assigned `credibility_score=0.70` (default).

**Graceful failure:** Tavily errors are caught and logged — pipeline continues with whatever corpus evidence exists.

---

## Layer 3 — Verdict (`app/pipeline/verdict.py`)

**Purpose:** First-pass ML signal + Claude structured verdict.

### XLM-RoBERTa Signal

Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (downloaded on first run, then cached by HuggingFace).

**Proxy mapping (intentional limitation):**
- Negative sentiment (index 0) → "fake"
- Positive sentiment (index 2) → "real"

Passed to Claude as `classifier_label` + `classifier_confidence`. Not used as the final verdict. Fallback: `("unknown", 0.0)` if model fails.

### Claude Verdict

`VERDICT_PROMPT` enforces:
- Response in input language (`English` or `Spanish`)
- JSON only — no markdown, no explanation outside JSON
- No fabricated sources — `key_evidence` must reference only sources in the provided evidence
- `"unverifiable"` label if evidence is insufficient

**Post-processing:** Claude's response has markdown fences stripped via regex before JSON parsing. If `json.loads()` fails → `ValueError` → HTTP 500 (no graceful fallback at this step).

### Verdict Schema

```json
{
  "label": "true" | "false" | "misleading" | "unverifiable",
  "confidence": 0.0–1.0,
  "explanation": "3–5 sentences in input language",
  "key_evidence": ["Source Name 1", "Source Name 2"]
}
```

---

## Ablation Study

4 conditions exposed via `no_rag=true` query parameter on `/verify`:

| Condition | Retrieval | Classifier | Description |
|---|---|---|---|
| Full pipeline | ✅ | ✅ | Baseline |
| No RAG | ❌ | ✅ | Evidence = [], classifier still runs |
| No Tavily | corpus only | ✅ | `MIN_CORPUS_RESULTS` effectively ∞ |
| No classifier | ✅ | ❌ | `("unknown", 0.0)` injected |

`evaluation/ablation.py` runs all 4 conditions and compares verdict quality.

---

## Constraints for AI

- **Never write to ChromaDB or `corpus/sources.json` at runtime** — read-only
- **Never change the bilingual embedding model** to an English-only alternative
- **`explanation` must be in input language** — Claude prompt enforces this; do not weaken the instruction
- **`key_evidence` must use only provided source names** — no fabrication allowed; this is a hard rule in the verdict prompt
- **Corpus rebuild runs on OOD** — do not attempt locally; `corpus/build_corpus.py` is compute-heavy
- **Both services must be running for the UI to work:** FastAPI on :8000 AND Streamlit on :8502
