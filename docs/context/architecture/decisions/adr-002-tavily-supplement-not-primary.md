# ADR-002: Tavily as Corpus Supplement, Not Primary Source

Status: Accepted | Date: 2026-05-30

## Context

Evidence retrieval needs both depth (curated, credibility-scored corpus) and breadth (live web for claims not in the corpus). Two approaches: use live web search as the primary retrieval source, or maintain a curated corpus as primary and supplement with live web only when the corpus is insufficient.

## Decision

ChromaDB curated corpus is primary. Tavily live web search supplements only when the corpus returns fewer than 3 results (`MIN_CORPUS_RESULTS = 3`). Tavily results receive a default credibility score of 0.70 (same as unknown sources in the corpus).

```python
# retrieval.py
if len(evidence) < MIN_CORPUS_RESULTS:
    # Supplement with Tavily
    response = tavily.search(query=search_query, max_results=5, search_depth="advanced")
    # Tavily results appended with credibility_score=0.70
```

Spanish claims get a bilingual Tavily query: `f"fact check {assertion} verificación"`.

## Alternatives Considered

**Tavily as primary, corpus as supplement:** Live web has no credibility control — any website can be returned. Reuters Fact Check and a random blog would be treated equally. Unacceptable for a fact-checking system. Rejected.

**Corpus only, no Tavily:** Many claims (especially recent ones) won't be in a pre-built corpus. Retrieval would fail silently and force "unverifiable" verdicts on newsworthy claims. Rejected.

**Equal weight between corpus and Tavily:** Dilutes credibility scoring — corpus results have verified scores, Tavily results don't. Re-ranking by similarity × credibility would disadvantage high-credibility corpus sources that score slightly below high-similarity low-credibility Tavily results. Rejected.

## Consequences

**Positive:**
- Corpus quality is maintained — Reuters/AP/WHO results always preferred over web scrapes
- Credibility re-ranking (similarity × credibility) means a high-credibility corpus source beats a higher-similarity Tavily result
- Graceful degradation: if Tavily fails, verdict proceeds with available corpus evidence
- Spanish queries explicitly optimized for bilingual web retrieval

**Negative:**
- Corpus must be pre-built and kept reasonably current — stale corpus on recent claims forces Tavily fallback
- Tavily credits are consumed per supplemented query (cost consideration for high-volume use)
- 0.70 default credibility for Tavily results may be too generous for low-quality web sources

**Constraint:** `corpus/sources.json` is read-only at runtime. New sources are added by editing `sources.json` and rebuilding the corpus on OOD — never at runtime.
