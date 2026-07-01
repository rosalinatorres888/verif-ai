"""
Layer 2 — Evidence Retrieval (RAG)
Embeds the assertion, queries ChromaDB, supplements with Tavily if needed,
and re-ranks by (similarity_score × credibility_score).

CRITICAL: never writes to corpus/sources.json — read only.

Fixes applied (Jun 16 2026):
  Fix 1 — Spanish Tavily query uses dedicated ES fact-check domains + Spanish terms
  Fix 2 — Tavily results filtered by minimum relevance threshold (0.60)
           to prevent noisy/irrelevant results from polluting evidence panel
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from tavily import TavilyClient
from app.pipeline.reranker import score_batch

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)

CHROMA_PATH          = os.path.join(os.path.dirname(__file__), "../../corpus/chroma_db")
SOURCES_PATH         = os.path.join(os.path.dirname(__file__), "../../corpus/sources.json")
SIMILARITY_THRESHOLD = 0.65   # ChromaDB minimum
TAVILY_MIN_SCORE     = 0.60   # Fix 2: minimum Tavily relevance score
TOP_K                = 5
MIN_CORPUS_RESULTS   = 3

# Trusted Spanish-language fact-check domains for Fix 1
ES_FACT_CHECK_DOMAINS = [
    "maldita.es",
    "newtral.es",
    "chequeado.com",
    "factchequeado.com",
    "colombiacheck.com",
    "verificat.cat",
    "afpfactual.afp.com",
    "animalpolitico.com",
    "pagina12.com.ar",
    "elpais.com",
    "bbc.com/mundo",
]

_model        = None
_chroma_client = None
_collection   = None
_tavily       = None
_credibility_map = None


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def _load_chroma():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=os.path.abspath(CHROMA_PATH))
        _collection = _chroma_client.get_or_create_collection("verif-ai-corpus")
    return _collection


def _load_credibility():
    global _credibility_map
    if _credibility_map is None:
        with open(os.path.abspath(SOURCES_PATH), "r") as f:
            sources = json.load(f)
        _credibility_map = {s["name"]: s["credibility_score"] for s in sources}
    return _credibility_map


def _load_tavily():
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _tavily


def _build_tavily_query(assertion: str, language: str) -> dict:
    """
    Fix 1 — Build a language-aware Tavily query.

    For Spanish claims:
      - Query in Spanish with fact-check terminology
      - Restrict to trusted Spanish-language fact-check domains
      - Use two-pass: domain-restricted first, open web fallback

    For English claims:
      - Standard fact-check query
    """
    if language == "es":
        # Primary: Spanish fact-check terminology + domain restriction
        query = f"verificación de hechos: {assertion}"
        return {
            "query": query,
            "search_depth": "advanced",
            "max_results": 7,
            "include_domains": ES_FACT_CHECK_DOMAINS,
        }
    else:
        # English: standard fact-check query
        return {
            "query": f"fact check: {assertion}",
            "search_depth": "advanced",
            "max_results": 5,
        }


def _tavily_search_with_fallback(tavily: TavilyClient, assertion: str, language: str) -> list:
    """
    Fix 1 + Fix 2 — Spanish-aware Tavily search with quality fallback.

    Strategy:
      1. Try domain-restricted Spanish search (high precision)
      2. If < 2 results, fall back to open Spanish web search
      3. Filter all results by TAVILY_MIN_SCORE (Fix 2)
    """
    params = _build_tavily_query(assertion, language)
    raw_results = []

    try:
        response = tavily.search(**params)
        raw_results = response.get("results", [])
    except Exception as e:
        print(f"[retrieval] Tavily primary search error: {e}")

    # Fix 1 fallback — if domain-restricted ES search returns too few results,
    # try open Spanish web search
    if language == "es" and len(raw_results) < 2:
        try:
            fallback_response = tavily.search(
                query=f"verificación: {assertion} es falso o verdadero",
                search_depth="advanced",
                max_results=5
            )
            fallback_results = fallback_response.get("results", [])
            # Merge, deduplicate by URL
            seen_urls = {r.get("url", "") for r in raw_results}
            for r in fallback_results:
                if r.get("url", "") not in seen_urls:
                    raw_results.append(r)
                    seen_urls.add(r.get("url", ""))
            print(f"[retrieval] ES fallback search added {len(fallback_results)} results")
        except Exception as e:
            print(f"[retrieval] Tavily ES fallback error: {e}")

    # Fix 2 — filter by minimum relevance score
    filtered = []
    for r in raw_results:
        score = r.get("score", 0.0)
        if score >= TAVILY_MIN_SCORE:
            filtered.append(r)
        else:
            print(f"[retrieval] Dropped low-quality result (score={score:.2f}): "
                  f"{r.get('url', '')[:60]}")

    if not filtered and raw_results:
        # If all results filtered out, keep the best one rather than returning nothing
        best = max(raw_results, key=lambda x: x.get("score", 0.0))
        filtered = [best]
        print(f"[retrieval] All results below threshold — keeping best "
              f"(score={best.get('score', 0.0):.2f})")

    return filtered


def retrieve_evidence(extracted_assertion: str, language: str) -> list:
    """
    Returns list of evidence dicts:
    [{source_name, source_url, passage, credibility_score, similarity_score}]
    Sorted descending by (similarity_score * credibility_score).

    Fix 1: Spanish queries use dedicated ES fact-check domains.
    Fix 2: Tavily results filtered by minimum relevance threshold.
    """
    model          = _load_model()
    collection     = _load_chroma()
    credibility_map = _load_credibility()

    query_embedding = model.encode(extracted_assertion).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    evidence = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = max(0.0, 1.0 - dist / 2.0)
            if similarity >= SIMILARITY_THRESHOLD:
                source_name = meta.get("source_name", "Unknown")
                evidence.append({
                    "source_name": source_name,
                    "source_url":  meta.get("source_url", ""),
                    "passage":     doc,
                    "credibility_score": credibility_map.get(source_name, 0.70),
                    "similarity_score":  round(similarity, 4),
                    "retrieval_method":  "corpus"
                })

    # Supplement with Tavily if corpus returns fewer than MIN_CORPUS_RESULTS
    if len(evidence) < MIN_CORPUS_RESULTS:
        try:
            tavily  = _load_tavily()
            tavily_results = _tavily_search_with_fallback(tavily, extracted_assertion, language)
            for r in tavily_results:
                evidence.append({
                    "source_name":       r.get("title", "Web Source"),
                    "source_url":        r.get("url", ""),
                    "passage":           r.get("content", ""),
                    "credibility_score": 0.70,
                    "similarity_score":  round(r.get("score", 0.60), 4),
                    "retrieval_method":  "tavily"
                })
        except Exception as e:
            print(f"[retrieval] Tavily error: {e}")

    # Re-rank using combined score: cosine × credibility × reranker
    if evidence:
        passages = [e["passage"] for e in evidence]
        reranker_scores = score_batch(extracted_assertion, passages, language)
        for e, rs in zip(evidence, reranker_scores):
            e["reranker_score"] = rs
            e["combined_score"] = round(
                e["similarity_score"] * e["credibility_score"] * rs, 4
            )
    else:
        for e in evidence:
            e["reranker_score"] = 0.5
            e["combined_score"] = round(
                e["similarity_score"] * e["credibility_score"], 4
            )

    evidence.sort(key=lambda x: x["combined_score"], reverse=True)
    return evidence[:TOP_K]
