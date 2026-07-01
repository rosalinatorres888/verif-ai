"""
corpus/build_corpus.py
Fetches content from sources in sources.json, chunks into 512-token passages,
embeds with paraphrase-multilingual-MiniLM-L12-v2, and indexes into ChromaDB.

Fetch strategy:
  1. Direct HTTP fetch via requests (no API key needed) — primary
  2. Tavily extract — fallback if direct fetch yields < 500 chars and TAVILY_API_KEY is set

Usage:
    python corpus/build_corpus.py
    python corpus/build_corpus.py --source src-003   # single source smoke test
"""
import os
import json
import argparse
import hashlib
import time
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

SOURCES_PATH = Path(__file__).parent / "sources.json"
CHROMA_PATH  = Path(__file__).parent / "chroma_db"
CHUNK_SIZE   = 512   # tokens (approx chars / 4)
CHUNK_OVERLAP = 64
MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; VerifAI-corpus-builder/1.0)"
}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    char_size    = chunk_size * 4
    char_overlap = overlap * 4
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + char_size].strip())
        start += char_size - char_overlap
    return [c for c in chunks if len(c) > 80]


def fetch_direct(url: str) -> str:
    """Fetch plain text from a URL via requests. Returns empty string on failure."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        # Strip obvious HTML tags for cleaner text
        text = r.text
        import re
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s{3,}", "\n\n", text)
        return text.strip()
    except Exception as e:
        print(f"    Direct fetch error: {e}")
        return ""


def fetch_tavily(url: str) -> str:
    """Fallback fetch via Tavily extract API."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key or api_key.startswith("tvly-dev-") or "YOUR_KEY" in api_key:
        return ""
    try:
        from tavily import TavilyClient
        t = TavilyClient(api_key=api_key)
        resp = t.extract(urls=[url])
        results = resp.get("results", [])
        return results[0].get("raw_content", "") if results else ""
    except Exception as e:
        print(f"    Tavily fallback error: {e}")
        return ""


def fetch_source_content(source: dict) -> list:
    """Returns list of text strings for a source. Tries direct then Tavily."""
    print(f"  Fetching: {source['name']} ({source['url']})")
    text = fetch_direct(source["url"])
    if len(text) < 500:
        print(f"    Direct fetch short ({len(text)} chars) — trying Tavily fallback...")
        text = fetch_tavily(source["url"])
    if not text:
        print(f"    No content retrieved.")
        return []
    print(f"    Got {len(text):,} chars")
    return [text]


def build_corpus(source_filter: str = None):
    with open(SOURCES_PATH) as f:
        sources = json.load(f)

    if source_filter:
        sources = [s for s in sources if s["id"] == source_filter]
        if not sources:
            print(f"Source '{source_filter}' not found in sources.json")
            return

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH.resolve()))
    collection = chroma.get_or_create_collection("verif-ai-corpus")

    total_chunks = 0
    for source in sources:
        print(f"\nProcessing: {source['name']}")
        texts = fetch_source_content(source)
        if not texts:
            continue

        chunks = []
        for text in texts:
            chunks.extend(chunk_text(text))

        if not chunks:
            print(f"  No usable chunks after splitting.")
            continue
        print(f"  {len(chunks)} chunks — embedding...")

        embeddings = model.encode(chunks, show_progress_bar=True).tolist()
        ids = [
            hashlib.md5(f"{source['id']}-{i}-{c[:50]}".encode()).hexdigest()
            for i, c in enumerate(chunks)
        ]
        metadatas = [{
            "source_id":        source["id"],
            "source_name":      source["name"],
            "source_url":       source["url"],
            "language":         source["language"],
            "credibility_score": source["credibility_score"]
        } for _ in chunks]

        for i in range(0, len(chunks), 100):
            collection.upsert(
                documents=chunks[i:i+100],
                embeddings=embeddings[i:i+100],
                ids=ids[i:i+100],
                metadatas=metadatas[i:i+100]
            )
        total_chunks += len(chunks)
        print(f"  Indexed {len(chunks)} chunks for {source['name']}")
        time.sleep(1)   # polite crawl delay

    print(f"\nCorpus build complete. Total chunks indexed: {total_chunks}")
    print(f"ChromaDB path: {CHROMA_PATH.resolve()}")

    print("\nSmoke test — querying 'vaccines cause autism':")
    test_emb = model.encode("vaccines cause autism").tolist()
    results  = collection.query(query_embeddings=[test_emb], n_results=3)
    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        print("  No results — corpus may be empty.")
    for doc, meta in zip(docs, metas):
        print(f"  [{meta['source_name']}] {doc[:120]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Single source ID, e.g. --source src-003")
    args = parser.parse_args()
    build_corpus(source_filter=args.source)
