"""
corpus/build_static_corpus.py

Build a substantially larger, INDEPENDENT, STATIC evidence corpus for VerifAI
retrieval, replacing the 43-passage v1 corpus that could not ground the claims.

Design choices (recorded in corpus/corpus_provenance.json):
  - Source: Wikipedia, a fixed dump snapshot (default 20231101), en + es.
    Independent of the LIAR benchmark → no train/eval contamination.
  - Embeddings are L2-NORMALIZED at build time. Combined with the normalized
    query embedding in retrieval.py, Chroma's (squared) L2 distance satisfies
    dist = 2*(1 - cosine), so retrieval's `similarity = 1 - dist/2` recovers
    true cosine. This fixes the v1 metric bug (unnormalized vectors → sim=0).
  - Written to a versioned path/collection (chroma_db_v2 / verif-ai-corpus-v2);
    v1 is left untouched for rollback.

Run (uses the working .venv313 interpreter):
    HF_HOME=<scratch> ./.venv313/bin/python corpus/build_static_corpus.py \
        --en-articles 6000 --es-articles 2000 --chunks-per-article 4

Then re-run the retrieval smoke test before building evidence CSVs.
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = str(ROOT / "corpus" / "chroma_db_v2")
PROVENANCE = ROOT / "corpus" / "corpus_provenance.json"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # matches retrieval.py


def chunk_article(text: str, chunk_chars: int, min_chars: int, max_chunks: int) -> list[str]:
    """Pack paragraphs into ~chunk_chars passages; keep up to max_chunks per article."""
    chunks, buf = [], ""
    for para in (p.strip() for p in text.split("\n") if p.strip()):
        if len(para) < 40:  # skip headers / boilerplate lines
            continue
        buf = f"{buf} {para}".strip() if buf else para
        if len(buf) >= chunk_chars:
            chunks.append(buf[:chunk_chars * 2])
            buf = ""
            if len(chunks) >= max_chunks:
                return chunks
    if buf and len(buf) >= min_chars and len(chunks) < max_chunks:
        chunks.append(buf)
    return chunks


def stream_passages(lang: str, dump: str, n_articles: int, chunk_chars: int,
                    min_chars: int, chunks_per_article: int):
    """Yield (passage, title, url) from the first n_articles of a wiki dump."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", f"{dump}.{lang}", split="train", streaming=True)
    seen = 0
    for row in ds:
        if seen >= n_articles:
            break
        seen += 1
        for passage in chunk_article(row["text"], chunk_chars, min_chars, chunks_per_article):
            yield passage, row["title"], row.get("url", "")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description="Build the v2 static Wikipedia evidence corpus.")
    p.add_argument("--en-articles", type=int, default=6000)
    p.add_argument("--es-articles", type=int, default=2000)
    p.add_argument("--chunks-per-article", type=int, default=4)
    p.add_argument("--chunk-chars", type=int, default=500)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--dump", default="20231101")
    p.add_argument("--out-path", default=DEFAULT_OUT)
    p.add_argument("--collection", default="verif-ai-corpus-v2")
    p.add_argument("--embed-model", default=EMBED_MODEL)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    print(f"[corpus] embedding model: {args.embed_model}")
    model = SentenceTransformer(args.embed_model)

    print(f"[corpus] collecting passages (en={args.en_articles}, es={args.es_articles}, "
          f"dump={args.dump})...")
    passages, metas = [], []
    per_lang = {}
    for lang, n in (("en", args.en_articles), ("es", args.es_articles)):
        before = len(passages)
        for passage, title, url in stream_passages(
            lang, args.dump, n, args.chunk_chars, args.min_chars, args.chunks_per_article
        ):
            passages.append(passage)
            metas.append({"source_name": f"Wikipedia ({lang})", "source_url": url,
                          "title": title, "language": lang})
        per_lang[lang] = len(passages) - before
        print(f"  {lang}: {per_lang[lang]} passages")

    if not passages:
        print("[corpus] ERROR: no passages collected")
        return 1

    print(f"[corpus] embedding {len(passages):,} passages (normalized)...")
    embeddings = model.encode(
        passages, batch_size=args.batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    )

    # Fresh collection (delete any prior build so this is reproducible).
    os.makedirs(args.out_path, exist_ok=True)
    client = chromadb.PersistentClient(path=os.path.abspath(args.out_path))
    try:
        client.delete_collection(args.collection)
    except Exception:
        pass
    col = client.create_collection(args.collection, metadata={"hnsw:space": "l2"})

    print(f"[corpus] writing {len(passages):,} docs to '{args.collection}'...")
    B = 2000
    for i in range(0, len(passages), B):
        col.add(
            ids=[f"wiki-{j}" for j in range(i, min(i + B, len(passages)))],
            documents=passages[i:i + B],
            embeddings=embeddings[i:i + B].tolist(),
            metadatas=metas[i:i + B],
        )

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "source": "wikimedia/wikipedia (HuggingFace)",
        "dump": args.dump,
        "languages": ["en", "es"],
        "articles_streamed": {"en": args.en_articles, "es": args.es_articles},
        "passages_per_language": per_lang,
        "total_passages": len(passages),
        "chunk_chars": args.chunk_chars,
        "min_chars": args.min_chars,
        "chunks_per_article": args.chunks_per_article,
        "embed_model": args.embed_model,
        "embeddings_normalized": True,
        "chroma_space": "l2 (squared) on unit vectors -> similarity = 1 - dist/2 == cosine",
        "collection": args.collection,
        "path": os.path.relpath(args.out_path, ROOT),
        "independent_of_benchmark": "LIAR (claims are not sourced from Wikipedia)",
        "build_seconds": round(time.time() - t0, 1),
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2))
    print(f"[corpus] provenance -> {PROVENANCE.relative_to(ROOT)}")
    print(f"[corpus] DONE: {len(passages):,} passages in {provenance['build_seconds']}s "
          f"| collection={args.collection} count={col.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
