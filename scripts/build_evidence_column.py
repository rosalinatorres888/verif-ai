"""
scripts/build_evidence_column.py

Follow-up #2 for lever 1 (cross-encoder). Adds an `evidence` column to a claims
CSV by running each claim through the EXISTING retrieval layer
(app.pipeline.retrieval.retrieve_evidence), so the from-scratch classifier can
be trained on [CLS] claim [SEP] evidence [SEP] pairs instead of the claim alone.

Input  CSV columns : text, label, language, source        (evidence optional)
Output CSV columns : ...same... + evidence

CONTAMINATION GUARD (default ON)
--------------------------------
The model was de-contaminated, so we must NOT bake live web text into the
training set. By default this script is CORPUS-ONLY: it neutralizes Tavily
web supplementation by setting retrieval.MIN_CORPUS_RESULTS = 0, so evidence
comes solely from the curated ChromaDB corpus (deterministic, reproducible).
Pass --allow-web to opt back into Tavily supplementation (NOT recommended for
train/val/test that overlaps an evaluation benchmark).

WHERE TO RUN
------------
Requires the built ChromaDB index + embedding model (and, if --allow-web, the
Tavily key). Per project rules the corpus is built on OOD, so run this where the
index and .env are available — not necessarily a laptop.

Usage:
    python scripts/build_evidence_column.py --input data/train.csv --output data/train_evidence.csv
    python scripts/build_evidence_column.py --input data/test.csv  --output data/test_evidence.csv --top-k 3
    # opt into live web (discouraged for eval-overlapping splits):
    python scripts/build_evidence_column.py --input data/train.csv --output data/train_evidence.csv --allow-web
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Import the existing retrieval layer (lazy-loads chroma/model/tavily on first call).
from app.pipeline import retrieval
from app.pipeline.retrieval import retrieve_evidence

CHECKPOINT_EVERY = 50  # rows between incremental writes, so long runs survive a crash


def build_evidence_text(claim: str, language: str, top_k: int, max_chars: int, sep: str) -> str:
    """Retrieve evidence for one claim and flatten the top-k passages into one string."""
    evidence = retrieve_evidence(claim, language)  # already sorted best-first
    passages = []
    for item in evidence[:top_k]:
        passage = (item.get("passage") or "").strip()
        if passage:
            passages.append(passage[:max_chars])
    return sep.join(passages)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add an evidence column to a claims CSV via the retrieval layer.")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--text-col", default="text", help="Column holding the claim text")
    p.add_argument("--lang-col", default="language", help="Column holding the language code")
    p.add_argument("--evidence-col", default="evidence", help="Name of the column to write")
    p.add_argument("--top-k", type=int, default=3, help="Number of passages to concatenate")
    p.add_argument("--max-chars", type=int, default=500, help="Max chars kept per passage")
    p.add_argument("--sep", default=" ||| ", help="Separator between passages")
    p.add_argument("--allow-web", action="store_true",
                   help="Allow Tavily web supplementation (default: corpus-only). "
                        "Discouraged for eval-overlapping splits — contamination risk.")
    p.add_argument("--min-similarity", type=float, default=None,
                   help="Override retrieval SIMILARITY_THRESHOLD for this build "
                        "(e.g. 0.45). App default (0.65) is left unchanged.")
    p.add_argument("--fast", action="store_true",
                   help="Skip the reranker cross-encoder; rank by similarity*credibility only. "
                        "Much faster for large CSVs; ranking is near-identical for corpus-only.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-retrieve rows that already have evidence (default: only fill blanks)")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N rows (for testing)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    in_path, out_path = Path(args.input), Path(args.output)
    if not in_path.exists():
        print(f"[error] input not found: {in_path}", file=sys.stderr)
        return 1

    # --- contamination guard ---------------------------------------------------
    if not args.allow_web:
        # Never fall back to Tavily: corpus results always satisfy the threshold.
        retrieval.MIN_CORPUS_RESULTS = 0
        print("[mode] CORPUS-ONLY (Tavily disabled). Use --allow-web to enable web supplementation.")
    else:
        print("[mode] ⚠️  WEB-ENABLED (Tavily). Do not use on splits that overlap an evaluation benchmark.")

    # --- per-build overrides (do not mutate app defaults on disk) --------------
    if args.min_similarity is not None:
        retrieval.SIMILARITY_THRESHOLD = args.min_similarity
        print(f"[cfg] SIMILARITY_THRESHOLD overridden to {args.min_similarity} for this build")
    if args.fast:
        retrieval.score_batch = lambda q, passages, lang: [1.0] * len(passages)
        print("[cfg] FAST mode: reranker cross-encoder skipped")

    df = pd.read_csv(in_path)
    if args.text_col not in df.columns:
        print(f"[error] text column '{args.text_col}' not in {list(df.columns)}", file=sys.stderr)
        return 1
    if args.evidence_col not in df.columns:
        df[args.evidence_col] = ""
    df[args.evidence_col] = df[args.evidence_col].fillna("").astype(str)

    n = len(df) if args.limit is None else min(args.limit, len(df))
    print(f"[build] {n} rows → {out_path}  (top_k={args.top_k}, max_chars={args.max_chars})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    filled = no_match = skipped = failed = 0

    for i in range(n):
        existing = str(df.at[i, args.evidence_col]).strip()
        if existing and not args.overwrite:
            skipped += 1
            continue

        claim = str(df.at[i, args.text_col])
        language = str(df.at[i, args.lang_col]).lower() if args.lang_col in df.columns else "en"

        try:
            evidence_text = build_evidence_text(
                claim, language, args.top_k, args.max_chars, args.sep
            )
            df.at[i, args.evidence_col] = evidence_text
            if evidence_text:
                filled += 1        # got real evidence above threshold
            else:
                no_match += 1      # retrieval ran but nothing passed threshold → claim-only
        except Exception as e:  # never let one bad row abort the whole build
            df.at[i, args.evidence_col] = ""
            failed += 1
            print(f"[warn] row {i} failed: {e}", file=sys.stderr)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            df.to_csv(out_path, index=False)
            print(f"  ...{i + 1}/{n} (filled={filled} no_match={no_match} "
                  f"skipped={skipped} failed={failed})")

    df.to_csv(out_path, index=False)
    processed = filled + no_match + failed
    coverage = (filled / processed * 100) if processed else 0.0
    print(f"[done] filled={filled} no_match={no_match} skipped={skipped} failed={failed}")
    print(f"[done] evidence coverage: {filled}/{processed} = {coverage:.1f}% "
          f"(rest fall back to claim-only)")
    print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
