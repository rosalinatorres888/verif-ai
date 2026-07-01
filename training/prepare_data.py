"""
training/prepare_data.py
Loads LIAR + MultiFC from HuggingFace (correct current IDs).
Maps all labels to unified 4-class schema.
Balances classes via upsampling.
Exports train.csv / val.csv / test.csv to data/

NOTE: Uses HuggingFace `datasets` for DATA LOADING ONLY — no model weights.

Usage:
    python training/prepare_data.py
    python training/prepare_data.py --liar-only   # use only LIAR if MultiFC fails
"""
import argparse
import random
import urllib.request
import zipfile
import io
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Label mapping ────────────────────────────────────────────────────────────

LIAR_INT_MAP = {
    0: "false", 1: "barely-true", 2: "half-true",
    3: "mostly-true", 4: "true", 5: "pants-fire"
}

LIAR_STR_MAP = {
    "true":        "true",
    "mostly-true": "true",
    "half-true":   "misleading",
    "barely-true": "unverifiable",
    "false":       "false",
    "pants-fire":  "false",
}

MULTIFC_MAP = {
    "true": "true", "correct": "true", "accurate": "true",
    "mostly true": "true", "mostly correct": "true",
    "false": "false", "incorrect": "false", "wrong": "false",
    "fake": "false", "pants on fire": "false", "pants-fire": "false",
    "misleading": "misleading", "mostly false": "misleading",
    "half true": "misleading", "half-true": "misleading",
    "mixture": "misleading", "mixed": "misleading",
    "partially true": "misleading", "partly true": "misleading",
    "out of context": "misleading", "missing context": "misleading",
    "unverifiable": "unverifiable", "unverified": "unverifiable",
    "unproven": "unverifiable", "disputed": "unverifiable",
    "unknown": "unverifiable", "uncertain": "unverifiable",
}

VALID_LABELS = {"true", "false", "misleading", "unverifiable"}


def map_liar_label(raw) -> str | None:
    if isinstance(raw, int):
        raw = LIAR_INT_MAP.get(raw, "")
    return LIAR_STR_MAP.get(str(raw).lower().strip(), None)


def map_multifc_label(raw: str) -> str | None:
    if not raw:
        return None
    normalized = raw.strip().lower()
    if normalized in MULTIFC_MAP:
        return MULTIFC_MAP[normalized]
    for key, val in MULTIFC_MAP.items():
        if key in normalized:
            return val
    return None


# ─── Dataset loaders ──────────────────────────────────────────────────────────

def load_liar_direct() -> pd.DataFrame:
    """
    Download LIAR directly from UCSB — bypasses HuggingFace loading script issues.
    Falls back to HuggingFace parquet versions if direct download fails.
    """
    print("Loading LIAR dataset...")

    # Try direct download from UCSB source
    url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    try:
        print("  Downloading from UCSB...")
        with urllib.request.urlopen(url, timeout=30) as resp:
            zdata = resp.read()
        with zipfile.ZipFile(io.BytesIO(zdata)) as z:
            rows = []
            split_files = {
                "train": "train.tsv",
                "validation": "valid.tsv",
                "test": "test.tsv"
            }
            for split_name, fname in split_files.items():
                if fname not in z.namelist():
                    continue
                with z.open(fname) as f:
                    for line in f:
                        parts = line.decode("utf-8").strip().split("\t")
                        if len(parts) < 3:
                            continue
                        # LIAR TSV: id, label, statement, ...
                        raw_label = parts[1].strip()
                        text = parts[2].strip()
                        label = map_liar_label(raw_label)
                        if label is None or not text:
                            continue
                        rows.append({
                            "text": text, "label": label,
                            "language": "en", "source": "liar",
                            "split": split_name
                        })
        df = pd.DataFrame(rows)
        print(f"  LIAR (direct): {len(df)} rows — {dict(df['label'].value_counts())}")
        return df
    except Exception as e:
        print(f"  Direct download failed: {e}")

    # Fallback: HuggingFace parquet versions
    from datasets import load_dataset
    for hf_id in ["ucsbnlp/liar", "liar"]:
        try:
            print(f"  Trying HuggingFace: {hf_id}...")
            ds = load_dataset(hf_id)
            rows = []
            for split_name, split in ds.items():
                for item in split:
                    raw_label = item.get("label", "")
                    label = map_liar_label(raw_label)
                    text = item.get("statement", item.get("claim", "")).strip()
                    if label is None or not text:
                        continue
                    rows.append({
                        "text": text, "label": label,
                        "language": "en", "source": "liar",
                        "split": split_name
                    })
            df = pd.DataFrame(rows)
            print(f"  LIAR ({hf_id}): {len(df)} rows — {dict(df['label'].value_counts())}")
            return df
        except Exception as e2:
            print(f"  {hf_id} failed: {e2}")

    print("  ERROR: Could not load LIAR from any source.")
    return pd.DataFrame()


def load_spanish_fakedes() -> pd.DataFrame:
    """
    Load FakeDeS Spanish fake news corpus (mariagrandury/fake_news_corpus_spanish).
    572 rows, true/false labels, academically sourced (IberLEF 2021).
    Author can verify Spanish quality directly.
    """
    print("Loading Spanish FakeDeS corpus...")
    from datasets import load_dataset
    try:
        ds = load_dataset("mariagrandury/fake_news_corpus_spanish")
        rows = []
        for split_name, split in ds.items():
            for item in split:
                # Use HEADLINE + TEXT for richer context
                headline = str(item.get("HEADLINE") or "").strip()
                text = str(item.get("TEXT") or "").strip()
                combined = f"{headline}. {text}" if headline and text else (headline or text)
                if not combined or len(combined) < 20:
                    continue
                # CATEGORY field: true=True (bool), false=False (bool)
                raw_label = item.get("CATEGORY")
                if raw_label is True or str(raw_label).lower() in ("true", "1"):
                    label = "true"
                elif raw_label is False or str(raw_label).lower() in ("false", "0"):
                    label = "false"
                else:
                    continue
                rows.append({
                    "text": combined[:512],  # cap at 512 chars
                    "label": label,
                    "language": "es",
                    "source": "fakedes",
                    "split": split_name
                })
        df = pd.DataFrame(rows)
        print(f"  FakeDeS: {len(df)} rows — {dict(df['label'].value_counts())}")
        return df
    except Exception as e:
        print(f"  FakeDeS load error: {e} — skipping.")
        return pd.DataFrame()


def load_multifc() -> pd.DataFrame:
    """Load MultiFC from HuggingFace."""
    print("Loading MultiFC dataset...")
    from datasets import load_dataset

    for hf_id in ["pszemraj/multi_fc", "multi_fc"]:
        try:
            print(f"  Trying {hf_id}...")
            ds = load_dataset(hf_id)
            rows = []
            for split_name, split in ds.items():
                for item in split:
                    text = item.get("claim", item.get("claimText", ""))
                    if not text or not isinstance(text, str):
                        continue
                    text = text.strip()
                    if not text:
                        continue
                    raw_label = item.get("label", item.get("veracity", ""))
                    if not raw_label or not isinstance(raw_label, str):
                        continue
                    label = map_multifc_label(str(raw_label))
                    if label is None:
                        continue
                    # Detect Spanish outlets
                    outlet = str(
                        item.get("claimant", "") or
                        item.get("outlet", "") or
                        item.get("author", "") or ""
                    ).lower()
                    spanish_outlets = [
                        "newtral", "maldita", "verificat", "colombiacheck",
                        "chequeado", "animal politico", "ojo publico",
                        "afp factual", "factchequeado"
                    ]
                    lang = "es" if any(s in outlet for s in spanish_outlets) else "en"
                    rows.append({
                        "text": text, "label": label,
                        "language": lang, "source": "multifc",
                        "split": split_name
                    })
            df = pd.DataFrame(rows)
            en_count = (df["language"] == "en").sum()
            es_count = (df["language"] == "es").sum()
            print(f"  MultiFC ({hf_id}): {len(df)} rows — EN: {en_count}, ES: {es_count}")
            print(f"  Labels: {dict(df['label'].value_counts())}")
            return df
        except Exception as e:
            print(f"  {hf_id} failed: {e}")

    print("  MultiFC unavailable — skipping.")
    return pd.DataFrame()


# ─── Balancing ────────────────────────────────────────────────────────────────

def balance_classes(df: pd.DataFrame, target_ratio: float = 0.8) -> pd.DataFrame:
    counts = df["label"].value_counts()
    majority_count = counts.max()
    target_count = int(majority_count * target_ratio)
    parts = []
    for label in VALID_LABELS:
        subset = df[df["label"] == label]
        if len(subset) == 0:
            print(f"  WARNING: no examples for label '{label}'")
            continue
        if len(subset) < target_count:
            upsampled = subset.sample(n=target_count, replace=True, random_state=SEED)
            parts.append(upsampled)
        else:
            parts.append(subset)
    balanced = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return balanced


# ─── Main ─────────────────────────────────────────────────────────────────────

def prepare_data(liar_only: bool = False):
    print("\n" + "="*60)
    print("VerifAI 2.0 — Data Preparation")
    print("="*60)

    dfs = []
    liar_df = load_liar_direct()
    if not liar_df.empty:
        dfs.append(liar_df)
    else:
        print("FATAL: LIAR dataset required. Exiting.")
        return

    # Spanish data — FakeDeS (IberLEF 2021)
    fakedes_df = load_spanish_fakedes()
    if not fakedes_df.empty:
        dfs.append(fakedes_df)

    if not liar_only:
        multifc_df = load_multifc()
        if not multifc_df.empty:
            dfs.append(multifc_df)
        else:
            print("  MultiFC unavailable — continuing with LIAR only.")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["language"].isin(["en", "es"])].copy()
    combined = combined[combined["text"].str.len() > 10].copy()

    print(f"\nCombined: {len(combined)} rows")
    print(f"Language — EN: {(combined['language']=='en').sum()}, "
          f"ES: {(combined['language']=='es').sum()}")

    # Use LIAR native test split as test set; everything else → train/val
    liar_test = combined[
        (combined["source"] == "liar") & (combined["split"] == "test")
    ].copy()
    train_val = combined[
        ~((combined["source"] == "liar") & (combined["split"] == "test"))
    ].copy()

    train_val = train_val.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_size = int(len(train_val) * 0.1)
    val_df = train_val[:val_size].copy()
    train_df = train_val[val_size:].copy()

    print("\nBalancing training set...")
    train_df = balance_classes(train_df)

    print(f"\nFinal splits:")
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    print(f"  Test:  {len(liar_test):,} rows (LIAR native test split)")

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts().to_string())
    print("\nTrain language distribution:")
    print(train_df["language"].value_counts().to_string())

    cols = ["text", "label", "language", "source"]
    train_df[cols].to_csv(DATA_DIR / "train.csv", index=False)
    val_df[cols].to_csv(DATA_DIR / "val.csv", index=False)
    liar_test[cols].to_csv(DATA_DIR / "test.csv", index=False)

    print(f"\n✅ Saved to {DATA_DIR}/")
    print(f"   train.csv — {len(train_df):,} rows")
    print(f"   val.csv   — {len(val_df):,} rows")
    print(f"   test.csv  — {len(liar_test):,} rows")

    print("\nSample training rows:")
    for _, row in train_df[cols].sample(3, random_state=SEED).iterrows():
        print(f"  [{row['language'].upper()}] [{row['label']}] {row['text'][:80]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--liar-only", action="store_true",
                        help="Use only LIAR dataset (skip MultiFC)")
    args = parser.parse_args()
    prepare_data(liar_only=args.liar_only)
