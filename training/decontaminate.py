"""
training/decontaminate.py
Removes train/test and train/val contamination from the merged corpus.

WHY THIS EXISTS
---------------
The training corpus merges MultiFC (Augenstein et al., 2019) with LIAR
(Wang, 2017). MultiFC aggregates claims scraped from 26 fact-checking
sites — one of which is PolitiFact. LIAR *is* PolitiFact. So although the
pipeline preserved LIAR's native train/test boundary on the LIAR side, the
boundary was breached from the MultiFC side: claims held out in LIAR's test
split re-entered training as MultiFC rows.

Audit of the original splits (data/train.csv, data/test.csv, data/val.csv):
    343 / 1283 test rows (26.7%) appear verbatim in train
      -> 339 arrived via MultiFC, 4 via LIAR
      -> 333 carry the same label (pure leakage); 62 carry a different one
    669 val rows also appear in train

Any test metric computed against the original split is therefore optimistic.
This script produces clean splits so the classifier can be retrained and
evaluated on a genuinely held-out test set.

WHAT IT DOES
------------
Drops from train every row whose claim text appears in the test or val
split (exact string match after whitespace normalization). Test and val are
left untouched — the held-out data is the ground truth being protected, so
the training set is what yields.

Note on duplicates: train also contains 12,172 duplicate rows over 25,436
unique texts. Those are deliberate — the corpus was class-balanced by
upsampling minority classes — and are NOT removed here. Upsampling within
the training set is a legitimate technique; leaking test rows into training
is not. Only the leak is fixed.

Usage:
    python training/decontaminate.py
    python training/decontaminate.py --dry-run    # report only, write nothing

Writes: data/train_clean.csv, data/val_clean.csv (originals untouched)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"


def normalize(series: pd.Series) -> pd.Series:
    """Whitespace-normalize claim text for comparison (not for storage)."""
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def audit(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Quantify contamination before removing it."""
    train_norm = normalize(train["text"])
    test_norm = normalize(test["text"])
    val_norm = normalize(val["text"])

    test_texts = set(test_norm)
    val_texts = set(val_norm)

    test_in_train_mask = test_norm.isin(set(train_norm))
    leaked_test_texts = set(test_norm[test_in_train_mask])

    # Which train source did the leaked rows arrive through?
    train_leaked = train[train_norm.isin(leaked_test_texts)]
    by_source = train_leaked["source"].value_counts().to_dict()

    # Same-label vs contradictory-label leakage
    t = pd.DataFrame({"text": test_norm, "label": test["label"]})
    r = pd.DataFrame({"text": train_norm, "label": train["label"]})
    merged = t[test_in_train_mask].merge(r, on="text", suffixes=("_test", "_train"))
    same = int((merged["label_test"] == merged["label_train"]).sum())
    diff = int((merged["label_test"] != merged["label_train"]).sum())

    return {
        "test_rows": len(test),
        "test_rows_leaked": int(test_in_train_mask.sum()),
        "test_leak_pct": round(100 * test_in_train_mask.sum() / len(test), 2),
        "leaked_arrived_via_train_source": by_source,
        "leaked_same_label_rows": same,
        "leaked_contradictory_label_rows": diff,
        "val_rows_also_in_train": int(val_norm.isin(set(train_norm)).sum()),
        "train_rows": len(train),
        "train_unique_texts": int(train_norm.nunique()),
        "train_duplicate_rows_from_upsampling": len(train) - int(train_norm.nunique()),
    }


def decontaminate(dry_run: bool = False) -> dict:
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")
    test = pd.read_csv(DATA / "test.csv")

    report = audit(train, val, test)

    print("=" * 64)
    print("CONTAMINATION AUDIT (original splits)")
    print("=" * 64)
    print(f"  test rows leaked into train : {report['test_rows_leaked']} / "
          f"{report['test_rows']}  ({report['test_leak_pct']}%)")
    print(f"    arrived via train source  : {report['leaked_arrived_via_train_source']}")
    print(f"    same-label / contradictory: {report['leaked_same_label_rows']} / "
          f"{report['leaked_contradictory_label_rows']}")
    print(f"  val rows also in train      : {report['val_rows_also_in_train']}")
    print(f"  train rows / unique texts   : {report['train_rows']} / "
          f"{report['train_unique_texts']} "
          f"({report['train_duplicate_rows_from_upsampling']} dupes from upsampling)")

    # Drop from TRAIN anything appearing in test or val. Held-out data wins.
    train_norm = normalize(train["text"])
    forbidden = set(normalize(test["text"])) | set(normalize(val["text"]))
    keep_mask = ~train_norm.isin(forbidden)

    train_clean = train[keep_mask].copy()
    # val must also not overlap test
    val_norm = normalize(val["text"])
    val_clean = val[~val_norm.isin(set(normalize(test["text"])))].copy()

    report["train_rows_after_clean"] = len(train_clean)
    report["train_rows_removed"] = len(train) - len(train_clean)
    report["val_rows_after_clean"] = len(val_clean)
    report["val_rows_removed"] = len(val) - len(val_clean)
    report["train_label_distribution_after"] = train_clean["label"].value_counts().to_dict()
    report["train_language_distribution_after"] = train_clean["language"].value_counts().to_dict()

    print()
    print("=" * 64)
    print("AFTER DECONTAMINATION")
    print("=" * 64)
    print(f"  train: {len(train)} -> {len(train_clean)}  "
          f"(-{report['train_rows_removed']} rows)")
    print(f"  val:   {len(val)} -> {len(val_clean)}  "
          f"(-{report['val_rows_removed']} rows)")
    print(f"  train labels : {report['train_label_distribution_after']}")
    print(f"  train langs  : {report['train_language_distribution_after']}")

    # Verify the fix actually worked
    resid_test = normalize(train_clean["text"]).isin(set(normalize(test["text"]))).sum()
    resid_val = normalize(train_clean["text"]).isin(set(normalize(val_clean["text"]))).sum()
    report["residual_train_test_overlap"] = int(resid_test)
    report["residual_train_val_overlap"] = int(resid_val)
    print(f"  residual train-test overlap: {resid_test}  (must be 0)")
    print(f"  residual train-val  overlap: {resid_val}  (must be 0)")
    assert resid_test == 0, "decontamination failed: train still overlaps test"

    if dry_run:
        print("\n[dry-run] no files written.")
        return report

    train_clean.to_csv(DATA / "train_clean.csv", index=False)
    val_clean.to_csv(DATA / "val_clean.csv", index=False)
    out = ROOT / "outputs" / "decontamination_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote data/train_clean.csv, data/val_clean.csv")
    print(f"Audit report: outputs/decontamination_report.json")
    print("Original splits are untouched.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="report contamination without writing clean files")
    args = parser.parse_args()
    decontaminate(dry_run=args.dry_run)
