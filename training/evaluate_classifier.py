"""
training/evaluate_classifier.py
Block E — Test set evaluation for VerifAI 2.0 trained classifier.

Loads best_model.pt, runs inference on test.csv (LIAR held-out),
produces per-class F1, EN vs ES breakdown, and confusion matrix.

Usage:
    python training/evaluate_classifier.py
"""
import sys, json, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import BPETokenizer
from model.architecture import VerifAIClassifier
from model.dataset import ClaimDataset

ROOT       = Path(__file__).parent.parent
VOCAB_PATH = ROOT / "models/verifai-classifier/vocab.json"
CKPT_PATH  = ROOT / "models/verifai-classifier/best_model.pt"
TEST_PATH  = ROOT / "data/test.csv"
OUT_DIR    = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

LABEL_NAMES = ["true", "false", "misleading", "unverifiable"]


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_languages = [], [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        language_ids   = batch["language_id"].to(device)
        labels         = batch["label"].to(device)
        logits = model(input_ids, attention_mask, language_ids)
        preds  = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_languages.extend(language_ids.cpu().numpy())
    return all_preds, all_labels, all_languages


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate():
    print("\n" + "="*60)
    print("VerifAI 2.0 — Block E: Test Set Evaluation")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading tokenizer...")
    tokenizer = BPETokenizer.load(str(VOCAB_PATH))

    print(f"Loading checkpoint...")
    ckpt   = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    print(f"  Epoch: {ckpt.get('epoch')}  Val F1: {ckpt.get('val_f1')}")

    model = VerifAIClassifier(
        vocab_size  = config.get("vocab_size", 16000),
        embed_dim   = config.get("embed_dim", 256),
        num_heads   = config.get("num_heads", 8),
        num_layers  = config.get("num_layers", 4),
        hidden_dim  = config.get("hidden_dim", 512),
        max_length  = config.get("max_length", 256),
        num_classes = config.get("num_labels", 4),
        dropout     = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print("  Model loaded.")

    print(f"\nLoading test set...")
    test_ds = ClaimDataset(str(TEST_PATH), tokenizer,
                           max_length=config.get("max_length", 256))
    test_loader = DataLoader(test_ds, batch_size=64,
                             shuffle=False, num_workers=0)

    print(f"Running inference on {len(test_ds)} examples...")
    t0 = time.time()
    preds, labels, languages = run_inference(model, test_loader, device)
    elapsed = time.time() - t0

    f1_macro    = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    precision   = precision_score(labels, preds, average="macro", zero_division=0)
    recall      = recall_score(labels, preds, average="macro",    zero_division=0)

    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS  (n={len(test_ds)}, {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"F1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(labels, preds,
                                target_names=LABEL_NAMES, zero_division=0))

    f1_per    = f1_score(labels, preds, average=None,
                         labels=list(range(4)), zero_division=0)
    per_class = {LABEL_NAMES[i]: round(float(f1_per[i]), 4) for i in range(4)}

    preds_arr  = np.array(preds)
    labels_arr = np.array(labels)
    lang_arr   = np.array(languages)
    en_mask    = lang_arr == 0
    es_mask    = lang_arr == 1

    en_results = {"n": int(en_mask.sum()), "f1_macro": 0.0, "per_class": {}}
    es_results = {"n": int(es_mask.sum()), "f1_macro": 0.0, "per_class": {}}

    print(f"\n{'='*60}")
    print(f"EN vs ES BREAKDOWN")
    print(f"{'='*60}")

    if en_mask.sum() > 0:
        en_f1  = f1_score(labels_arr[en_mask], preds_arr[en_mask],
                          average="macro", zero_division=0)
        en_per = f1_score(labels_arr[en_mask], preds_arr[en_mask],
                          average=None, labels=list(range(4)), zero_division=0)
        en_results["f1_macro"]  = round(float(en_f1), 4)
        en_results["per_class"] = {LABEL_NAMES[i]: round(float(en_per[i]), 4)
                                   for i in range(4)}
        print(f"English (n={en_results['n']:,}): F1 macro = {en_f1:.4f}")
        print(f"  {en_results['per_class']}")

    if es_mask.sum() > 0:
        es_f1  = f1_score(labels_arr[es_mask], preds_arr[es_mask],
                          average="macro", zero_division=0)
        es_per = f1_score(labels_arr[es_mask], preds_arr[es_mask],
                          average=None, labels=list(range(4)), zero_division=0)
        es_results["f1_macro"]  = round(float(es_f1), 4)
        es_results["per_class"] = {LABEL_NAMES[i]: round(float(es_per[i]), 4)
                                   for i in range(4)}
        print(f"Spanish (n={es_results['n']:,}): F1 macro = {es_f1:.4f}")
        gap = abs(en_results["f1_macro"] - es_results["f1_macro"])
        print(f"EN vs ES gap: {gap:.4f}")
    else:
        print("Test set is EN-only (LIAR). ES evaluation in Block I.")

    cm_path = OUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(labels, preds, cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")

    results = {
        "run_date":               time.strftime("%Y-%m-%d"),
        "checkpoint_epoch":       ckpt.get("epoch"),
        "checkpoint_val_f1":      ckpt.get("val_f1"),
        "n_test":                 len(test_ds),
        "f1_macro":               round(f1_macro, 4),
        "f1_weighted":            round(f1_weighted, 4),
        "precision_macro":        round(precision, 4),
        "recall_macro":           round(recall, 4),
        "per_class_f1":           per_class,
        "latency_ms_per_example": round(elapsed / len(test_ds) * 1000, 2),
        "en":                     en_results,
        "es":                     es_results,
    }
    out_path = OUT_DIR / "classifier_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")
    print(f"\n✅ Block E complete. Next: Block F (reranker.py)")


if __name__ == "__main__":
    evaluate()
