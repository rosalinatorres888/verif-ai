"""
training/train_classifier.py
VerifAI 2.0 — Training loop for custom from-scratch bilingual classifier.

Requires GPU for reasonable training time.
On laptop (CPU): ~2hrs/epoch — use Colab T4 instead.
On OOD (GPU):    ~2min/epoch.

Usage:
    python training/train_classifier.py
    python training/train_classifier.py --epochs 5 --lr 5e-5
"""
import sys, os, json, time, math, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import BPETokenizer
from model.architecture import VerifAIClassifier
from model.dataset import ClaimDataset, get_class_weights, LABEL2ID, ID2LABEL

# ── Paths ─────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
VOCAB_PATH = ROOT / "models/verifai-classifier/vocab.json"
CKPT_DIR   = ROOT / "models/verifai-classifier"
TRAIN_PATH = ROOT / "data/train.csv"
VAL_PATH   = ROOT / "data/val.csv"
LOG_PATH   = ROOT / "outputs/training_log.json"
ROOT.joinpath("outputs").mkdir(exist_ok=True)

# ── Default config ────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "vocab_size":             16000,
    "embed_dim":              256,
    "num_heads":              8,
    "num_layers":             4,
    "hidden_dim":             512,
    "max_length":             256,
    "num_labels":             4,
    "dropout":                0.1,
    "batch_size":             32,
    "learning_rate":          5e-5,
    "warmup_steps":           1000,
    "num_epochs":             15,
    "weight_decay":           0.01,
    "gradient_clip":          1.0,
    "label_smoothing":        0.1,
    "early_stopping_patience": 3,
    "seed":                   42,
}


def train_epoch(model, loader, optimizer, scheduler, criterion, device, clip):
    model.train()
    total_loss, total_samples = 0.0, 0
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        language_ids   = batch["language_id"].to(device)
        labels         = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, language_ids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        preds = logits.argmax(dim=-1)
        total_loss    += loss.item() * len(labels)
        total_samples += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total_samples, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        language_ids   = batch["language_id"].to(device)
        labels         = batch["label"].to(device)
        logits = model(input_ids, attention_mask, language_ids)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        total_loss    += loss.item() * len(labels)
        total_samples += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total_samples, f1, all_preds, all_labels


def train(config: dict):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    # Prefer CUDA, then Apple Silicon GPU (MPS), then CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: No GPU detected. Training will be very slow.")
        print("Use Colab T4 GPU for reasonable training time.")

    # Load tokenizer
    print(f"\nLoading tokenizer from {VOCAB_PATH}...")
    tokenizer = BPETokenizer.load(str(VOCAB_PATH))

    # Load datasets (paths overridable via --train/--val for clean-split runs)
    train_path = config.get("train_path", str(TRAIN_PATH))
    val_path   = config.get("val_path",   str(VAL_PATH))
    print(f"\nLoading datasets...\n  train: {train_path}\n  val:   {val_path}")
    train_ds = ClaimDataset(train_path, tokenizer, config["max_length"])
    val_ds   = ClaimDataset(val_path,   tokenizer, config["max_length"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True,  num_workers=0, pin_memory=device.type=="cuda")
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"]*2,
                              shuffle=False, num_workers=0, pin_memory=device.type=="cuda")

    # Model
    print("\nInitializing model (from scratch — no pretrained weights)...")
    model = VerifAIClassifier(
        vocab_size  = config["vocab_size"],
        embed_dim   = config["embed_dim"],
        num_heads   = config["num_heads"],
        num_layers  = config["num_layers"],
        hidden_dim  = config["hidden_dim"],
        max_length  = config["max_length"],
        num_classes = config["num_labels"],
        dropout     = config["dropout"],
    ).to(device)

    # Loss, optimizer, scheduler
    class_weights = get_class_weights(str(TRAIN_PATH)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config["label_smoothing"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = config["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    training_log = []
    best_val_f1  = 0.0
    best_epoch   = 0
    patience_ctr = 0
    PATIENCE     = config["early_stopping_patience"]

    print("\n" + "="*60)
    print("VerifAI 2.0 — Training Run 4")
    print(f"Epochs: {config['num_epochs']} | LR: {config['learning_rate']} | "
          f"Label smoothing: {config['label_smoothing']}")
    print("="*60)

    for epoch in range(1, config["num_epochs"] + 1):
        t0 = time.time()
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, config["gradient_clip"]
        )
        val_loss, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        log_entry = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_f1": round(train_f1, 4), "val_loss": round(val_loss, 4),
            "val_f1": round(val_f1, 4), "lr": round(lr_now, 8),
            "elapsed_s": round(elapsed, 1)
        }
        training_log.append(log_entry)

        print(f"Epoch {epoch:02d}/{config['num_epochs']} "
              f"| train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"| val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
              f"| lr={lr_now:.2e} | {elapsed:.0f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            patience_ctr = 0
            ckpt_path = CKPT_DIR / config.get("ckpt_name", "best_model.pt")
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": val_f1, "config": config,
            }, ckpt_path)
            print(f"  ✅ New best checkpoint saved (val_f1={val_f1:.4f})")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    log_path = config.get("log_path", str(LOG_PATH))
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nTraining complete. Best val_f1={best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {CKPT_DIR / config.get('ckpt_name', 'best_model.pt')}")

    print(f"\nFinal classification report (val set):")
    print(classification_report(
        val_labels, val_preds,
        target_names=["true", "false", "misleading", "unverifiable"],
        zero_division=0
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--lr",     type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--batch",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--train", type=str, default=str(TRAIN_PATH),
                        help="training CSV (use data/train_clean.csv for the "
                             "de-contaminated run)")
    parser.add_argument("--val", type=str, default=str(VAL_PATH),
                        help="validation CSV")
    parser.add_argument("--ckpt-name", type=str, default="best_model.pt",
                        help="checkpoint filename inside models/verifai-classifier/")
    parser.add_argument("--log-path", type=str, default=str(LOG_PATH),
                        help="where to write the per-epoch training log")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["num_epochs"]    = args.epochs
    config["learning_rate"] = args.lr
    config["batch_size"]    = args.batch
    config["train_path"]    = args.train
    config["val_path"]      = args.val
    config["ckpt_name"]     = args.ckpt_name
    config["log_path"]      = args.log_path
    train(config)
