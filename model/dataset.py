"""
model/dataset.py
PyTorch Dataset for VerifAI claim classification.
Works both locally and in Google Colab.

No pretrained weights anywhere in this file.

Usage:
    from model.dataset import ClaimDataset
    from model.tokenizer import BPETokenizer
    tok = BPETokenizer.load("models/verifai-classifier/vocab.json")
    ds = ClaimDataset("data/train.csv", tok, max_length=256)
    item = ds[0]  # {"input_ids", "attention_mask", "label", "language_id"}
"""
import sys
from pathlib import Path

# Support both local (model.tokenizer) and Colab (/content/tokenizer) imports
try:
    from model.tokenizer import BPETokenizer, PAD_ID
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent))
    from tokenizer import BPETokenizer, PAD_ID

import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL2ID = {"true": 0, "false": 1, "misleading": 2, "unverifiable": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LANG2ID  = {"en": 0, "es": 1}


class ClaimDataset(Dataset):
    """
    PyTorch Dataset for claim classification.

    Returns per item:
        input_ids:      LongTensor [max_length]
        attention_mask: LongTensor [max_length]  (1=real token, 0=pad)
        language_id:    LongTensor scalar         (0=EN, 1=ES)
        label:          LongTensor scalar         (0-3)
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: BPETokenizer,
        max_length: int = 256,
        label2id: dict = LABEL2ID,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text", "label"])
        df = df[df["label"].isin(label2id.keys())]
        df = df[df["text"].str.len() > 5]
        df = df.reset_index(drop=True)

        self.texts     = df["text"].tolist()
        self.labels    = df["label"].tolist()
        self.languages = df["language"].fillna("en").tolist()

        print(f"  Dataset loaded: {len(self.texts)} examples")
        label_counts = df["label"].value_counts().to_dict()
        print(f"  Labels: {label_counts}")
        lang_counts = df["language"].value_counts().to_dict()
        print(f"  Languages: {lang_counts}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text    = str(self.texts[idx])
        label   = self.label2id[self.labels[idx]]
        lang_id = LANG2ID.get(str(self.languages[idx]).lower(), 0)

        input_ids = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            add_special_tokens=True
        )
        input_ids, attention_mask = self.tokenizer.pad(input_ids, self.max_length)

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "language_id":    torch.tensor(lang_id, dtype=torch.long),
            "label":          torch.tensor(label, dtype=torch.long),
        }


def get_class_weights(csv_path: str, label2id: dict = LABEL2ID) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    Returns: Tensor of shape [num_classes]
    """
    df = pd.read_csv(csv_path)
    df = df[df["label"].isin(label2id.keys())]
    counts = df["label"].value_counts()
    total = len(df)
    num_classes = len(label2id)
    weights = torch.zeros(num_classes)
    for label, idx in label2id.items():
        count = counts.get(label, 1)
        weights[idx] = total / (num_classes * count)
    return weights
