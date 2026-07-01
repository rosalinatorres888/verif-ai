"""
training/build_tokenizer.py
Builds BPE vocabulary from scratch using the training corpus.
Must be run before train_classifier.py.

No pretrained weights. No external tokenizer libraries.
Reads data/train.csv, builds BPE vocab, saves to models/verifai-classifier/vocab.json.

Usage:
    python training/build_tokenizer.py
    python training/build_tokenizer.py --vocab-size 16000   # smaller for testing
"""
import sys
import argparse
import time
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.tokenizer import BPETokenizer, train_bpe

TRAIN_PATH  = Path(__file__).parent.parent / "data" / "train.csv"
VOCAB_PATH  = Path(__file__).parent.parent / "models" / "verifai-classifier" / "vocab.json"


def build_tokenizer(vocab_size: int = 16000, min_freq: int = 2):
    print("\n" + "="*60)
    print("VerifAI 2.0 — BPE Tokenizer Build")
    print("="*60)

    # Load training corpus
    print(f"\nLoading training data from {TRAIN_PATH}...")
    df = pd.read_csv(TRAIN_PATH)
    texts = df["text"].dropna().tolist()
    print(f"  {len(texts):,} training texts")
    print(f"  EN: {(df['language']=='en').sum():,}  ES: {(df['language']=='es').sum():,}")

    # Train BPE
    print(f"\nTraining BPE tokenizer (vocab_size={vocab_size}, min_freq={min_freq})...")
    start = time.time()
    token2id, merges = train_bpe(
        texts=texts,
        vocab_size=vocab_size,
        min_freq=min_freq,
        verbose=True
    )
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final vocabulary size: {len(token2id):,} tokens")
    print(f"Number of BPE merges: {len(merges):,}")

    # Build tokenizer and save
    tokenizer = BPETokenizer(token2id=token2id, merges=merges)
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(VOCAB_PATH))
    print(f"\n✅ Saved to {VOCAB_PATH}")

    # Smoke test
    print("\n" + "-"*40)
    print("Smoke test:")
    test_cases = [
        ("EN", "Vaccines do not cause autism according to multiple studies."),
        ("ES", "Las vacunas no causan autismo según múltiples estudios científicos."),
        ("EN", "The moon landing in 1969 was faked by NASA."),
        ("ES", "El gobierno está mintiendo sobre el cambio climático."),
    ]
    for lang, text in test_cases:
        ids = tokenizer.encode(text, max_length=64)
        decoded = tokenizer.decode(ids)
        tokens = tokenizer.tokenize(text)
        print(f"\n[{lang}] Input:   {text[:70]}")
        print(f"       Tokens:  {tokens[:10]}{'...' if len(tokens)>10 else ''}")
        print(f"       IDs:     {ids[:10]}{'...' if len(ids)>10 else ''}")
        print(f"       Decoded: {decoded[:70]}")

    # Verify [CLS] and [SEP] present
    ids = tokenizer.encode("test sentence")
    assert ids[0] == 2, "Expected [CLS]=2 at position 0"
    assert ids[-1] == 3, "Expected [SEP]=3 at last position"

    # Verify pair encoding works
    input_ids, token_type_ids = tokenizer.encode_pair(
        "Vaccines cause autism.",
        "There is no evidence linking vaccines to autism."
    )
    assert len(input_ids) == len(token_type_ids), "Mismatch in pair encoding lengths"
    print(f"\nPair encoding test: {len(input_ids)} tokens, "
          f"type_ids sum={sum(token_type_ids)} (segment B tokens)")

    print("\n✅ All smoke tests passed.")
    print(f"\nVocabulary saved to: {VOCAB_PATH}")
    print("Next step: python training/train_classifier.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=16000,
                        help="Target vocabulary size (default: 16000)")
    parser.add_argument("--min-freq", type=int, default=2,
                        help="Minimum pair frequency for BPE merge (default: 2)")
    args = parser.parse_args()
    build_tokenizer(vocab_size=args.vocab_size, min_freq=args.min_freq)
