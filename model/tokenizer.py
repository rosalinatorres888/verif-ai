"""
model/tokenizer.py
Custom Byte-Pair Encoding (BPE) tokenizer built from scratch.
No external tokenizer libraries. No pretrained weights.

Shared vocabulary across EN and ES (bilingual).
Special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4

Usage:
    from model.tokenizer import BPETokenizer
    tok = BPETokenizer.load("models/verifai-classifier/vocab.json")
    ids = tok.encode("Vaccines cause autism.")
    text = tok.decode(ids)
"""
import re
import json
import collections
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# ─── Special tokens ───────────────────────────────────────────────────────────

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD_ID  = 0
UNK_ID  = 1
CLS_ID  = 2
SEP_ID  = 3
MASK_ID = 4


# ─── Text normalization ───────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Lightweight normalization:
    - Lowercase
    - Normalize unicode whitespace
    - Keep Spanish accented chars (á é í ó ú ü ñ ¿ ¡)
    - Keep alphanumeric + basic punctuation
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def pretokenize(text: str) -> List[str]:
    """
    Split text into words before BPE.
    Adds a special end-of-word marker </w> to each token
    so BPE merges don't cross word boundaries.
    """
    text = normalize_text(text)
    # Split on whitespace and punctuation, keep punctuation as separate tokens
    tokens = re.findall(
        r"[a-záéíóúüñàèìòùâêîôûäëïöü]+|[0-9]+|[^\w\s]",
        text,
        flags=re.UNICODE
    )
    # Add end-of-word marker to each word token
    return [" ".join(list(tok)) + " </w>" for tok in tokens if tok.strip()]


# ─── BPE training ─────────────────────────────────────────────────────────────

def get_vocab_from_corpus(texts: List[str]) -> Dict[str, int]:
    """
    Build initial character-level vocabulary from corpus.
    Each word is represented as a sequence of chars + </w> marker.
    Returns: {word_with_spaces: count}
    """
    vocab = collections.defaultdict(int)
    for text in texts:
        for word_repr in pretokenize(text):
            vocab[word_repr] += 1
    return dict(vocab)


def get_pair_frequencies(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """Count frequency of all adjacent symbol pairs in vocab."""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return dict(pairs)


def merge_vocab(pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
    """Apply a single BPE merge to the vocabulary."""
    new_vocab = {}
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    merged = "".join(pair)
    for word, freq in vocab.items():
        new_word = pattern.sub(merged, word)
        new_vocab[new_word] = freq
    return new_vocab


def train_bpe(
    texts: List[str],
    vocab_size: int = 32000,
    min_freq: int = 2,
    verbose: bool = True
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train BPE from scratch on a list of texts.

    Args:
        texts: list of raw text strings (training corpus)
        vocab_size: target vocabulary size
        min_freq: minimum pair frequency to merge
        verbose: print progress every 1000 merges

    Returns:
        token2id: {token: id} mapping
        merges: list of (a, b) merge rules in order
    """
    if verbose:
        print(f"  Building initial character vocabulary...")
    vocab = get_vocab_from_corpus(texts)

    # Initial symbol set = all unique characters + special tokens
    symbols = set()
    for word in vocab:
        for sym in word.split():
            symbols.add(sym)

    # Build initial token2id: special tokens first, then chars
    token2id: Dict[str, int] = {}
    for st in SPECIAL_TOKENS:
        token2id[st] = len(token2id)
    for sym in sorted(symbols):
        if sym not in token2id:
            token2id[sym] = len(token2id)

    if verbose:
        print(f"  Initial vocab size: {len(token2id)} characters")
        print(f"  Running BPE merges (target: {vocab_size})...")

    merges: List[Tuple[str, str]] = []
    num_merges = vocab_size - len(token2id)

    for i in range(num_merges):
        pairs = get_pair_frequencies(vocab)
        if not pairs:
            break

        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        best_freq = pairs[best_pair]

        if best_freq < min_freq:
            break

        # Apply merge
        vocab = merge_vocab(best_pair, vocab)
        new_token = "".join(best_pair)
        if new_token not in token2id:
            token2id[new_token] = len(token2id)
        merges.append(best_pair)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Merge {i+1}/{num_merges} — vocab size: {len(token2id)}")

        if len(token2id) >= vocab_size:
            break

    if verbose:
        print(f"  BPE training complete. Final vocab size: {len(token2id)}")

    return token2id, merges


# ─── BPETokenizer class ───────────────────────────────────────────────────────

class BPETokenizer:
    """
    BPE tokenizer with encode/decode.
    Built entirely from scratch — no HuggingFace tokenizers library.
    """

    def __init__(self, token2id: Dict[str, int], merges: List[Tuple[str, str]]):
        self.token2id = token2id
        self.id2token = {v: k for k, v in token2id.items()}
        # Build merge priority lookup for O(1) merge application
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.vocab_size = len(token2id)

    def _apply_merges(self, word_chars: List[str]) -> List[str]:
        """Apply BPE merges to a list of characters."""
        word = word_chars[:]
        while len(word) > 1:
            # Find the highest-priority merge available
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            ranked = [(self.merge_ranks.get(p, float("inf")), p) for p in pairs]
            best_rank, best_pair = min(ranked)
            if best_rank == float("inf"):
                break
            # Apply merge
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def tokenize(self, text: str) -> List[str]:
        """Convert text to list of BPE tokens."""
        tokens = []
        for word_repr in pretokenize(text):
            chars = word_repr.split()  # chars + </w>
            merged = self._apply_merges(chars)
            tokens.extend(merged)
        return tokens

    def encode(
        self,
        text: str,
        max_length: int = 256,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode text to token IDs.
        Adds [CLS] at start and [SEP] at end if add_special_tokens=True.
        Truncates to max_length.
        """
        tokens = self.tokenize(text)
        ids = [self.token2id.get(t, UNK_ID) for t in tokens]

        if add_special_tokens:
            ids = [CLS_ID] + ids + [SEP_ID]

        # Truncate (keep [CLS] and [SEP])
        if len(ids) > max_length:
            if add_special_tokens:
                ids = ids[:max_length-1] + [SEP_ID]
            else:
                ids = ids[:max_length]

        return ids

    def encode_pair(
        self,
        text_a: str,
        text_b: str,
        max_length: int = 256
    ) -> Tuple[List[int], List[int]]:
        """
        Encode a pair of texts for cross-attention tasks (e.g. claim + passage).
        Returns (input_ids, token_type_ids).
        Format: [CLS] text_a [SEP] text_b [SEP]
        token_type_ids: 0 for text_a, 1 for text_b
        """
        ids_a = self.encode(text_a, add_special_tokens=False)
        ids_b = self.encode(text_b, add_special_tokens=False)

        # Truncate to fit max_length: [CLS] + a + [SEP] + b + [SEP]
        max_content = max_length - 3
        if len(ids_a) + len(ids_b) > max_content:
            # Truncate longer sequence
            half = max_content // 2
            ids_a = ids_a[:half]
            ids_b = ids_b[:max_content - len(ids_a)]

        input_ids = [CLS_ID] + ids_a + [SEP_ID] + ids_b + [SEP_ID]
        token_type_ids = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)

        return input_ids, token_type_ids

    def pad(
        self,
        ids: List[int],
        max_length: int,
        return_attention_mask: bool = True
    ) -> Tuple[List[int], List[int]]:
        """Pad sequence to max_length. Returns (padded_ids, attention_mask)."""
        attention_mask = [1] * len(ids)
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids = ids + [PAD_ID] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        return ids, attention_mask

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        special_ids = {PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID}
        tokens = []
        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            tokens.append(self.id2token.get(i, "[UNK]"))
        # Reconstruct text from BPE tokens
        text = " ".join(tokens)
        text = text.replace(" </w>", " ").replace("</w>", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def save(self, path: str) -> None:
        """Save tokenizer to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.token2id,
            "merges": [list(m) for m in self.merge_ranks.keys()]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        token2id = data["vocab"]
        merges = [tuple(m) for m in data["merges"]]
        return cls(token2id=token2id, merges=merges)

    def __len__(self) -> int:
        return self.vocab_size
