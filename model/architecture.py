"""
model/architecture.py
Custom transformer encoder for bilingual misinformation classification.
Built entirely from scratch in PyTorch — no pretrained weights.

Architecture:
  TokenEmbedding + PositionalEncoding + LanguageEmbedding
  → 4x TransformerEncoderBlock (MHA + FFN + LayerNorm + Dropout)
  → [CLS] pooling
  → ClassificationHead (Linear → ReLU → Dropout → Linear)
  → 4-class softmax (true / false / misleading / unverifiable)

Usage:
    from model.architecture import VerifAIClassifier
    model = VerifAIClassifier(vocab_size=16000)
    logits = model(input_ids, attention_mask, language_ids)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 1. Token Embedding ───────────────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    """
    Learned token embeddings.
    Randomly initialized — no pretrained weights.
    """
    def __init__(self, vocab_size: int, embed_dim: int, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.embed_dim = embed_dim
        # Scale embeddings by sqrt(embed_dim) per original transformer paper
        self.scale = math.sqrt(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len] → [batch, seq_len, embed_dim]
        return self.embedding(x) * self.scale


# ─── 2. Positional Encoding ───────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding + learned offset.
    Sinusoidal component: from "Attention Is All You Need" (Vaswani et al. 2017).
    Learned offset: small additive correction learned during training.
    """
    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Sinusoidal encoding — fixed, not learned
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer("pe", pe)

        # Learned offset — small correction on top of sinusoidal
        self.learned_offset = nn.Embedding(max_len, embed_dim)
        nn.init.normal_(self.learned_offset.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, embed_dim]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_enc = self.pe[:, :seq_len, :] + self.learned_offset(positions)
        return self.dropout(x + pos_enc)


# ─── 3. Language Embedding ────────────────────────────────────────────────────

class LanguageEmbedding(nn.Module):
    """
    Learned language embeddings: EN=0, ES=1.
    Allows the model to condition on input language.
    """
    def __init__(self, num_languages: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_languages, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, language_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        # language_ids: [batch] → expand to [batch, seq_len, embed_dim]
        lang_emb = self.embedding(language_ids)          # [batch, embed_dim]
        lang_emb = lang_emb.unsqueeze(1)                 # [batch, 1, embed_dim]
        lang_emb = lang_emb.expand(-1, seq_len, -1)      # [batch, seq_len, embed_dim]
        return lang_emb


# ─── 4. Multi-Head Self-Attention ─────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention from scratch.
    Implements scaled dot-product attention with causal masking support.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(p=dropout)

        # Initialize weights with small normal
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x:               [batch, seq_len, embed_dim]
            attention_mask:  [batch, seq_len] — 1=attend, 0=ignore (padding)
        Returns:
            out:             [batch, seq_len, embed_dim]
        """
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)                             # [B, T, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)                     # each [B, T, C]

        # Reshape for multi-head attention
        def reshape(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            # → [B, num_heads, T, head_dim]

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, T, T]

        # Apply padding mask: set padding positions to -inf before softmax
        if attention_mask is not None:
            # attention_mask: [B, T] → [B, 1, 1, T]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN from fully-masked rows (all padding) with 0
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        out = torch.matmul(attn, v)                        # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        out = self.out_proj(out)
        return out


# ─── 5. Feed-Forward Network ──────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Two linear layers with GELU activation and dropout.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(p=dropout),
        )
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── 6. Transformer Encoder Block ─────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block:
    LayerNorm → MHA → residual
    LayerNorm → FFN → residual
    Pre-norm formulation (more stable training than post-norm).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = FeedForward(embed_dim, hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Pre-norm + residual for attention
        x = x + self.attn(self.norm1(x), attention_mask)
        # Pre-norm + residual for FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ─── 7. Classification Head ───────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Two-layer MLP classification head.
    Input: [CLS] token representation [batch, embed_dim]
    Output: logits [batch, num_classes]
    """
    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── 8. VerifAIClassifier — full model ───────────────────────────────────────

class VerifAIClassifier(nn.Module):
    """
    Full from-scratch bilingual misinformation classifier.

    Input:
        input_ids:      [batch, seq_len]  — token IDs from BPETokenizer
        attention_mask: [batch, seq_len]  — 1=real, 0=padding
        language_ids:   [batch]           — 0=EN, 1=ES

    Output:
        logits:         [batch, num_classes]  — raw scores (apply softmax for probs)

    All weights randomly initialized — no pretrained weights used anywhere.
    """
    def __init__(
        self,
        vocab_size: int   = 16000,
        embed_dim: int    = 256,
        num_heads: int    = 8,
        num_layers: int   = 4,
        hidden_dim: int   = 512,
        max_length: int   = 256,
        num_classes: int  = 4,
        num_languages: int = 2,
        dropout: float    = 0.1,
        pad_id: int       = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding layers — all randomly initialized
        self.token_emb    = TokenEmbedding(vocab_size, embed_dim, pad_id)
        self.pos_enc      = PositionalEncoding(embed_dim, max_length, dropout)
        self.lang_emb     = LanguageEmbedding(num_languages, embed_dim)
        self.embed_norm   = nn.LayerNorm(embed_dim)
        self.embed_drop   = nn.Dropout(p=dropout)

        # Transformer encoder stack
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.classifier = ClassificationHead(embed_dim, num_classes, dropout)

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VerifAIClassifier initialized:")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable:,}")

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor = None,
        language_ids:   torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        Returns logits [batch, num_classes].
        """
        B, T = input_ids.shape

        # Token embeddings
        x = self.token_emb(input_ids)                     # [B, T, embed_dim]

        # Positional encoding
        x = self.pos_enc(x)                                # [B, T, embed_dim]

        # Language embedding (add bilingual signal)
        if language_ids is not None:
            x = x + self.lang_emb(language_ids, T)        # [B, T, embed_dim]

        x = self.embed_norm(x)
        x = self.embed_drop(x)

        # Transformer encoder blocks
        for block in self.encoder:
            x = block(x, attention_mask)                   # [B, T, embed_dim]

        x = self.final_norm(x)                             # [B, T, embed_dim]

        # [CLS] token pooling (position 0)
        cls_repr = x[:, 0, :]                              # [B, embed_dim]

        # Classification head
        logits = self.classifier(cls_repr)                 # [B, num_classes]
        return logits

    def predict(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor = None,
        language_ids:   torch.Tensor = None,
    ) -> tuple:
        """
        Convenience method returning (predicted_class, probabilities).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, language_ids)
            probs  = F.softmax(logits, dim=-1)
            preds  = torch.argmax(probs, dim=-1)
        return preds, probs

    @classmethod
    def from_config(cls, config: dict) -> "VerifAIClassifier":
        """Instantiate from config.yaml model section."""
        m = config["model"]
        return cls(
            vocab_size   = m["vocab_size"],
            embed_dim    = m["embed_dim"],
            num_heads    = m["num_heads"],
            num_layers   = m["num_layers"],
            hidden_dim   = m["hidden_dim"],
            max_length   = m["max_length"],
            num_classes  = m["num_labels"],
            dropout      = m["dropout"],
        )
