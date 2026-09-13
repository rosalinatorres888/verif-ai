"""
app/pipeline/nli_reranker.py

NLI / stance reranker: score how much a passage *verifies* a claim (supports or
refutes it) rather than merely mentions its topic. This is the fix for the
"topical but not verifying" retrieval failure documented in the technical report
(§5.6): dense cosine measures aboutness, NLI measures entailment.

For each (claim, passage) pair we run a multilingual NLI model with
  premise   = passage (the retrieved evidence)
  hypothesis = claim
and read three probabilities: entailment (passage supports claim),
contradiction (passage refutes claim), neutral (passage does not bear on claim).

We define a **stance** score = P(entail) + P(contradict) = 1 - P(neutral). A
verifying passage takes a stance (high); a topical-but-irrelevant passage is
neutral (low). Gating on stance keeps only passages that actually argue for or
against the claim.

Known limitation (observed): NLI can mislabel an *entity-mismatched* passage as
contradiction when attributes clash (e.g. a passage about a different freeway's
lane count vs a claim about the Katy Freeway). NLI improves precision over
cosine but does not itself verify entity alignment.

Default model: MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli (EN+ES capable,
small enough for MPS/CPU). Swap via --nli-model.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"


def _pick_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NLIReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = 16, max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = _pick_device(device)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length
        # Map label names to indices (order varies by model).
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.i_entail = next(i for i, l in id2label.items() if "entail" in l)
        self.i_contra = next(i for i, l in id2label.items() if "contra" in l)
        self.i_neutral = next(i for i, l in id2label.items() if "neutral" in l)

    @torch.no_grad()
    def stance(self, claim: str, passages: list[str]) -> list[dict]:
        """Return [{entail, contradict, neutral, stance}] for each passage."""
        results: list[dict] = []
        for i in range(0, len(passages), self.batch_size):
            batch = passages[i:i + self.batch_size]
            enc = self.tokenizer(
                batch, [claim] * len(batch),
                return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            probs = F.softmax(self.model(**enc).logits, dim=-1).cpu()
            for p in probs:
                e, n, c = float(p[self.i_entail]), float(p[self.i_neutral]), float(p[self.i_contra])
                results.append({"entail": e, "contradict": c, "neutral": n, "stance": e + c})
        return results
