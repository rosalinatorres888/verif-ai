# VerifAI: Bilingual Retrieval-Augmented Misinformation Detection

**Rosalina Torres** · IE 7374 — Generative AI · Northeastern University · Summer 2026
Solo project. Repository: https://github.com/rosalinatorres888/verif-ai

---

## 1. Introduction & Motivation

Misinformation crosses languages. Most automated verification tools don't.

Spanish-speaking communities face comparable or higher misinformation exposure
than English-speaking ones — Abrajano et al. (2024) provide the first empirical
demonstration that Latinos who rely on Spanish-language social media are
significantly more likely to hold false political beliefs — while having access
to far fewer automated verification tools. Of 264,487 fact-checks analyzed
globally by Quelle et al. (2025), 33.1% were in English against only 8.78% in
Spanish, despite English speakers being a minority of global internet users.
That gap between exposure and coverage is the equity problem this project
addresses.

VerifAI is a bilingual (English/Spanish) retrieval-augmented fact-checking
prototype. It accepts a claim in either language, retrieves relevant evidence,
and returns a grounded verdict — *true*, *false*, *misleading*, or
*unverifiable* — with a generated explanation in the input language. It
combines three kinds of model: a 6.3M-parameter transformer classifier trained
entirely from scratch, a pretrained multilingual sentence-embedding model for
cross-lingual retrieval, and a pretrained large language model (Claude,
`claude-sonnet-4-5`) for evidence-grounded verdict synthesis.

The project pre-registered three research questions:

- **RQ1:** Does RAG-grounded verdict generation produce higher factual
  faithfulness than LLM-only generation on bilingual misinformation claims, as
  measured by RAGAS faithfulness and answer relevancy on 100 sampled verdicts?
- **RQ2:** Does using the from-scratch classifier as a RAG evidence reranker
  improve retrieval quality over cosine similarity alone, and does that
  improvement translate to higher verdict precision on the LIAR benchmark?
- **RQ3:** What is the English-vs-Spanish F1 gap of the trained classifier, and
  which training data source contributes most to Spanish-language performance?

One design decision shaped everything downstream: the classifier and its
tokenizer were built and trained from scratch rather than fine-tuned from
pretrained weights. Two reasons — full engineering ownership of the
architecture and training loop as a learning objective, and an honest
from-scratch baseline that shows how data scarcity manifests *before* the
confound of pretrained multilingual weights is introduced.

## 2. Related Work

**Task framing.** Guo, Schlichtkrull, and Vlachos (2022) decompose automated
fact-checking into claim detection, evidence retrieval, and verdict prediction.
VerifAI's three-layer pipeline (intake → retrieval → verdict) follows that
decomposition deliberately: it lets each stage be evaluated and ablated
independently, matching the survey's observation that fact-checking failures
are usually attributable to a specific stage rather than the system as a whole.

**Retrieval-augmented generation.** Lewis et al. (2020) introduced RAG, showing
that grounding a generator in retrieved non-parametric evidence improves
factual accuracy on knowledge-intensive tasks including fact verification.
VerifAI operationalizes the paper's central claim as a hard prompt constraint —
the LLM is instructed to use only the evidence provided — and RQ1 is a
bilingual replication of the comparison Lewis et al. ran for English.

**Cross-lingual retrieval.** Reimers and Gurevych (2020) showed multilingual
transformers produce sentence embeddings that are *not* aligned across
languages out of the box, and introduced the knowledge-distillation method
behind `paraphrase-multilingual-MiniLM-L12-v2`, the embedding model VerifAI
uses. Substituting an English-only embedder would silently degrade
Spanish-query retrieval.

**The multilingual gap.** X-FACT (Gupta & Srikumar, 2021) benchmarks fact
verification across 25 languages; its best model reaches only ~40% F-score,
establishing that multilingual fact verification is substantially harder than
the English-only setting. FacTeR-Check (Martín et al., 2022) is one of few
systems built specifically for Spanish misinformation. Beyond Translation
(Chung, Cobo, & Serna, 2025) introduces MultiSynFact — 2.2M synthetic
claim-source pairs for Spanish, German, English, and additional languages —
motivated by the scarcity of real labeled fact-checking data outside English.
VerifAI's synthetic Spanish corpus is a smaller-scale, manually verified
instance of the same strategy.

**Evaluation.** Retrieval and generation quality are measured with RAGAS
(Es et al., 2024), which scores faithfulness (is the answer grounded in the
retrieved context?) and answer relevancy without requiring reference answers.

**Positioning.** VerifAI does not introduce a new paradigm — RAG verification,
multilingual retrieval, and reranking are established individually. Its
contribution is combining them for a language pair and resource level the
literature shows is underserved, with one uncommon element: a single
from-scratch classifier serving as both first-pass verdict signal and RAG
evidence reranker.

## 3. Dataset Summary

Four sources were merged into a four-class dataset (true / false / misleading /
unverifiable):

| Source | Type | Rows (original train) |
|---|---|---|
| MultiFC (Augenstein et al., 2019) | English, 26 fact-check sites | 21,673 |
| LIAR (Wang, 2017) | English, PolitiFact | 14,944 |
| FakeDeS (Gómez-Adorno et al., 2021) | Spanish, IberLEF shared task | 516 |
| Synthetic ES (this project) | Spanish, Claude-generated, author-verified | 475 |

Preprocessing (`training/prepare_data.py`) normalized labels — MultiFC's 165
free-text site-specific labels required fuzzy matching onto four classes — and
applied **class balancing by upsampling minority classes**, which is why the
original training set held 37,608 rows over 25,436 unique texts. LIAR's six
classes were mapped to four, preserving LIAR's native test split rather than
re-splitting.

### 3.1 A contamination audit, and what it found

Late in the project I audited the splits for verbatim text overlap. **343 of the
1,283 held-out test claims (26.7%) were present in the training set**, 339 of
them arriving via MultiFC and 333 carrying the same label. 669 validation rows
were also in train.

The cause is a merge artifact neither dataset advertises: **MultiFC aggregates
claims scraped from 26 fact-checking sites, one of which is PolitiFact — and
LIAR *is* PolitiFact.** LIAR's native test boundary was preserved on the LIAR
side and silently breached from the MultiFC side.

`training/decontaminate.py` rebuilds the splits, dropping from train any row
whose text appears in test or val (held-out data is never modified) and
asserting zero residual overlap. Final splits: **36,312 train** (35,321 EN /
991 ES), **3,354 validation**, **1,283 test** (LIAR, English only). Every
removed row was English MultiFC — the Spanish corpus is 991 rows before and
after, so the data-scarcity findings do not depend on the leak.

Two dataset characteristics drive this project's results. First, the ~36:1
English-to-Spanish imbalance, which is a faithful reflection of what labeled
data exists, and measuring its consequences is RQ3. Second, the synthetic
Spanish corpus: 475 generated, manually reviewed examples created to shore up
the Spanish side.

## 4. Methodology

### 4.1 System architecture

Three layers, served by a FastAPI backend with a Streamlit frontend:

1. **Intake** — language detection and claim extraction.
2. **Retrieval** — a ChromaDB vector store over a curated trusted-source
   corpus, supplemented by live Tavily web search when the corpus returns too
   few results; candidates scored by
   `combined_score = cosine_similarity × credibility_score × reranker_score`.
3. **Verdict** — the classifier produces a first-pass signal; Claude receives
   the claim, retrieved evidence, and that signal, and returns a structured
   verdict with an explanation in the input language, under a hard constraint
   to use only the provided evidence.

The **pretrained generative model** is Claude (`claude-sonnet-4-5`), invoked
via API. The **pretrained retrieval model** is
`paraphrase-multilingual-MiniLM-L12-v2`, run locally. The from-scratch
classifier is an additional component alongside these, not a substitute.

### 4.2 The from-scratch classifier

**VerifAIClassifier** is a 6.3M-parameter transformer encoder: 4 blocks,
embedding dimension 256, 8 attention heads, hidden dimension 512, maximum
sequence length 256, 4-class head. The tokenizer is a custom byte-pair-encoding
vocabulary of 16,000 tokens trained over the combined English/Spanish corpus,
with a learned language embedding (EN=0, ES=1).

The same checkpoint plays a dual role (RQ2): first-pass verdict signal, and
evidence reranker inside retrieval scoring.

Training: AdamW, learning rate 5e-5, 1,000 warmup steps with cosine decay,
weight decay 0.01, batch size 32, label smoothing 0.1, gradient clipping 1.0,
dropout 0.1, early stopping patience 3, seed 42, up to 15 epochs. The best
checkpoint by validation macro-F1 landed at epoch 15. These values are the
defaults in `training/train_classifier.py` (`DEFAULT_CONFIG`), which is what
the training run executed; the architecture config stored in the shipped
checkpoint is documented in `configs/model_config.yaml`. *(Note:
`training/config.yaml` documents an earlier abandoned configuration and is not
read by the training path.)*

Both the original and de-contaminated models were trained on Colab with
identical hardware (T4, ~107 s/epoch) and identical hyperparameters, so the
data was the only changed variable between them.

### 4.3 Key design decisions

**Pretrained proxy first, from-scratch second.** The first working verdict
layer used `cardiffnlp/twitter-xlm-roberta-base-sentiment`, a pretrained
multilingual sentiment model, as a fake/real proxy (negative→fake,
positive→real). It ran in the live pipeline, and its shortcomings were
instructive: sentiment is not veracity, the proxy was binary rather than
four-class, and it was never a fair baseline because it was never fine-tuned
for the task. ADR-001 and ADR-003 in the repository document the decision and
its reversal. No evaluation metrics were recorded for the proxy phase, so this
report makes no quantitative claim about it; a fair comparison (fine-tuning
`xlm-roberta-base` on the same labels and splits) is future work.

**Graceful degradation.** Claude is optional at runtime. If the API is
unavailable, the verdict layer returns the classifier's label, the retrieved
evidence, and an honest non-generative explanation, with a `generation_mode`
field (`claude` vs `classifier_fallback`) so the interface discloses which mode
produced any verdict. This began as an outage workaround and became a design
feature: it makes the pipeline's stages separable and independently observable.

**Honest labeling.** The interface distinguishes "With RAG — Classifier +
Retrieved Evidence" from "Without RAG — Classifier Only" and displays the
generation mode explicitly. An earlier version mislabeled fallback verdicts as
"LLM only" when no LLM had run; correcting that was treated as a correctness
fix, not cosmetics.

## 5. Experiments & Results

All results below were produced by the de-contaminated model shipped in the
repository. Scripts: `training/evaluate_classifier.py`, `src/model_runner.py`,
`evaluation/ablation.py`, `evaluation/ragas_eval.py`.

### 5.1 Classifier evaluation (held-out test set)

On the 1,283-claim LIAR test split (English only), the clean model reaches
**macro-F1 0.3313** (weighted 0.3618; validation macro-F1 0.3849 at epoch 15).

| Class | F1 |
|---|---|
| true | 0.4961 |
| false | 0.3614 |
| misleading | 0.2316 |
| unverifiable | 0.2362 |

**Figure 1** — `outputs/confusion_matrix.png`: test-set confusion matrix. The
diagonal is strongest for *true*; *misleading* and *unverifiable* absorb the
bulk of the confusion, consistent with their per-class F1.

**Contamination effect.** The same architecture and hyperparameters trained on
the contaminated corpus reported macro-F1 0.3647 — **10.1% higher than the
clean result**:

| | Contaminated | Clean | Δ |
|---|---|---|---|
| Test F1 macro | 0.3647 | **0.3313** | −0.0334 |
| Test F1 weighted | 0.3986 | 0.3618 | −0.0368 |
| Validation F1 | 0.4049 | 0.3849 | −0.0200 |
| true | 0.5503 | 0.4961 | −0.0542 |
| false | 0.3828 | 0.3614 | −0.0214 |
| misleading | 0.2809 | 0.2316 | −0.0493 |
| unverifiable | 0.2449 | 0.2362 | −0.0087 |

The classes that lost most (*true*, *misleading*) are those with the most
same-label leaked rows — the fingerprint of memorization rather than
generalization. Both checkpoints and both metric files are preserved in the
repository so the comparison is reproducible.

For context rather than direct comparison: X-FACT's best multilingual system
reaches ~40% F-score using pretrained multilingual transformers *with*
retrieved evidence. A 6.3M-parameter from-scratch model with no pretraining
reaching 0.3313 macro-F1 on four-class LIAR is consistent with the task's
documented difficulty. It is reported as the honest baseline it was designed to
be, not as a competitive result. Classifier inference latency is 13.4 ms per
example on CPU — negligible against the 6–13 s the full pipeline spends on
retrieval and generation.

### 5.2 End-to-end pipeline demonstration (n=10)

Ten curated claims (5 EN / 5 ES, ground-truth labeled, spanning health,
science, geography, history) run through the full pipeline via
`src/model_runner.py`, all in `claude` generation mode:

- **Full pipeline: 9/10 correct.** The single miss returned *unverifiable*
  rather than a wrong label — retrieval found no usable evidence for a
  common-knowledge claim ("el sol sale por el este"), and the generator
  correctly declined to assert beyond its evidence.
- **Classifier alone: 5/10** on the same claims. It labeled two famous myths
  *true* — drinking bleach cures COVID-19, and humans use 10% of their brains —
  and evidence retrieval corrected both.

None of these ten claims appear in any data split (verified), so they are
unaffected by the contamination described in §3.1.

### 5.3 Ablation study

Four conditions over the same ten claims, all in `claude` mode. "No classifier"
means the classifier is not run and Claude's prompt receives a neutral
placeholder signal, since the prompt template always contains a classifier
field:

| Condition | RAG | Classifier | Accuracy | Macro-F1 | Latency p50 |
|---|---|---|---|---|---|
| full_pipeline | ✓ | ✓ | **90%** | 0.4167 | 12.8 s |
| no_rag | ✗ | ✓ | **90%** | 0.4167 | 6.9 s |
| no_classifier | ✓ | ✗ | **90%** | 0.4167 | 7.5 s |
| baseline | ✗ | ✗ | 60% | 0.2143 | 6.9 s |

**Figure 2** — accuracy by condition (bar chart, `outputs/ablation_report.md`).

The three conditions retaining at least one component tie at 90%; removing both
drops accuracy to 60%. On this set retrieval and the classifier signal act as
**redundant substitutes**: either alone recovers full accuracy, and only their
joint absence degrades the system. All four conditions miss the same claim
(hc-004), and the two conditions without retrieval also lose claims the
retrieval-equipped conditions get right — the failure is concentrated in
common-knowledge claims that fact-checkers never write about.

Caveats, stated plainly: n=10, a single nondeterministic run, and a demo set
skewed 8 false / 2 true, so a degenerate "always false" strategy would score
80% and ceiling effects at 90% are real. Accuracy is the readable metric;
macro-F1 is depressed by label classes that barely appear. Latency differences
track retrieval, not classifier cost (13.4 ms).

**Pre-registered latency target not met.** The proposal set p50 ≤ 8 s per
claim. The full pipeline misses it at 12.8 s; every condition without live
retrieval clears it (6.9–7.5 s). Retrieval is the cost, and on this evidence it
buys robustness rather than accuracy — but the target as written is not met.

### 5.4 RAGAS retrieval and generation quality

RAGAS over the same ten claims. Judge LLM: Claude. Embeddings: the same
multilingual MiniLM model the retrieval layer uses, so relevancy is scored in
the system's own retrieval space.

| Metric | Score | Pre-registered target |
|---|---|---|
| Faithfulness | 0.7761 | ≥ 0.75 |
| Answer relevancy | 0.7994 | ≥ 0.75 |

Both clear their thresholds, but two qualifications matter. **First, sample
size:** the proposal pre-registered these metrics on 100 sampled verdicts; this
run covers 10, so these are the pre-registered metrics at a tenth of the
pre-registered sample size. **Second, the means conceal a bimodal
distribution** — three of ten samples fall below the faithfulness target
(0.167, 0.600, 0.667), all Spanish claims where retrieval returned weak or
tangential evidence, while the remainder score at or near 1.0. Faithfulness on
this set is mostly excellent and occasionally poor, and the mean is a fragile
summary of that.

**RQ1 is not fully answered.** RQ1 asks whether RAG-grounded generation is
*more faithful than LLM-only generation*. That requires two RAGAS arms; only
the full-pipeline arm was run. What §5.4 establishes is a threshold check, not
the comparison. The missing arm — RAGAS with retrieval disabled — is the
cheapest outstanding experiment and is listed first in future work.

### 5.5 Cross-language behavior (RQ3, partial)

The identical fabricated claim ("a brand-new Mayan pyramid, hidden from the
public, discovered in the Yucatán") run in both languages under
classifier-fallback mode produced the same label (false) with a 15-point
confidence drop in Spanish (72% EN → 57% ES) — directionally consistent with
the training imbalance, though one claim pair is not a statistical test. The
RAGAS distribution in §5.4 points the same way: all three low-faithfulness
samples are Spanish.

A quantitative EN/ES F1 comparison requires a dedicated Spanish holdout set.
The current test split contains no Spanish examples, so RQ3's first half is
unquantified and its second half — which data source contributes most to
Spanish performance — required a training-data ablation (LIAR only / +MultiFC /
full corpus) that was designed in the proposal and not run. Both halves remain
open.

## 6. Analysis & Discussion

**Grounding works, and the system fails honestly.** RAGAS faithfulness of 0.776
and the ablation's 30-point gap between "some component" and "no components"
both support the core design. More telling: when retrieval comes back empty,
the system prefers *unverifiable* to a confident wrong answer, and the
interface always discloses which generation mode produced a verdict.

**The classifier's failure pattern is the most informative result.** The claims
it mislabeled *true* are famous myths phrased as confident declarations. A small
model trained on claim text alone cannot consult the world; surface-assertive
phrasing is weak evidence of truth, and misinformation is precisely the genre
that exploits this. That retrieval corrected both is the clearest qualitative
illustration this project produced of why evidence matters.

**Redundancy, not dominance (RQ2, revised).** An earlier ablation run —
conducted before the contamination was found, with the contaminated classifier
— showed retrieval outperforming the classifier signal by 20 points, and I
initially wrote that retrieval was the load-bearing component. Re-running
against the clean model overturned that: all three conditions retaining at
least one component tie at 90%. The honest reading is that the two components
are substitutes on this set, and the pipeline's robustness comes from having
two independent paths to a verdict — which is precisely what makes the
classifier-fallback mode viable rather than a degraded stopgap.

RQ2 as written asks whether the classifier-as-reranker improves retrieval over
cosine similarity alone. That was proposal ablation condition (D), "no
reranker," which was designed and not run; four of five conditions were
executed. The reranker remains in the scoring product on the strength of its
design rationale, not on evidence produced here. RQ2 is unanswered.

**Data scarcity shows up where predicted.** The EN/ES confidence gap, the
concentration of low-faithfulness samples in Spanish, and the literature's
independent documentation of the same gap (X-FACT's ~40% ceiling; Quelle et
al.'s 33.1%-vs-8.78% coverage split) all point the same direction. The
from-scratch design makes this visible rather than masking it behind pretrained
multilingual weights — which was its purpose.

**On finding the contamination.** The leak inflated the headline metric by
10.1%, and finding it required specifically hunting for verbatim overlap
between two corpora with no obvious reason to share rows. The lesson
generalizes beyond this project: any work combining an aggregator corpus like
MultiFC with a single-source corpus should audit for overlap before trusting a
held-out metric, because neither dataset's documentation warns of it.

**Known fragilities.** Slightly rewording a claim can flip the classifier's
label, not just its confidence (observed informally, unmeasured). Live web
retrieval makes end-to-end results non-reproducible run to run. Explanation
quality has a known artifact class: in one sample the verdict was correct but
the cited reasoning was imperfect — a "taller than the Eiffel Tower" argument
citing a non-European building — right label, wobbly justification.

## 7. Conclusions & Future Work

VerifAI demonstrates a complete bilingual retrieval-augmented verification
workflow: dataset construction under real scarcity constraints, a from-scratch
bilingual classifier as an honest baseline, cross-lingual retrieval with
credibility-aware reranking, LLM verdict synthesis under a groundedness
constraint, and an evaluation suite that measures the pieces separately.

Findings against the pre-registered questions:

- **RQ1 — partially answered.** Faithfulness (0.776) and relevancy (0.799)
  clear their targets, but the LLM-only comparison arm was never run, and the
  sample size is 10 rather than the pre-registered 100.
- **RQ2 — unanswered.** The reranker-isolation condition was designed and not
  executed. The ablation that *was* run shows retrieval and the classifier are
  redundant substitutes on this set, which was not the expected result.
- **RQ3 — open.** Both halves. No Spanish test split exists, and the
  training-data ablation was not run.
- **Unplanned finding.** 26.7% train/test contamination, root-caused to the
  MultiFC/LIAR overlap, corrected, and quantified at 10.1% inflation of the
  headline metric.

The most transferable lesson: **a generative system is only as trustworthy as
its input discipline and its failure disclosure.** The single biggest quality
lever was not model size but the prompt constraint to use only retrieved
evidence. The single biggest honesty lever was labeling which mode produced
each verdict — and, as it turned out, auditing my own data before trusting my
own number.

Future work, in priority order:

1. **The missing RQ1 arm** — RAGAS with retrieval disabled, at the
   pre-registered n=100. The cheapest outstanding experiment.
2. **A dedicated Spanish holdout set** — every reported F1 is currently an
   English number; RQ3's first half cannot close without this.
3. **Training-data ablation** (LIAR only / +MultiFC / full corpus) — RQ3's
   second half, designed in the proposal and not run.
4. **Reranker isolation** — the fifth ablation condition; RQ2's answer.
5. **A fair pretrained baseline** — fine-tune `xlm-roberta-base` on the same
   labels and splits to price what pretraining buys over the from-scratch
   design.
6. **Scaled, repeated ablation** — larger, label-balanced claim set with
   repeated trials to average out generation nondeterminism and escape the
   ceiling effects visible at n=10.
7. **Near-duplicate de-contamination** — the current audit is exact-match;
   claims reworded across fact-checking sites would survive it, so 0.3313 is a
   tighter bound than 0.3647 but not provably a floor.

## References

Abrajano, M., Nagler, J., Garcia, M., Pope, A., Vidigal, R., & Tucker, J. A.
(2024). How reliance on Spanish-language social media predicts beliefs in false
political narratives amongst Latinos. *PNAS Nexus, 3*(11), pgae442.

Augenstein, I., Lioma, C., Wang, D., Lima, L. C., Hansen, C., Hansen, C., &
Simonsen, J. G. (2019). MultiFC: A real-world multi-domain dataset for
evidence-based fact checking of claims. *Proceedings of EMNLP-IJCNLP 2019.*

Chung, Y.-L., Cobo, A., & Serna, P. (2025). Beyond translation: LLM-based data
generation for multilingual fact-checking. arXiv:2502.15419.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated
evaluation of retrieval augmented generation. *Proceedings of EACL 2024: System
Demonstrations.*

Gómez-Adorno, H., Posadas-Durán, J. P., Enguix, G. B., & Capetillo, C. P.
(2021). Overview of FakeDeS at IberLEF 2021: Fake news detection in Spanish
shared task. *Procesamiento del Lenguaje Natural, 67*, 223–231.

Guo, Z., Schlichtkrull, M., & Vlachos, A. (2022). A survey on automated
fact-checking. *Transactions of the Association for Computational Linguistics,
10*, 178–206.

Gupta, A., & Srikumar, V. (2021). X-FACT: A new benchmark dataset for
multilingual fact checking. *Proceedings of ACL-IJCNLP 2021 (Short Papers).*

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D.
(2020). Retrieval-augmented generation for knowledge-intensive NLP tasks.
*Advances in Neural Information Processing Systems 33.*

Martín, A., Huertas-Tato, J., Huertas-García, Á., Villar-Rodríguez, G., &
Camacho, D. (2022). FacTeR-Check: Semi-automated fact-checking through semantic
similarity and natural language inference. *Knowledge-Based Systems, 251*,
109265.

Quelle, D., Cheng, C. Y., Bovet, A., & Hale, S. A. (2025). Lost in translation:
Using global fact-checks to measure multilingual misinformation prevalence,
spread, and evolution. *EPJ Data Science, 14.*

Reimers, N., & Gurevych, I. (2020). Making monolingual sentence embeddings
multilingual using knowledge distillation. *Proceedings of EMNLP 2020.*

Wang, W. Y. (2017). "Liar, liar pants on fire": A new benchmark dataset for
fake news detection. *Proceedings of ACL 2017.*
