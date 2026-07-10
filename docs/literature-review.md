# Literature Review

*Supports Milestone 3, Required Component 1: Research and Selection of Methods — Literature Review.*
*Author: Rosalina Torres · VerifAI 2.0 · IE7374, Northeastern University*

This review situates VerifAI's design choices — a from-scratch bilingual classifier, retrieval-augmented verdict generation, and a Claude-based synthesis layer — against the current automated fact-checking literature. It is organized around the specific decisions the system makes, not a general survey of NLP.

---

## 1. Automated Fact-Checking: Task Framing

Guo, Schlichtkrull, and Vlachos (2022) survey the field and decompose automated fact-checking into three stages: claim detection, evidence retrieval, and verdict prediction. VerifAI's three-layer pipeline (intake → retrieval → verdict) follows this decomposition directly, which is a deliberate choice rather than a coincidence — it lets each stage be evaluated and ablated independently (see `evaluation/ablation.py`), matching the survey's observation that most fact-checking failures are attributable to a specific stage rather than the system as a whole.

## 2. Datasets This Project Builds On

Three of VerifAI's four training sources come directly from the fact-checking literature rather than being collected from scratch:

- **LIAR** (Wang, 2017) — 12.8K PolitiFact statements labeled across six veracity classes, introduced as a benchmark specifically because prior fake-news datasets were an order of magnitude smaller. VerifAI maps its six classes down to four (true / false / misleading / unverifiable) and uses LIAR's native test split as the held-out benchmark for Block H, preserving the original paper's train/test boundary rather than re-splitting it.
- **MultiFC** (Augenstein et al., 2019) — 34,924 claims from 26 fact-checking sites with free-text veracity labels and retrieved evidence documents. Its scale is why it dominates VerifAI's English training data (21,673 of 37,608 training rows), but its free-text labels also required the fuzzy-matching normalization step in `training/prepare_data.py` — a direct consequence of the dataset's real-world, non-standardized label taxonomy that the original paper itself flags as a challenge.
- **FakeDeS** (Gómez-Adorno et al., 2021, IberLEF) — a Spanish-language shared task corpus, one of very few labeled Spanish fact-checking datasets available on Hugging Face. Its small size (572 rows) relative to LIAR and MultiFC is itself evidence of the resource gap this project's research questions address (RQ3).

## 3. Retrieval-Augmented Generation for Knowledge-Grounded Verification

Lewis et al. (2020) introduced RAG, combining a neural retriever with a sequence-to-sequence generator so that outputs are grounded in retrieved, non-parametric evidence rather than relying solely on parametric memory — and showed this approach improves factual accuracy specifically on knowledge-intensive tasks, including fact verification. VerifAI's retrieval layer (ChromaDB + Tavily) and verdict layer (Claude synthesizing over retrieved passages) is an application of this same principle: the LLM is explicitly instructed not to answer from parametric knowledge alone ("use only sources provided in Evidence above"), which is the RAG paper's central claim operationalized as a hard prompt constraint. RQ1 in the project proposal — whether RAG-grounded generation produces higher faithfulness than LLM-only generation — is a direct, bilingual replication of the comparison Lewis et al. ran for English fact verification.

## 4. Multilingual Sentence Embeddings for Cross-Lingual Retrieval

Reimers and Gurevych (2020) address a specific failure mode: multilingual transformers like mBERT and XLM-R produce sentence embeddings that are *not* aligned across languages out-of-the-box, meaning semantically identical claims in English and Spanish can land far apart in embedding space. Their knowledge-distillation method — training a multilingual student to match a monolingual teacher's embedding space — is the basis for `paraphrase-multilingual-MiniLM-L12-v2`, the retrieval embedding model VerifAI uses. This is why the model choice in `docs/context/project-overview.md` explicitly warns against substituting `all-MiniLM-L6-v2` (English-only): doing so would reintroduce exactly the cross-lingual misalignment Reimers and Gurevych's method was built to solve, silently degrading Spanish-query retrieval without an obvious error signal.

## 5. The Multilingual and Spanish-Language Gap

This is the literature most directly relevant to the project's motivating claim and RQ3.

- **X-FACT** (Gupta & Srikumar, 2021) benchmarks fact verification across 25 languages and reports the best model reaching only ~40% F-score — establishing empirically that multilingual fact verification is substantially harder than the English-only setting, not just under-resourced. This is the strongest evidence that VerifAI's EN/ES F1 gap (RQ3) is measuring a real, literature-documented phenomenon rather than an artifact of this project's specific implementation.
- **FacTeR-Check** (Martín et al., 2022) is one of the few systems built and validated specifically on Spanish-language misinformation (COVID-19 claims from Spanish social media). Its comparative scarcity in the literature — most fact-checking systems papers evaluate English or treat other languages as an afterthought — is itself informal evidence for the tooling gap this project's Section 1 rationale claims.
- **Beyond Translation** (Chung, Cobo, & Serna, 2025) tackles this gap directly, introducing MultiSynFact (2.2M synthetic claim-source pairs for Spanish, German, and English) specifically because real-world Spanish fact-checking training data is scarce. VerifAI's own synthetic ES corpus (475 Claude-generated, author-verified examples) is a smaller-scale, manually quality-controlled instance of the same underlying strategy: when labeled Spanish data doesn't exist at sufficient volume, generate and verify it rather than train on English data alone.

## How This Project Positions Itself

VerifAI does not introduce a new fact-checking paradigm — RAG-based verification, multilingual sentence retrieval, and classifier-as-reranker are all established techniques individually. Its contribution is combining them for a language pair (EN/ES) and resource level (course-project scale, not industrial) that the literature above shows is still underserved: X-FACT establishes the multilingual gap empirically, FacTeR-Check and Beyond Translation show Spanish-specific work is recent and sparse, and none of the reviewed systems use a from-scratch bilingual classifier in a dual role as both first-pass classifier and RAG evidence reranker — the design implemented in `app/pipeline/verdict.py` (`combined_score = cosine_similarity × credibility_score × reranker_score`). Note: `docs/context/architecture/decisions/adr-001-xlm-roberta-as-proxy.md` documents an earlier version of this decision (the XLM-RoBERTa sentiment proxy) that Block G replaced with the trained VerifAIClassifier; that ADR still needs a superseding note, tracked separately from this review.

---

## References

Augenstein, I., Lioma, C., Wang, D., Lima, L. C., Hansen, C., Hansen, C., & Simonsen, J. G. (2019). MultiFC: A Real-World Multi-Domain Dataset for Evidence-Based Fact Checking of Claims. *Proceedings of EMNLP-IJCNLP 2019.* https://aclanthology.org/D19-1475/

Chung, Y.-L., Cobo, A., & Serna, P. (2025). Beyond Translation: LLM-Based Data Generation for Multilingual Fact-Checking. arXiv:2502.15419. https://arxiv.org/abs/2502.15419

Gómez-Adorno, H., Posadas-Durán, J. P., Enguix, G. B., & Capetillo, C. P. (2021). Overview of FakeDeS at IberLEF 2021: Fake News Detection in Spanish Shared Task. *Procesamiento del Lenguaje Natural, 67*, 223–231.

Guo, Z., Schlichtkrull, M., & Vlachos, A. (2022). A Survey on Automated Fact-Checking. *Transactions of the Association for Computational Linguistics, 10*, 178–206. https://aclanthology.org/2022.tacl-1.11/

Gupta, A., & Srikumar, V. (2021). X-FACT: A New Benchmark Dataset for Multilingual Fact Checking. *Proceedings of ACL-IJCNLP 2021 (Short Papers).* https://aclanthology.org/2021.acl-short.86/

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems 33 (NeurIPS 2020).*

Martín, A., Huertas-Tato, J., Huertas-García, Á., Villar-Rodríguez, G., & Camacho, D. (2022). FacTeR-Check: Semi-automated fact-checking through semantic similarity and natural language inference. arXiv:2110.14532. https://arxiv.org/abs/2110.14532

Reimers, N., & Gurevych, I. (2020). Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation. *Proceedings of EMNLP 2020.* arXiv:2004.09813.

Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017).* https://aclanthology.org/P17-2067/
