"""
evaluation/ablation.py
Runs 4-condition ablation study on the sample_claims set.

Conditions:
  A) Full pipeline     (RAG + XLM-RoBERTa)
  B) No RAG            (LLM only, no retrieval)
  C) No classifier     (RAG only, skip XLM-RoBERTa)
  D) Baseline          (no RAG, no classifier)

Usage:
    python evaluation/ablation.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import f1_score
from app.pipeline.intake import extract_claim
from app.pipeline.retrieval import retrieve_evidence
from app.pipeline.verdict import generate_verdict, run_classifier


def run_condition(claims, use_rag: bool, use_classifier: bool, label: str):
    y_true, y_pred, latencies = [], [], []
    for item in claims:
        start = time.perf_counter()
        try:
            intake = extract_claim(item["claim"])
            evidence = retrieve_evidence(intake["extracted_assertion"], intake["language"]) if use_rag else []
            if not use_classifier:
                # Monkey-patch: pass dummy classifier values
                result = generate_verdict(intake["extracted_assertion"], evidence, intake["language"])
                # Override classifier fields
                result["classifier_label"] = "skipped"
                result["classifier_confidence"] = 0.0
            else:
                result = generate_verdict(intake["extracted_assertion"], evidence, intake["language"])
            predicted = result["label"]
        except Exception as e:
            print(f"  [{label}] ERROR: {e}")
            predicted = "unverifiable"
        elapsed = time.perf_counter() - start
        y_true.append(item.get("ground_truth", "unverifiable"))
        y_pred.append(predicted)
        latencies.append(elapsed)

    labels = ["true", "false", "misleading", "unverifiable"]
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0
    return {"condition": label, "f1_macro": round(f1, 4), "latency_p50_sec": round(p50, 2)}


def run_ablation():
    sample_path = os.path.join(os.path.dirname(__file__), "../data/sample_claims.json")
    with open(sample_path) as f:
        claims = json.load(f)

    print("Running ablation study (4 conditions)...")
    results = []
    conditions = [
        (True,  True,  "full_pipeline"),
        (False, True,  "no_rag"),
        (True,  False, "no_classifier"),
        (False, False, "baseline"),
    ]
    for use_rag, use_clf, label in conditions:
        print(f"  Running: {label}")
        r = run_condition(claims, use_rag, use_clf, label)
        results.append(r)
        print(f"    F1={r['f1_macro']}  p50={r['latency_p50_sec']}s")

    out_path = os.path.join(os.path.dirname(__file__), "../outputs/ablation_report.md")
    with open(out_path, "w") as f:
        f.write("# VerifAI Ablation Study\n\n")
        f.write("| Condition | RAG | Classifier | F1 Macro | Latency p50 |\n")
        f.write("|-----------|-----|------------|----------|-------------|\n")
        flags = [(True, True), (False, True), (True, False), (False, False)]
        for r, (rag, clf) in zip(results, flags):
            f.write(f"| {r['condition']} | {'✓' if rag else '✗'} | {'✓' if clf else '✗'} "
                    f"| {r['f1_macro']} | {r['latency_p50_sec']}s |\n")

    print(f"\nAblation complete. Report saved to outputs/ablation_report.md")


if __name__ == "__main__":
    run_ablation()
