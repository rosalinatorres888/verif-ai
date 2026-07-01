"""
evaluation/benchmark_liar.py
Runs the full VerifAI pipeline against the LIAR test split.
Target: F1 macro >= 0.72

Run on OOD for full evaluation (1267 test claims).

Usage:
    python evaluation/benchmark_liar.py
    python evaluation/benchmark_liar.py --limit 50   # quick dev run
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from app.pipeline.intake import extract_claim
from app.pipeline.retrieval import retrieve_evidence
from app.pipeline.verdict import generate_verdict

# LIAR has 6 classes — map to our 4
LIAR_LABEL_MAP = {
    "true": "true",
    "mostly-true": "true",
    "half-true": "misleading",
    "barely-true": "misleading",
    "false": "false",
    "pants-fire": "false"
}


def run_benchmark(limit: int = None):
    print("Loading LIAR dataset...")
    dataset = load_dataset("liar", split="test")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    y_true, y_pred, latencies = [], [], []

    for i, item in enumerate(dataset):
        claim_text = item["statement"]
        ground_truth = LIAR_LABEL_MAP.get(item["label"], "unverifiable")

        start = time.perf_counter()
        try:
            intake = extract_claim(claim_text)
            evidence = retrieve_evidence(intake["extracted_assertion"], intake["language"])
            result = generate_verdict(intake["extracted_assertion"], evidence, intake["language"])
            predicted = result["label"]
        except Exception as e:
            print(f"  ERROR on claim {i}: {e}")
            predicted = "unverifiable"
        elapsed = time.perf_counter() - start

        y_true.append(ground_truth)
        y_pred.append(predicted)
        latencies.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(dataset)} claims...")

    labels = ["true", "false", "misleading", "unverifiable"]
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]

    output = {
        "run_date": time.strftime("%Y-%m-%d"),
        "condition": "full_pipeline",
        "n_claims": len(dataset),
        "f1_macro": round(f1, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "latency_p50_sec": round(p50, 2),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0)
    }

    out_path = os.path.join(os.path.dirname(__file__), "../outputs/results_liar.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== LIAR Benchmark Results ===")
    print(f"F1 macro:   {f1:.4f}  (target >= 0.72)")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"Latency p50: {p50:.2f}s")
    print(f"\nResults saved to outputs/results_liar.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims (for dev runs)")
    args = parser.parse_args()
    run_benchmark(limit=args.limit)
