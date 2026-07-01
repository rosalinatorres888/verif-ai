"""
evaluation/ragas_eval.py
Evaluates retrieval quality using RAGAS metrics on 100 sampled verdicts.
Targets: faithfulness >= 0.75, answer_relevancy >= 0.75

Usage:
    python evaluation/ragas_eval.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from app.pipeline.intake import extract_claim
from app.pipeline.retrieval import retrieve_evidence
from app.pipeline.verdict import generate_verdict


def run_ragas(sample_path: str = None, n: int = 100):
    if sample_path is None:
        sample_path = os.path.join(os.path.dirname(__file__), "../data/sample_claims.json")

    with open(sample_path) as f:
        claims = json.load(f)

    claims = claims[:n]
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in claims:
        try:
            intake = extract_claim(item["claim"])
            evidence = retrieve_evidence(intake["extracted_assertion"], intake["language"])
            result = generate_verdict(intake["extracted_assertion"], evidence, intake["language"])

            rows["question"].append(item["claim"])
            rows["answer"].append(result["explanation"])
            rows["contexts"].append([e["passage"] for e in evidence] or ["No evidence retrieved."])
            rows["ground_truth"].append(item.get("ground_truth", ""))
        except Exception as e:
            print(f"  Skipping claim {item['claim_id']}: {e}")

    if not rows["question"]:
        print("No claims evaluated.")
        return

    dataset = Dataset.from_dict(rows)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    output = {
        "n_evaluated": len(rows["question"]),
        "faithfulness": round(result["faithfulness"], 4),
        "answer_relevancy": round(result["answer_relevancy"], 4)
    }

    out_path = os.path.join(os.path.dirname(__file__), "../outputs/results_ragas.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== RAGAS Results ===")
    print(f"Faithfulness:     {output['faithfulness']}  (target >= 0.75)")
    print(f"Answer Relevancy: {output['answer_relevancy']}  (target >= 0.75)")
    print(f"Results saved to outputs/results_ragas.json")


if __name__ == "__main__":
    run_ragas()
