"""
evaluation/ragas_eval.py
Evaluates retrieval/generation quality with RAGAS metrics (faithfulness,
answer relevancy) over the curated demo claim set.

Judge/embeddings configuration (no OpenAI dependency):
  - Judge LLM: Claude via the Anthropic API (same provider the pipeline
    itself uses for verdict synthesis).
  - Embeddings: paraphrase-multilingual-MiniLM-L12-v2 run locally — the
    same sentence-embedding model the retrieval layer uses, so relevancy
    is scored in the same embedding space the system retrieves in.

Targets: faithfulness >= 0.75, answer_relevancy >= 0.75

Usage:
    python evaluation/ragas_eval.py
"""
import sys, os, json, types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- Compatibility shim -----------------------------------------------------
# ragas 0.4.x still imports a legacy module removed from langchain-community
# 0.4.x (ChatVertexAI). We don't use Vertex; stub the module so ragas imports.
_VERTEX_MOD = "langchain_community.chat_models.vertexai"
if _VERTEX_MOD not in sys.modules:
    try:
        __import__(_VERTEX_MOD)
    except ModuleNotFoundError:
        _stub = types.ModuleType(_VERTEX_MOD)
        _stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[_VERTEX_MOD] = _stub
# -----------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

from app.pipeline.intake import extract_claim
from app.pipeline.retrieval import retrieve_evidence
from app.pipeline.verdict import generate_verdict

JUDGE_MODEL = "claude-sonnet-4-5"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def collect_rows(claims: list[dict]) -> list[dict]:
    """Run each claim through the full live pipeline and collect RAGAS rows."""
    rows = []
    for item in claims:
        try:
            intake = extract_claim(item["claim"])
            evidence = retrieve_evidence(
                intake["extracted_assertion"], intake["language"]
            )
            result = generate_verdict(
                extracted_assertion=intake["extracted_assertion"],
                evidence=evidence,
                language=intake["language"],
            )
            rows.append({
                "user_input": item["claim"],
                "response": result["explanation"],
                "retrieved_contexts": (
                    [e["passage"] for e in evidence] or ["No evidence retrieved."]
                ),
                "reference": item.get("ground_truth", ""),
            })
            print(f"  collected {item['claim_id']} "
                  f"({len(evidence)} evidence, mode={result.get('generation_mode')})")
        except Exception as e:
            print(f"  Skipping claim {item['claim_id']}: {e}")
    return rows


def run_ragas(sample_path: str = None):
    if sample_path is None:
        sample_path = os.path.join(
            os.path.dirname(__file__), "../data/sample_claims.json"
        )
    with open(sample_path) as f:
        claims = json.load(f)

    print(f"Collecting pipeline outputs for {len(claims)} claims...")
    rows = collect_rows(claims)
    if not rows:
        print("No claims evaluated.")
        return

    judge = LangchainLLMWrapper(ChatAnthropic(model=JUDGE_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )

    print(f"Scoring {len(rows)} rows with RAGAS "
          f"(judge={JUDGE_MODEL}, embeddings={EMBED_MODEL})...")
    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    output = {
        "n_evaluated": len(rows),
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBED_MODEL,
        "faithfulness": round(float(df["faithfulness"].mean()), 4),
        "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
        "per_sample": [
            {
                "claim": rows[i]["user_input"][:80],
                "faithfulness": round(float(df["faithfulness"][i]), 4),
                "answer_relevancy": round(float(df["answer_relevancy"][i]), 4),
            }
            for i in range(len(rows))
        ],
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "../outputs/results_ragas.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== RAGAS Results (n={output['n_evaluated']}) ===")
    print(f"Faithfulness:     {output['faithfulness']}  (target >= 0.75)")
    print(f"Answer Relevancy: {output['answer_relevancy']}  (target >= 0.75)")
    print(f"Results saved to outputs/results_ragas.json")


if __name__ == "__main__":
    run_ragas()
