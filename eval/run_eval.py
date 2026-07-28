"""
Eval harness.

Runs every question in eval_set.json through the full graph and records:
- whether the final answer was marked grounded by the validator
- how many retries it took
- latency and cost for the whole run (pulled from logs/run_log.jsonl)
- a simple pass/fail heuristic: questions tagged "out of scope" should
  result in the answer explicitly saying context is insufficient, rather
  than a confident hallucinated answer. This is the key signal that the
  guardrail (not just the demo) actually works.

Usage: python -m eval.run_eval
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import run_query

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

INSUFFICIENCY_PHRASES = ["insufficient", "does not contain", "cannot answer", "not enough information", "context does not"]


def scores_as_out_of_scope(answer: str) -> bool:
    return any(p in answer.lower() for p in INSUFFICIENCY_PHRASES)


def main():
    eval_set = json.loads(EVAL_SET_PATH.read_text())
    results = []

    for item in eval_set:
        state = run_query(item["question"])
        is_out_of_scope_case = "out of scope" in item["expected_topic"]
        answer = state.get("draft_answer", "")
        correctly_declined = scores_as_out_of_scope(answer) if is_out_of_scope_case else None

        result = {
            "id": item["id"],
            "question": item["question"],
            "expected_topic": item["expected_topic"],
            "answer": answer,
            "grounded": state.get("is_grounded"),
            "retries": state.get("_retry_count", 0),
            "is_out_of_scope_case": is_out_of_scope_case,
            "correctly_declined": correctly_declined,
        }
        results.append(result)
        print(f"[{item['id']}] grounded={result['grounded']} retries={result['retries']}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    grounded_rate = sum(1 for r in results if r["grounded"]) / len(results)
    oos_cases = [r for r in results if r["is_out_of_scope_case"]]
    decline_rate = (
        sum(1 for r in oos_cases if r["correctly_declined"]) / len(oos_cases) if oos_cases else None
    )

    print(f"\nGrounded rate: {grounded_rate:.0%}")
    if decline_rate is not None:
        print(f"Correct-decline rate on out-of-scope questions: {decline_rate:.0%}")
    print(f"Full results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
