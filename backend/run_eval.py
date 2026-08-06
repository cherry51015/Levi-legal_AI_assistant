"""
Run the full eval suite against Levi and produce eval_report.json + a summary
you can paste straight into the README.

Usage:
    python run_eval.py eval_dataset.jsonl

Assumes Levi's backend exposes ask_gemini(prompt, document) from llm.py,
matching the current README's architecture section.
"""

import json
import sys
import time
import statistics
from rubric import judge_answer

# adjust this import path to match where llm.py actually lives in backend/
from llm import ask_gemini


def load_dataset(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(dataset_path):
    rows = load_dataset(dataset_path)
    results = []

    for row in rows:
        start = time.perf_counter()
        answer = ask_gemini(row["query"], row["document_context"])
        latency = round(time.perf_counter() - start, 3)

        score = judge_answer(row["document_context"], row["query"], answer)

        results.append({
            "id": row["id"],
            "category": row["category"],
            "query": row["query"],
            "answer": answer,
            "latency_seconds": latency,
            "faithfulness": score["faithfulness"],
            "completeness": score["completeness"],
            "fabricated": score["fabricated"],
            "reasoning": score["reasoning"],
        })
        print(f"{row['id']} [{row['category']}] faithfulness={score['faithfulness']} "
              f"completeness={score['completeness']} fabricated={score['fabricated']} "
              f"latency={latency}s")

    with open("eval_report.json", "w") as f:
        json.dump(results, f, indent=2)

    summarize(results)


def summarize(results):
    faith = [r["faithfulness"] for r in results]
    comp = [r["completeness"] for r in results]
    latencies = sorted(r["latency_seconds"] for r in results)
    fabricated_count = sum(1 for r in results if r["fabricated"])
    n = len(results)

    # "answer relevance" as used in the resume = avg of faithfulness+completeness
    # scaled to a percentage. Document this formula in the README so it's
    # reproducible, not a mystery number.
    relevance_pct = round(((statistics.mean(faith) + statistics.mean(comp)) / 2) / 5 * 100, 1)
    avg_latency = statistics.mean(latencies)
    p95_latency = latencies[int(n * 0.95) - 1] if n >= 20 else latencies[-1]

    print("\n--- SUMMARY ---")
    print(f"n = {n}")
    print(f"avg faithfulness: {statistics.mean(faith):.2f} / 5")
    print(f"avg completeness: {statistics.mean(comp):.2f} / 5")
    print(f"fabrication rate: {fabricated_count}/{n} ({fabricated_count/n*100:.1f}%)")
    print(f"answer relevance (resume metric): {relevance_pct}%")
    print(f"avg latency: {avg_latency:.2f}s | p95 latency: {p95_latency:.2f}s")

    with open("eval_summary.md", "w") as f:
        f.write("# Levi Eval Results\n\n")
        f.write(f"- Queries evaluated: {n}\n")
        f.write(f"- Avg faithfulness: {statistics.mean(faith):.2f}/5\n")
        f.write(f"- Avg completeness: {statistics.mean(comp):.2f}/5\n")
        f.write(f"- Fabrication rate: {fabricated_count}/{n} ({fabricated_count/n*100:.1f}%)\n")
        f.write(f"- **Answer relevance: {relevance_pct}%**\n")
        f.write(f"- Avg latency: {avg_latency:.2f}s | p95 latency: {p95_latency:.2f}s\n\n")
        f.write("Formula: ((mean(faithfulness) + mean(completeness)) / 2) / 5 * 100\n")
        f.write("Judge model: llama-3.3-70b-versatile via Groq (kept separate from Gemini, "
                 "the answering model, to avoid self-preference bias)\n")
        f.write("Latency measured client-side with time.perf_counter() around the ask_gemini "
                 "call — no cloud monitoring dependency.\n")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "eval_dataset_template.jsonl")
